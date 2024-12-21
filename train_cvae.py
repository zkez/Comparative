import os
import torch
import random
from tqdm import tqdm
from torch.nn import functional as F
from generative.losses import PatchAdversarialLoss, PerceptualLoss
from generative.networks.nets import PatchDiscriminator
from os.path import join as j
from torchvision.utils import make_grid, save_image

import sys
sys.path.append('./')
from model.conditional_vae import AutoencoderKL
from model.conditional_encoder import MaskConditionEncoder
from config.config_cvae import Config
from utils.common import get_parameters, save_config_to_yaml, get_dataloader


def main():
    # 保存配置并获取配置字典
    cf = save_config_to_yaml(Config, Config.project_dir)
    
    # 获取设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 获取数据加载器
    train_dataloader, val_dataloader = get_dataloader(Config)

    # 初始化模型
    up_and_down = Config.up_and_down
    attention_levels = (False, ) * len(up_and_down)
    vae = AutoencoderKL(
        spatial_dims=3, 
        in_channels=Config.in_channels, 
        out_channels=Config.out_channels, 
        num_channels=Config.up_and_down, 
        latent_channels=4,
        num_res_blocks=Config.num_res_layers, 
        attention_levels=attention_levels
    )
    if len(Config.vae_path):
        vae.load_state_dict(torch.load(Config.vae_path, map_location=device))

    con_encoder = MaskConditionEncoder(
        1, Config.up_and_down[0], 
        Config.up_and_down[-1], 
        stride=4
    )
    
    discriminator = PatchDiscriminator(
        spatial_dims=3, num_channels=64, 
        in_channels=Config.out_channels, out_channels=1
    )
    if len(Config.dis_path):
        discriminator.load_state_dict(torch.load(Config.dis_path, map_location=device))

    perceptual_loss = PerceptualLoss(spatial_dims=3, network_type="vgg")
    
    # 将模型移动到设备
    vae = vae.to(device)
    con_encoder = con_encoder.to(device)
    discriminator = discriminator.to(device)
    perceptual_loss = perceptual_loss.to(device)
    vae.requires_grad_(False).eval()

    # 定义损失函数和优化器
    adv_loss = PatchAdversarialLoss(criterion="least_squares")
    adv_weight = 0.01
    perceptual_weight = 0.001
    
    optimizer_g = torch.optim.Adam(params=con_encoder.parameters(), lr=1e-4)
    optimizer_d = torch.optim.Adam(params=discriminator.parameters(), lr=1e-4)

    # 训练参数
    val_interval = Config.val_inter
    save_interval = Config.save_inter
    autoencoder_warm_up_n_epochs = 0 if (len(Config.vae_path) and len(Config.dis_path)) else Config.autoencoder_warm_up_n_epochs

    global_step = 0
    for epoch in range(Config.num_epochs):
        # 初始化进度条
        progress_bar = tqdm(total=len(train_dataloader), desc=f"Epoch {epoch+1}", disable=False)
        
        con_encoder.train()
        discriminator.train()
        for step, batch in enumerate(train_dataloader):
            t1, t2 = batch[1].to(device), batch[2].to(device)
            optimizer_g.zero_grad(set_to_none=True)
            condition_im = con_encoder(t1)
            reconstruction, _, _ = vae(t2, condition_im=condition_im)

            # 计算重建损失
            recons_loss = F.mse_loss(reconstruction.float(), t2.float())

            # 计算感知损失
            p_loss = perceptual_loss(reconstruction.float(), t2.float())
            loss_g = recons_loss + perceptual_weight * p_loss

            # 如果超过预热期，计算对抗损失
            if epoch+1 > autoencoder_warm_up_n_epochs:
                logits_fake = discriminator(reconstruction.contiguous().float())[-1]
                generator_loss = adv_loss(logits_fake, target_is_real=True, for_discriminator=False)
                loss_g += adv_weight * generator_loss

            # 反向传播和优化
            loss_g.backward()
            optimizer_g.step()

            if epoch+1 > autoencoder_warm_up_n_epochs:
                optimizer_d.zero_grad(set_to_none=True)

                logits_fake = discriminator(reconstruction.contiguous().detach())[-1]
                loss_d_fake = adv_loss(logits_fake, target_is_real=False, for_discriminator=True)
                logits_real = discriminator(t2.contiguous().detach())[-1]
                loss_d_real = adv_loss(logits_real, target_is_real=True, for_discriminator=True)
                discriminator_loss = (loss_d_fake + loss_d_real) * 0.5

                loss_d = adv_weight * discriminator_loss

                loss_d.backward()
                optimizer_d.step()

            # 更新进度条
            progress_bar.update(1)
            logs = {
                "gen_loss": loss_g.detach().item(), 
                "dis_loss": loss_d.detach().item() if epoch+1 > autoencoder_warm_up_n_epochs else 0, 
                "pp_loss": p_loss.detach().item(), 
                "adv_loss": generator_loss.detach().item() if epoch+1 > autoencoder_warm_up_n_epochs else 0
            }
            progress_bar.set_postfix(**logs)
            global_step += 1

        progress_bar.close()

        # 验证
        if (epoch + 1) % val_interval == 0 or epoch == Config.num_epochs - 1:
            con_encoder.eval()
            total_mse_loss = 0.0
            with torch.no_grad():
                for batch_idx, batch in enumerate(val_dataloader):
                    t0, t1, t2 = batch[0].to(device), batch[1].to(device), batch[2].to(device)

                    condition_im = con_encoder(t1)
                    val_recon, _, _ = vae(t2, condition_im=condition_im)
                    mse_loss = F.mse_loss(val_recon, t2)
                    total_mse_loss += mse_loss

                average_mse_loss = total_mse_loss / len(val_dataloader)
                print(f'Epoch {epoch + 1}, Average MSE Loss: {average_mse_loss.item()}')
                del average_mse_loss, total_mse_loss, mse_loss
                
        # 保存模型
        if (epoch + 1) % save_interval == 0 or epoch == Config.num_epochs - 1:
            gen_path = j(Config.project_dir, 'gen_save')
            dis_path = j(Config.project_dir, 'dis_save')
            os.makedirs(gen_path, exist_ok=True)
            os.makedirs(dis_path, exist_ok=True)
            torch.save(vae.state_dict(), j(gen_path, 'vae.pth'))
            torch.save(discriminator.state_dict(), j(dis_path, 'dis.pth'))


if __name__ == '__main__':
    main()

from dataclasses import dataclass


@dataclass
class Config():
    csv_path = '/home/zk/MICCAI/newmainroi.csv'  
    data_dir = '/home/zk/MICCAI/roiresize'
    spilt_size = 0.2
    
    num_epochs = 2000
    train_batch_size = 1
    eval_batch_size = 1 

    up_and_down = (128, 256, 512)
    in_channels = 1
    out_channels = 1
    num_res_layers = 2

    vae_path = ''
    dis_path = ''
    project_dir = '/home/zk/MICCAI/ZK/25-2/Comparative/save'

    val_inter = 80
    save_inter = 100

    autoencoder_warm_up_n_epochs = 100

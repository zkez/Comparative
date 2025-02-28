from torch import nn
import torch

def zero_module(module):
    for p in module.parameters():
        nn.init.zeros_(p)
    return module

class MaskConditionEncoder(nn.Module):
    """
    用于 AsymmetricAutoencoderKL 的 MaskConditionEncoder
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int = 192,
        res_ch: int = 768,
        stride: int = 16,
    ) -> None:
        super().__init__()

        channels = []
        while stride > 1:
            stride = stride // 2
            in_ch_ = out_ch * 2
            if out_ch > res_ch:
                out_ch = res_ch
            if stride == 1:
                in_ch_ = res_ch
            channels.append((in_ch_, out_ch))
            out_ch *= 2

        out_channels = []
        for _in_ch, _out_ch in channels:
            out_channels.append(_out_ch)
        out_channels.append(channels[-1][0])

        layers = []
        adjust_sh = []
        gated_cnn = []
        in_ch_ = in_ch
        for l in range(len(out_channels)):
            out_ch_ = out_channels[l]
            if l == 0:
                layers.append(nn.Conv3d(in_ch_, out_ch_, kernel_size=3, stride=1, padding=1))
                # gated_cnn.append(nn.Conv3d(out_ch_, out_ch_, kernel_size=3, stride=1, padding=1))
            else:
                layers.append(nn.Conv3d(in_ch_, out_ch_, kernel_size=4, stride=2, padding=1))
                # gated_cnn.append(nn.Conv3d(out_ch_, out_ch_, kernel_size=3, stride=2, padding=1))
            gated_cnn.append(nn.Conv3d(out_ch_, out_ch_, kernel_size=3, stride=1, padding=1))
            adjust_sh.append(nn.Conv3d(in_ch_, 1, kernel_size=1, stride=1, padding=0))
            
            in_ch_ = out_ch_

        self.layers = nn.Sequential(*layers)
        self.adjust_chs = nn.Sequential(*adjust_sh)
        self.gated_cnns = nn.Sequential(*gated_cnn)
        self.layers, self.adjust_chs, self.gated_cnns = zero_module(self.layers), zero_module(self.adjust_chs), zero_module(self.gated_cnns)

    def forward(self, x: torch.Tensor, mask=None) -> torch.Tensor:
        r"""MaskConditionEncoder 的前向传播方法。"""
        out = {}
        for l in range(len(self.layers)):
            layer = self.layers[l]
            gated_cnn = self.gated_cnns[l]
            adjust_sh = self.adjust_chs[l]
            # gated_tensor = torch.sigmoid(gated_cnn(adjust_sh(x)))
            # x = layer(x)
            x = layer(x)
            gated_tensor = torch.sigmoid(gated_cnn(x))
            out[str(tuple(x.shape))] = torch.mul(x, gated_tensor)
            x = torch.relu(x)
        return out


if __name__ == "__main__":
    device = "cuda:1"
    model = MaskConditionEncoder(in_ch=3, out_ch=128, res_ch=512, stride=4).to(device)
    a = torch.randn((4, 3, 512, 512)).to(device)
    out = model(a)
    for key in out:
        print(key)
        
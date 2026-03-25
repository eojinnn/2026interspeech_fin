import torch
import torch.nn as nn
import torch.nn.functional as F

from .resnet import resnet18_nopool
from .conformer import ConformerBlock


class MapTo10_F64_T100(nn.Module):
    """
    Input : (B, 4, 32, 1500) = (B, C, F, T)
    Output: (B, 10, 64, 100) = (B, C_out, F_out, T_out)

    Steps:
      1) channel mixing conv2d: 4 -> hidden
      2) freq upsample: 32 -> 64 via ConvTranspose2d (stride=2 on F)
      3) time downsample: 1500 -> 100 via Conv2d stride=15 on T
      4) project to 10 channels
    """
    def __init__(self, hidden=10, leak=0.2):
        super().__init__()
        self.act = nn.LeakyReLU(leak, inplace=True)

        # 1) 채널 간 특징 학습 (C=4를 섞음): (B,4,32,1500) -> (B,hidden,32,1500)
        self.mix = nn.Conv2d(
            in_channels=4, out_channels=hidden,
            kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),
            bias=True
        )

        # 3) 시간 다운샘플: T 1500 -> 100 (정확히 /15)
        # outT = floor((inT + 2p - k)/s) + 1
        #     = floor((1500 + 0 - 15)/15) + 1 = 100
        self.down_t = nn.Conv2d(
            in_channels=hidden, out_channels=hidden,
            kernel_size=(3, 15), stride=(1, 15), padding=(1, 0),
            bias=True
        )

        # 4) 출력 채널 10으로
        self.to10 = nn.Conv2d(hidden, 10, kernel_size=1, bias=True)

    def forward(self, x):
        # x: (B,4,64,1500)
        h = self.act(self.mix(x))     # (B,hidden,64,1500)
        h = self.act(self.down_t(h))  # (B,hidden,64,100)
        y = self.to10(h)              # (B,10,64,100)
        return y
    
class DownConvDecimate5x(nn.Module):
    """
    Input : (B, 4, 48000)
    Step  : reshape -> (B*4, 48000, 1)  (채널=1, 길이=48000)
            then 5x [conv1d(stride=1) -> decimate(::2)]
    Output: (B, 4, 1500, 32)
    """
    def __init__(self, out_ch=64, kernel_size=15, padding="same", leak=0.2):
        super().__init__()
        assert padding in ["same", "valid"]

        pad = kernel_size // 2 if padding == "same" else 0
        self.act = nn.LeakyReLU(leak, inplace=True)

        # 첫 conv: 1 -> out_ch, 이후 conv: out_ch -> out_ch
        convs = []
        for i in range(5):
            in_ch = 1 if i == 0 else out_ch
            convs.append(nn.Conv1d(in_ch, out_ch, kernel_size, stride=1, padding=pad, bias=True))
        self.convs = nn.ModuleList(convs)
        self.map = MapTo10_F64_T100(hidden=10)

    def forward(self, x):
        """
        x: (B,4,48000)
        return: (B,4,1500,32)
        """
        B, C, T = x.shape
        
        # (B,4,48000) -> (B*4,48000,1)
        h = x.reshape(B * C, 1, T)

        for conv in self.convs:
            h = conv(h)        # (B*4, 32, L)
            h = self.act(h)
            h = h[..., ::2]    # decimate (time) by 2 torch.Size([128, 32, 1500])

        # L: 48000 -> 24000 -> 12000 -> 6000 -> 3000 -> 1500
        # (B*4,32,1500) -> (B,4,1500,32)
        h = h.reshape(B, C, h.shape[1], h.shape[2]) #(B,4,32,1500)
        out = self.map(h) 
        out = out.permute(0,1,3,2).contiguous()
        return out

    
def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class ResnetConformer_seddoa_nopool_2023(nn.Module):
    def __init__(self, in_channel, in_dim, out_dim):
        super().__init__()
        self.resnet = resnet18_nopool(in_channel=in_channel)
        embedding_dim = in_dim // 32 * 256
        encoder_dim = 256
        self.input_projection = nn.Sequential(
            nn.Linear(embedding_dim, encoder_dim),
            nn.Dropout(p=0.05),
        )
        num_layers = 8
        self.frontend = DownConvDecimate5x()
        self.conformer_layers = nn.ModuleList(
            [ConformerBlock(
                dim = encoder_dim,
                dim_head = 32,
                heads = 8,
                ff_mult = 2,
                conv_expansion_factor = 2,
                conv_kernel_size = 7,
                attn_dropout = 0.1,
                ff_dropout = 0.1,
                conv_dropout = 0.1
            ) for _ in range(num_layers)]
        )
        self.t_pooling = nn.MaxPool1d(kernel_size=5)
        self.sed_out_layer = nn.Sequential(
            nn.Linear(encoder_dim, encoder_dim),
            nn.LeakyReLU(),
            nn.Linear(encoder_dim, 13),
            nn.Sigmoid()
        )
        self.out_layer = nn.Sequential(
            nn.Linear(encoder_dim, encoder_dim),
            nn.LeakyReLU(),
            nn.Linear(encoder_dim, out_dim),
            nn.Tanh()
        ) 

    def forward(self, x):
        B, M, T, L = x.shape
        N = T * L

        x = x.reshape(B,M,N).contiguous() # (B, 4, 48000)
        x = self.frontend(x) # (B, 4, 64, 500)
        # pdb.set_trace() # [32, 7, 500, 64]
        conv_outputs = self.resnet(x)
        N,C,T,W = conv_outputs.shape
        conv_outputs = conv_outputs.permute(0,2,1,3).reshape(N, T, C*W)

        conformer_outputs = self.input_projection(conv_outputs)
        #conformer_outputs = conv_outputs
        for layer in self.conformer_layers:
            conformer_outputs = layer(conformer_outputs)
        outputs = conformer_outputs.permute(0,2,1)
        outputs = self.t_pooling(outputs)
        outputs = outputs.permute(0,2,1)
        sed = self.sed_out_layer(outputs)
        doa = self.out_layer(outputs)
        # dist = self.dist_out_layer(outputs)
        pred = torch.cat((sed, doa), dim=-1) # [32, 100, 52]
        # pdb.set_trace()
        return pred
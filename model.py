import torch
import torch.nn as nn
#import torch.nn.functional as F
import numpy as np
import math
from data_loader import get_loader

# NOTE: GLU, IN affine true, up/down sample bias false, embedding-> linear, convtranspose2d
# repeat(1,2,1,1) expand(-1,-1,h,w) only expand on size=1, ModuleList, PixelShuffle change channel, ReLU, stride ceiling floor

class GLU2d(nn.Module):
    def __init__(self, dim=1):
        super(GLU2d, self).__init__()
        self.glu=nn.GLU(dim=1)

    # assume NCHW
    # TODO: check output dim
    def forward(self, x):
        x=x.repeat(1, 2, 1, 1)
        return self.glu(x)

class DownSample_Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(DownSample_Block, self).__init__()

        self.main = nn.Sequential(nn.Conv2d(in_channels=in_channels,
                                                 out_channels=out_channels,
                                                 kernel_size=kernel_size,
                                                 stride=stride,
                                                 padding=padding,
                                                 bias=False),
                                       nn.InstanceNorm2d(num_features=out_channels,
                                                         track_running_stats=True,
                                                         affine=True))
        self.main_gate = nn.Sequential(nn.Conv2d(in_channels=in_channels,
                                                 out_channels=out_channels,
                                                 kernel_size=kernel_size,
                                                 stride=stride,
                                                 padding=padding,
                                                 bias=False),
                                       nn.InstanceNorm2d(num_features=out_channels,
                                                         track_running_stats=True,
                                                         affine=True))
        self.glu=nn.GLU(dim=1)

    def forward(self, input):
        #GLU
        return self.glu(
                torch.cat(
                    (self.main(input), self.main_gate(input)), 1)
                )

class UpSample_Block(nn.Module):
    # stride is normally 1
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, upscale_factor=2):
        super(UpSample_Block, self).__init__()

        self.main = nn.Sequential(nn.Conv2d(in_channels=in_channels,
                                                 out_channels=out_channels,
                                                 kernel_size=kernel_size,
                                                 stride=stride,
                                                 padding=padding),
                                       nn.PixelShuffle(upscale_factor))
                                       # removed from CyclaGANVC2
                                       #nn.InstanceNorm2d(num_features=out_channels,
                                       #                  affine=True))
        self.main_gate = nn.Sequential(nn.Conv2d(in_channels=in_channels,
                                                 out_channels=out_channels,
                                                 kernel_size=kernel_size,
                                                 stride=stride,
                                                 padding=padding),
                                       nn.PixelShuffle(upscale_factor))
        self.glu=nn.GLU(dim=1)

    def forward(self, input):
        #GLU
        return self.glu(
                torch.cat(
                    (self.main(input), self.main_gate(input)), 1)
                )

class Convert_2D_1D(nn.Module):
    def __init__(self, reshape_channels, out_channels, kernel_size=1, stride=1, padding=0):
        super(Convert_2D_1D, self).__init__()

        self.reshape_channels=reshape_channels # for dim check
        self.convLayer = nn.Sequential(nn.Conv2d(in_channels=reshape_channels,
                                                 out_channels=out_channels,
                                                 kernel_size=kernel_size,
                                                 stride=stride,
                                                 padding=padding,
                                                 bias=False),
                                       nn.InstanceNorm2d(num_features=out_channels,
                                                         track_running_stats=True,
                                                         affine=True))

    # TODO: assume NCHW
    # TODO: check reshape_channels
    # [B, C, D, T//4] -> [B, C', 1, T//4]
    def forward(self, input):
        input = input.view(input.size(0), -1, 1, input.size(3))
        assert input.size(1) == self.reshape_channels, "the reshaped channel is different from the conv input channel"
        return self.convLayer(input) # H/D is 1 now

class Convert_1D_2D(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, kernel_size=1, stride=1, padding=0):
        super(Convert_1D_2D, self).__init__()

        self.hidden_channels=hidden_channels # for checking
        self.out_channels=out_channels # for reshaping 
        self.convLayer = nn.Conv2d(in_channels=in_channels,
                                                 out_channels=hidden_channels,
                                                 kernel_size=kernel_size,
                                                 stride=stride,
                                                 padding=padding)
                                       # removed from CyclaGANVC2
                                       #nn.InstanceNorm2d(num_features=hidden_channels,
                                       #                  affine=True))

    # TODO: assume NCHW
    # TODO: check hidden_channel
    def forward(self, input):
        input=self.convLayer(input)
        output=input.view(input.size(0), self.out_channels, -1, input.size(3))
        return output

class ConditionalInstanceNormWithRunningStats(nn.Module):
    def __init__(self, input_dim, dim_domain, eps=1e-05, momentum=0.1):
        super().__init__()
        self.norm = nn.InstanceNorm2d(input_dim, eps=eps, momentum=momentum, affine=False, track_running_stats=True)
        self.w_embedding=nn.Linear(dim_domain, input_dim)
        self.b_embedding=nn.Linear(dim_domain, input_dim)
        #w=torch.empty(dim_domain, input_dim)
        #b=torch.empty(dim_domain, input_dim)
        #nn.init.xavier_uniform_(w)
        #nn.init.xavier_uniform_(b)
        ##TODO, check will be updated
        #self.weight=nn.Parameter(w)
        #self.bias=nn.Parameter(b)

    # [B, C, 1, T], [B, Cat1*Cat2] | [Cat2*Cat1, C]
    # output: [B, C, 1, T]
    def forward(self, x, c):
        h, w=x.size()[-2], x.size()[-1]
        self.gamma=self.w_embedding(c).unsqueeze_(2).unsqueeze_(3)
        self.gamma=self.gamma.expand(-1, -1, h, w)
        self.beta=self.b_embedding(c).unsqueeze_(2).unsqueeze_(3)
        self.beta=self.beta.expand(-1, -1, h, w)
        out = self.norm(x)
        out = self.gamma * out + self.beta
        return out

#class ResidualBlockV2(nn.Module): # not residual anymore
class CINBlock(nn.Module):
    def __init__(self, in_channels, hidden_channels, dim_domain, kernel_size=(1, 5), stride=1, padding=(0, 2)):
        super(CINBlock, self).__init__()

        self.conv = nn.Conv2d(in_channels=in_channels,
                                                    out_channels=hidden_channels,
                                                    kernel_size=kernel_size,
                                                    stride=stride,
                                                    padding=padding,
                                                    bias=False)
        self.cin = ConditionalInstanceNormWithRunningStats(input_dim=hidden_channels, dim_domain=dim_domain)

        self.conv_gate = nn.Conv2d(in_channels=in_channels,
                                                    out_channels=hidden_channels,
                                                    kernel_size=kernel_size,
                                                    stride=stride,
                                                    padding=padding,
                                                    bias=False)
        self.cin_gate = ConditionalInstanceNormWithRunningStats(input_dim=hidden_channels, dim_domain=dim_domain)

        self.glu=nn.GLU(dim=1)

        # removed from CyclaGANVC2
        #self.conv1d_layer2 = nn.Sequential(nn.Conv1d(in_channels=hidden_channels,
        #                                                out_channels=in_channels,
        #                                                kernel_size=kernel_size,
        #                                                stride=stride,
        #                                                padding=padding),
        #                                      nn.InstanceNorm1d(num_features=in_channels,
        #                                                        affine=True))

    # [B, C, 1, T], [B, Cat1*Cat2]
    # output: [B, C, 1, T]
    def forward(self, input, c):
        h=self.conv(input)
        h_norm=self.cin(h, c)

        h_gate=self.conv_gate(input)
        h_gate_norm=self.cin_gate(h_gate, c)

        # removed from CyclaGANVC2
        #h2_norm = self.conv1d_layer2(h1_glu)
        #return input + h2_norm

        # GLU
        return self.glu(
                torch.cat(
                    (h_norm, h_gate_norm), 1)
                )

class GeneratorV2(nn.Module):
    def __init__(self, dim_domain=16):
        super(GeneratorV2, self).__init__()
        # we use 64, 128, 256 instead of 128, 256, 512, to make 212d c as 2304
        # otherwise it will be 4608 which is too big
        self.conv_in=nn.Conv2d(in_channels=1,
                out_channels=64,
                kernel_size=(5, 15),
                padding=(2, 7))
        self.conv_in_gate=nn.Conv2d(in_channels=1,
                out_channels=64,
                kernel_size=(5, 15),
                padding=(2, 7))
        self.glu=nn.GLU(dim=1)
        # same padding: 2*p >= k-s
        self.down_sample=nn.Sequential(
                DownSample_Block(in_channels=64,
                    out_channels=128,
                    kernel_size=5,
                    stride=2,
                    padding=2),
                DownSample_Block(in_channels=128,
                    out_channels=256,
                    kernel_size=5,
                    stride=2,
                    padding=2))
        self.convert_2d_1d=Convert_2D_1D(reshape_channels=2304, out_channels=256)
        self.cin_blocks=[]
        self.cin_blocks.append(CINBlock(in_channels=256, hidden_channels=512, dim_domain=dim_domain))
        for i in range(8):
            self.cin_blocks.append(CINBlock(in_channels=512, hidden_channels=512, dim_domain=dim_domain))
        self.cin_blocks=nn.ModuleList(self.cin_blocks)
        self.convert_1d_2d=Convert_1D_2D(in_channels=512, hidden_channels=2304, out_channels=256)
        self.up_sample=nn.Sequential(
                UpSample_Block(in_channels=256,
                    out_channels=1024,
                    kernel_size=5,
                    stride=1, # use PixelShuffle instead
                    padding=2),
                UpSample_Block(in_channels=256, # 1024/4
                    out_channels=512,
                    kernel_size=5,
                    stride=1, # use PixelShuffle instead
                    padding=2))
        self.conv_out=nn.Conv2d(in_channels=128, # 512/4
                out_channels=1, # in paper it is 35 which is a typo i think
                kernel_size=(5, 15),
                padding=(2, 7))

    # [B, 1, 36, 256], [B, 16]
    # output: [B, 1, 36, 256]
    def forward(self, x, c):
        x1=self.conv_in(x)
        x1_gate=self.conv_in_gate(x)
        x2=self.glu(torch.cat((x1, x1_gate), 1))
        x_down=self.down_sample(x2)
        x_1d=self.convert_2d_1d(x_down)
        for cin_block in self.cin_blocks: x_1d=cin_block(x_1d, c)
        x_2d=self.convert_1d_2d(x_1d)
        x_up=self.up_sample(x_2d)
        output=self.conv_out(x_up)
        return output

# Projection Discriminator
class DiscriminatorV2(nn.Module):
    def __init__(self, dim_domain=16):
        super(DiscriminatorV2, self).__init__()
        self.conv_in=nn.Conv2d(in_channels=1, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv_in_gate=nn.Conv2d(in_channels=1, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.glu=nn.GLU(dim=1)

        height, width=36, 128
        num_downsamples=4
        in_channels=128
        down_samples=[]
        for i in range(num_downsamples-1):
            down_samples.append(DownSample_Block(in_channels=in_channels, out_channels=in_channels*2, kernel_size=3, stride=2, padding=1))
            in_channels*=2
        down_samples.append(DownSample_Block(in_channels=in_channels, out_channels=in_channels, kernel_size=(1, 5), stride=1, padding=(0, 2)))
        self.down_samples=nn.Sequential(*down_samples)
        #TODO check output shape [5, 32]
        self.fc=nn.Linear(in_channels, 1)
        self.projection=nn.Linear(dim_domain, in_channels)

        # PatchGAN classifier
        # TODO floor or ceil
        kernel_size_0 = math.ceil(height / np.power(2, num_downsamples-1)) # 4
        kernel_size_1 = math.ceil(width / np.power(2, num_downsamples-1)) # 16
        # make it a single value
        self.conv_clf_spks = nn.Conv2d(in_channels, 4, kernel_size=(kernel_size_0, kernel_size_1), stride=1, padding=0, bias=False)  # for num_speaker

    # [B, 1, D, T], [B, Cat]
    # [B], [B, num_spk, 1, 1]
    def forward(self, x, c):
        x1=self.conv_in(x)
        x1_gate=self.conv_in_gate(x)
        x1=self.glu(torch.cat((x1, x1_gate), 1))
        for down_sample in self.down_samples: x1=down_sample(x1)
        #[B, C, D, T] -> [B, C]
        x1_vec=torch.sum(x1, dim=(2, 3)) # GlobalSumPooling
        #[B, C] -> [B, 1] -> [B]
        x_fc=self.fc(x1_vec).squeeze(1)
        #[B, Cat] -> [B, C]
        projections=self.projection(c)
        projected=x1_vec*projections
        # [B, C], [B, C] -> [B]
        projected=torch.sum(projected, dim=1)
        # [B, C, D, T] -> [B, num_spk, 1, 1]
        clf_spks=self.conv_clf_spks(x1).squeeze(3).squeeze(2)

        return x_fc+projected, clf_spks

# original
class ResidualBlockGradualV1(nn.Module):
    """Residual Block with instance normalization."""
    def __init__(self, dim_in, dim_out):
        super(ResidualBlockGradualV1, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True))

    def forward(self, x):
        return x + self.main(x)

# Original
class GeneratorGradualV1(nn.Module):
    """Generator network."""
    def __init__(self, conv_dim=64, num_speakers=10, repeat_num=6):
        super(GeneratorGradualV1, self).__init__()
        c_dim = num_speakers
        # Dim conversion
        self.conv_in=nn.Conv2d(1+c_dim, conv_dim, kernel_size=(3, 9), padding=(1, 4), bias=False)# NCHW, [B, C+1, D, T]
        self.in1=nn.InstanceNorm2d(conv_dim, affine=True, track_running_stats=True)
        self.act1=nn.ReLU(inplace=True)

        # Down-sampling layers.
        curr_dim = conv_dim
        self.down_conv1=nn.Conv2d(curr_dim, curr_dim*2, kernel_size=(4, 8), stride=(2, 2), padding=(1, 3), bias=False) # the padding is same padding
        self.down_in1=nn.InstanceNorm2d(curr_dim*2, affine=True, track_running_stats=True)
        self.down_act1=nn.ReLU(inplace=True)
        curr_dim = curr_dim * 2
        self.down_conv2=nn.Conv2d(curr_dim, curr_dim*2, kernel_size=(4, 8), stride=(2, 2), padding=(1, 3), bias=False) # the padding is same padding
        self.down_in2=nn.InstanceNorm2d(curr_dim*2, affine=True, track_running_stats=True)
        self.down_act2=nn.ReLU(inplace=True)
        curr_dim = curr_dim * 2

        # Bottleneck layers.
        self.res_block1=ResidualBlockGradualV1(dim_in=curr_dim, dim_out=curr_dim)
        self.res_block2=ResidualBlockGradualV1(dim_in=curr_dim, dim_out=curr_dim)
        self.res_block3=ResidualBlockGradualV1(dim_in=curr_dim, dim_out=curr_dim)
        self.res_block4=ResidualBlockGradualV1(dim_in=curr_dim, dim_out=curr_dim)
        self.res_block5=ResidualBlockGradualV1(dim_in=curr_dim, dim_out=curr_dim)
        self.res_block6=ResidualBlockGradualV1(dim_in=curr_dim, dim_out=curr_dim)

        # Up-sampling layers.
        self.up_conv1=nn.ConvTranspose2d(curr_dim, curr_dim//2, kernel_size=4, stride=2, padding=1, bias=False)
        self.up_in1=nn.InstanceNorm2d(curr_dim//2, affine=True, track_running_stats=True)
        self.up_act1=nn.ReLU(inplace=True)
        curr_dim = curr_dim // 2
        self.up_conv2=nn.ConvTranspose2d(curr_dim, curr_dim//2, kernel_size=4, stride=2, padding=1, bias=False)
        self.up_in2=nn.InstanceNorm2d(curr_dim//2, affine=True, track_running_stats=True)
        self.up_act2=nn.ReLU(inplace=True)
        curr_dim = curr_dim // 2

        # Dim conversion
        self.conv_out=nn.Conv2d(curr_dim, 1, kernel_size=7, stride=1, padding=3, bias=False)

    # x:[B, 1, D, T], c:[B, C]
    def forward(self, x, c):
        # Replicate spatially and concatenate domain information.
        c = c.view(c.size(0), c.size(1), 1, 1)
        c = c.repeat(1, 1, x.size(2), x.size(3)) # c:[B, C, D, T]
        x = torch.cat([x, c], dim=1) # x:[B, C+1, D, T] the c+1 dim will be [D, T] one matrix, others are zero matrix
        # [B, C+1, D, T] -> [B, 1, D, T]
        x_conv_in=self.conv_in(x)
        x_in_in=self.in1(x_conv_in)
        x_act_in=self.act1(x_in_in)

        x_conv1_down=self.down_conv1(x_act_in)
        x_in1_down=self.down_in1(x_conv1_down)
        x_act1_down=self.down_act1(x_in1_down)
        x_conv2_down=self.down_conv2(x_act1_down)
        x_in2_down=self.down_in2(x_conv2_down)
        x_act2_down=self.down_act2(x_in2_down)

        x_res=self.res_block1(x_act2_down)
        x_res=self.res_block2(x_res)
        x_res=self.res_block3(x_res)
        x_res=self.res_block4(x_res)
        x_res=self.res_block5(x_res)
        x_res=self.res_block6(x_res)

        x_conv1_up=self.up_conv1(x_res)
        x_in1_up=self.up_in1(x_conv1_up)
        x_act1_up=self.up_act1(x_in1_up)
        x_conv2_up=self.up_conv2(x_act1_up)
        x_in2_up=self.up_in2(x_conv2_up)
        x_act2_up=self.up_act2(x_in2_up)

        x_out=self.conv_out(x_act2_up)

        return x_out

#GLU only
class ResidualBlockGradualV2(nn.Module):
    """Residual Block with instance normalization."""
    def __init__(self, dim_in, dim_out):
        super(ResidualBlockGradualV2, self).__init__()
        #self.main = nn.Sequential(
        #    nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
        #    nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True),
        #    nn.ReLU(inplace=True),
        #    nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
        #    nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True))
        # NOTE: mod1
        self.conv1=nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False)
        self.in1=nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True)
        self.conv1_gate=nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False)
        self.in1_gate=nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True)
        self.glu=nn.GLU(dim=1)
        self.conv2=nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False)
        self.in2=nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True)

    def forward(self, x):
        #return x + self.main(x)
        # NOTE: mod1
        x1=self.conv1(x)
        x1=self.in1(x1)
        x2=self.conv1_gate(x)
        x2=self.in1_gate(x2)
        x3=self.glu(torch.cat((x1, x2), 1))
        x3=self.conv2(x3)
        return self.in2(x3)

#GLU only
class GeneratorGradualV2(nn.Module):
    """Generator network."""
    def __init__(self, conv_dim=64, num_speakers=10, repeat_num=6):
        super(GeneratorGradualV2, self).__init__()
        c_dim = num_speakers
        # Dim conversion
        self.conv_in=nn.Conv2d(1+c_dim, conv_dim, kernel_size=(3, 9), padding=(1, 4), bias=False)# NCHW, [B, C+1, D, T]
        self.in1=nn.InstanceNorm2d(conv_dim, affine=True, track_running_stats=True)
        self.act1=nn.ReLU(inplace=True)

        # NOTE: mod1
        self.conv_in_gate=nn.Conv2d(1+c_dim, conv_dim, kernel_size=(3, 9), padding=(1, 4), bias=False)# NCHW, [B, C+1, D, T]
        self.in1_gate=nn.InstanceNorm2d(conv_dim, affine=True, track_running_stats=True)
        self.glu_in=nn.GLU(dim=1)

        # Down-sampling layers.
        curr_dim = conv_dim
        self.down_conv1=nn.Conv2d(curr_dim, curr_dim*2, kernel_size=(4, 8), stride=(2, 2), padding=(1, 3), bias=False) # the padding is same padding
        self.down_in1=nn.InstanceNorm2d(curr_dim*2, affine=True, track_running_stats=True)
        self.down_act1=nn.ReLU(inplace=True)

        # NOTE: mod1
        self.down_conv1_gate=nn.Conv2d(curr_dim, curr_dim*2, kernel_size=(4, 8), stride=(2, 2), padding=(1, 3), bias=False) # the padding is same padding
        self.down_in1_gate=nn.InstanceNorm2d(curr_dim*2, affine=True, track_running_stats=True)
        self.glu_down1=nn.GLU(dim=1)

        curr_dim = curr_dim * 2

        self.down_conv2=nn.Conv2d(curr_dim, curr_dim*2, kernel_size=(4, 8), stride=(2, 2), padding=(1, 3), bias=False) # the padding is same padding
        self.down_in2=nn.InstanceNorm2d(curr_dim*2, affine=True, track_running_stats=True)
        self.down_act2=nn.ReLU(inplace=True)

        # NOTE: mod1
        self.down_conv2_gate=nn.Conv2d(curr_dim, curr_dim*2, kernel_size=(4, 8), stride=(2, 2), padding=(1, 3), bias=False) # the padding is same padding
        self.down_in2_gate=nn.InstanceNorm2d(curr_dim*2, affine=True, track_running_stats=True)
        self.glu_down2=nn.GLU(dim=1)

        curr_dim = curr_dim * 2

        # Bottleneck layers.
        self.res_block1=ResidualBlockGradualV2(dim_in=curr_dim, dim_out=curr_dim)
        self.res_block2=ResidualBlockGradualV2(dim_in=curr_dim, dim_out=curr_dim)
        self.res_block3=ResidualBlockGradualV2(dim_in=curr_dim, dim_out=curr_dim)
        self.res_block4=ResidualBlockGradualV2(dim_in=curr_dim, dim_out=curr_dim)
        self.res_block5=ResidualBlockGradualV2(dim_in=curr_dim, dim_out=curr_dim)
        self.res_block6=ResidualBlockGradualV2(dim_in=curr_dim, dim_out=curr_dim)

        # Up-sampling layers.
        self.up_conv1=nn.ConvTranspose2d(curr_dim, curr_dim//2, kernel_size=4, stride=2, padding=1, bias=False)
        self.up_in1=nn.InstanceNorm2d(curr_dim//2, affine=True, track_running_stats=True)
        self.up_act1=nn.ReLU(inplace=True)

        # NOTE: mod1
        self.up_conv1_gate=nn.ConvTranspose2d(curr_dim, curr_dim//2, kernel_size=4, stride=2, padding=1, bias=False)
        self.up_in1_gate=nn.InstanceNorm2d(curr_dim//2, affine=True, track_running_stats=True)
        self.glu_up1=nn.GLU(dim=1)

        curr_dim = curr_dim // 2

        self.up_conv2=nn.ConvTranspose2d(curr_dim, curr_dim//2, kernel_size=4, stride=2, padding=1, bias=False)
        self.up_in2=nn.InstanceNorm2d(curr_dim//2, affine=True, track_running_stats=True)
        self.up_act2=nn.ReLU(inplace=True)

        # NOTE: mod1
        self.up_conv2_gate=nn.ConvTranspose2d(curr_dim, curr_dim//2, kernel_size=4, stride=2, padding=1, bias=False)
        self.up_in2_gate=nn.InstanceNorm2d(curr_dim//2, affine=True, track_running_stats=True)
        self.glu_up2=nn.GLU(dim=1)
        curr_dim = curr_dim // 2

        # Dim conversion
        self.conv_out=nn.Conv2d(curr_dim, 1, kernel_size=7, stride=1, padding=3, bias=False)

    # x:[B, 1, D, T], c:[B, C]
    def forward(self, x, c):
        # Replicate spatially and concatenate domain information.
        c = c.view(c.size(0), c.size(1), 1, 1)
        c = c.repeat(1, 1, x.size(2), x.size(3)) # c:[B, C, D, T]
        x = torch.cat([x, c], dim=1) # x:[B, C+1, D, T] the c+1 dim will be [D, T] one matrix, others are zero matrix
        # [B, C+1, D, T] -> [B, 1, D, T]
        x_conv_in=self.conv_in(x)
        x_in_in=self.in1(x_conv_in)
        #x_act_in=self.act1(x_in_in)

        # NOTE: mod1
        x_conv_in_gate=self.conv_in_gate(x)
        x_in_in_gate=self.in1_gate(x_conv_in_gate)
        x_act_in=self.glu_in(torch.cat((x_in_in, x_in_in_gate), 1))

        x_conv1_down=self.down_conv1(x_act_in)
        x_in1_down=self.down_in1(x_conv1_down)
        #x_act1_down=self.down_act1(x_in1_down)
        # NOTE: mod1
        x_conv1_down_gate=self.down_conv1_gate(x_act_in)
        x_in1_down_gate=self.down_in1_gate(x_conv1_down_gate)
        x_act1_down=self.glu_down1(torch.cat((x_in1_down, x_in1_down_gate), 1))

        x_conv2_down=self.down_conv2(x_act1_down)
        x_in2_down=self.down_in2(x_conv2_down)
        #x_act2_down=self.down_act2(x_in2_down)
        # NOTE: mod1
        x_conv2_down_gate=self.down_conv2_gate(x_act1_down)
        x_in2_down_gate=self.down_in2_gate(x_conv2_down_gate)
        x_act2_down=self.glu_down2(torch.cat((x_in2_down, x_in2_down_gate), 1))

        x_res=self.res_block1(x_act2_down)
        x_res=self.res_block2(x_res)
        x_res=self.res_block3(x_res)
        x_res=self.res_block4(x_res)
        x_res=self.res_block5(x_res)
        x_res=self.res_block6(x_res)

        x_conv1_up=self.up_conv1(x_res)
        x_in1_up=self.up_in1(x_conv1_up)
        #x_act1_up=self.up_act1(x_in1_up)
        # NOTE: mod1
        x_conv1_up_gate=self.up_conv1_gate(x_res)
        x_in1_up_gate=self.up_in1_gate(x_conv1_up_gate)
        x_act1_up=self.glu_up1(torch.cat((x_in1_up, x_in1_up_gate), 1))

        x_conv2_up=self.up_conv2(x_act1_up)
        x_in2_up=self.up_in2(x_conv2_up)
        #x_act2_up=self.up_act2(x_in2_up)
        # NOTE: mod1
        x_conv2_up_gate=self.up_conv2_gate(x_act1_up)
        x_in2_up_gate=self.up_in2_gate(x_conv2_up_gate)
        x_act2_up=self.glu_up2(torch.cat((x_in2_up, x_in2_up_gate), 1))


        x_out=self.conv_out(x_act2_up)

        return x_out

# 2-1-2D
class ResidualBlockGradualV2B(nn.Module):
    """Residual Block with instance normalization."""
    def __init__(self, dim_in, dim_out, kernel_size=3, padding=1):
        super(ResidualBlockGradualV2B, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, stride=1, padding=padding, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=kernel_size, stride=1, padding=padding, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True))

    def forward(self, x):
        return x + self.main(x)

# 2-1-2D
class GeneratorGradualV2B(nn.Module):
    """Generator network."""
    def __init__(self, input_size=(36, 128), conv_dim=64, num_speakers=10, repeat_num=6):
        super(GeneratorGradualV2B, self).__init__()
        c_dim = num_speakers
        # Dim conversion
        self.conv_in=nn.Conv2d(1+c_dim, conv_dim, kernel_size=(3, 9), padding=(1, 4), bias=False)# NCHW, [B, C+1, D, T]
        self.in1=nn.InstanceNorm2d(conv_dim, affine=True, track_running_stats=True)
        self.act1=nn.ReLU(inplace=True)

        # Down-sampling layers.
        curr_dim = conv_dim
        self.down_conv1=nn.Conv2d(curr_dim, curr_dim*2, kernel_size=(4, 8), stride=(2, 2), padding=(1, 3), bias=False) # the padding is same padding
        self.down_in1=nn.InstanceNorm2d(curr_dim*2, affine=True, track_running_stats=True)
        self.down_act1=nn.ReLU(inplace=True)
        curr_dim = curr_dim * 2
        self.down_conv2=nn.Conv2d(curr_dim, curr_dim*2, kernel_size=(4, 8), stride=(2, 2), padding=(1, 3), bias=False) # the padding is same padding
        self.down_in2=nn.InstanceNorm2d(curr_dim*2, affine=True, track_running_stats=True)
        self.down_act2=nn.ReLU(inplace=True)
        curr_dim = curr_dim * 2

        # 2-1D
        self.convert_2d_1d=Convert_2D_1D(reshape_channels=int(curr_dim*input_size[0]/4), out_channels=curr_dim)

        # Bottleneck layers.
        self.res_block1=ResidualBlockGradualV2B(dim_in=curr_dim, dim_out=curr_dim, kernel_size=(1, 3), padding=(0, 1))
        self.res_block2=ResidualBlockGradualV2B(dim_in=curr_dim, dim_out=curr_dim, kernel_size=(1, 3), padding=(0, 1))
        self.res_block3=ResidualBlockGradualV2B(dim_in=curr_dim, dim_out=curr_dim, kernel_size=(1, 3), padding=(0, 1))
        self.res_block4=ResidualBlockGradualV2B(dim_in=curr_dim, dim_out=curr_dim, kernel_size=(1, 3), padding=(0, 1))
        self.res_block5=ResidualBlockGradualV2B(dim_in=curr_dim, dim_out=curr_dim, kernel_size=(1, 3), padding=(0, 1))
        self.res_block6=ResidualBlockGradualV2B(dim_in=curr_dim, dim_out=curr_dim, kernel_size=(1, 3), padding=(0, 1))

        # 1-2D
        self.convert_1d_2d=Convert_1D_2D(in_channels=curr_dim, hidden_channels=int(curr_dim*input_size[0]/4), out_channels=curr_dim)

        # Up-sampling layers.
        self.up_conv1=nn.ConvTranspose2d(curr_dim, curr_dim//2, kernel_size=4, stride=2, padding=1, bias=False)
        self.up_in1=nn.InstanceNorm2d(curr_dim//2, affine=True, track_running_stats=True)
        self.up_act1=nn.ReLU(inplace=True)
        curr_dim = curr_dim // 2
        self.up_conv2=nn.ConvTranspose2d(curr_dim, curr_dim//2, kernel_size=4, stride=2, padding=1, bias=False)
        self.up_in2=nn.InstanceNorm2d(curr_dim//2, affine=True, track_running_stats=True)
        self.up_act2=nn.ReLU(inplace=True)
        curr_dim = curr_dim // 2

        # Dim conversion
        self.conv_out=nn.Conv2d(curr_dim, 1, kernel_size=7, stride=1, padding=3, bias=False)

    # x:[B, 1, D, T], c:[B, C]
    def forward(self, x, c):
        # Replicate spatially and concatenate domain information.
        c = c.view(c.size(0), c.size(1), 1, 1)
        c = c.repeat(1, 1, x.size(2), x.size(3)) # c:[B, C, D, T]
        x = torch.cat([x, c], dim=1) # x:[B, C+1, D, T] the c+1 dim will be [D, T] one matrix, others are zero matrix
        # [B, C+1, D, T] -> [B, 1, D, T]
        x_conv_in=self.conv_in(x)
        x_in_in=self.in1(x_conv_in)
        x_act_in=self.act1(x_in_in)

        x_conv1_down=self.down_conv1(x_act_in)
        x_in1_down=self.down_in1(x_conv1_down)
        x_act1_down=self.down_act1(x_in1_down)
        x_conv2_down=self.down_conv2(x_act1_down)
        x_in2_down=self.down_in2(x_conv2_down)
        x_act2_down=self.down_act2(x_in2_down)

        x_1d=self.convert_2d_1d(x_act2_down)

        x_res=self.res_block1(x_1d)
        x_res=self.res_block2(x_res)
        x_res=self.res_block3(x_res)
        x_res=self.res_block4(x_res)
        x_res=self.res_block5(x_res)
        x_res=self.res_block6(x_res)

        x_2d=self.convert_1d_2d(x_res)

        x_conv1_up=self.up_conv1(x_2d)
        x_in1_up=self.up_in1(x_conv1_up)
        x_act1_up=self.up_act1(x_in1_up)
        x_conv2_up=self.up_conv2(x_act1_up)
        x_in2_up=self.up_in2(x_conv2_up)
        x_act2_up=self.up_act2(x_in2_up)

        x_out=self.conv_out(x_act2_up)

        return x_out

# (36, 128) version PatchGAN
class DiscriminatorV1(nn.Module):
    """Discriminator network with PatchGAN."""
    def __init__(self, input_size=(36, 128), conv_dim=64, repeat_num=5, num_speakers=4):
        super(DiscriminatorV1, self).__init__()
        layers = []
        # Dim convert & Downsample
        layers.append(nn.Conv2d(1, conv_dim, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.01))

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.01))
            curr_dim = curr_dim * 2

        kernel_size_0 = int(input_size[0] / np.power(2, repeat_num)) # 1
        kernel_size_1 = int(input_size[1] / np.power(2, repeat_num)) # 8
        self.main = nn.Sequential(*layers)
        # make it a single value
        self.conv_dis = nn.Conv2d(curr_dim, 1, kernel_size=(kernel_size_0, kernel_size_1), stride=1, padding=0, bias=False) # padding should be 0
        self.conv_clf_spks = nn.Conv2d(curr_dim, num_speakers, kernel_size=(kernel_size_0, kernel_size_1), stride=1, padding=0, bias=False)  # for num_speaker

    def forward(self, x):
        # [B, 1, D, T] -> [B, 64, 18, 128] -> [B, 64*2^4, 1, 8]
        h = self.main(x)
        # [B, 1024, 1, 8] -> [B, 1, 1, 1]
        out_src = self.conv_dis(h)
        # [B, 1024, 1, 8] -> [B, 10, 1, 1]
        out_cls_spks = self.conv_clf_spks(h)
        return out_src, out_cls_spks.view(out_cls_spks.size(0), out_cls_spks.size(1))

# ================= V1 =======================
class ResidualBlock(nn.Module):
    """Residual Block with instance normalization."""
    def __init__(self, dim_in, dim_out):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True))

    def forward(self, x):
        return x + self.main(x)

class Generator(nn.Module):
    """Generator network."""
    def __init__(self, conv_dim=64, num_speakers=10, repeat_num=6):
        super(Generator, self).__init__()
        c_dim = num_speakers
        layers = []
        # Dim conversion
        layers.append(nn.Conv2d(1+c_dim, conv_dim, kernel_size=(3, 9), padding=(1, 4), bias=False))# NCHW, [B, C+1, D, T]
        layers.append(nn.InstanceNorm2d(conv_dim, affine=True, track_running_stats=True))
        layers.append(nn.ReLU(inplace=True))

        # Down-sampling layers.
        curr_dim = conv_dim
        for i in range(2):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=(4, 8), stride=(2, 2), padding=(1, 3), bias=False)) # the padding is same padding
            layers.append(nn.InstanceNorm2d(curr_dim*2, affine=True, track_running_stats=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim * 2

        # Bottleneck layers.
        for i in range(repeat_num):
            layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))

        # Up-sampling layers.
        for i in range(2):
            layers.append(nn.ConvTranspose2d(curr_dim, curr_dim//2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim//2, affine=True, track_running_stats=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim // 2

        # Dim conversion
        layers.append(nn.Conv2d(curr_dim, 1, kernel_size=7, stride=1, padding=3, bias=False))
        self.main = nn.Sequential(*layers)

    # x:[B, 1, D, T], c:[B, C]
    def forward(self, x, c):
        # Replicate spatially and concatenate domain information.
        c = c.view(c.size(0), c.size(1), 1, 1)
        c = c.repeat(1, 1, x.size(2), x.size(3)) # c:[B, C, D, T]
        x = torch.cat([x, c], dim=1) # x:[B, C+1, D, T] the c+1 dim will be [D, T] one matrix, others are zero matrix
        # [B, C+1, D, T] -> [B, 1, D, T]
        return self.main(x)

class Discriminator(nn.Module):
    """Discriminator network with PatchGAN."""
    def __init__(self, input_size=(36, 256), conv_dim=64, repeat_num=5, num_speakers=10):
        super(Discriminator, self).__init__()
        layers = []
        # Dim convert & Downsample
        layers.append(nn.Conv2d(1, conv_dim, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.01))

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.01))
            curr_dim = curr_dim * 2

        kernel_size_0 = int(input_size[0] / np.power(2, repeat_num)) # 1
        kernel_size_1 = int(input_size[1] / np.power(2, repeat_num)) # 8
        self.main = nn.Sequential(*layers)
        # make it a single value
        self.conv_dis = nn.Conv2d(curr_dim, 1, kernel_size=(kernel_size_0, kernel_size_1), stride=1, padding=0, bias=False) # padding should be 0
        self.conv_clf_spks = nn.Conv2d(curr_dim, num_speakers, kernel_size=(kernel_size_0, kernel_size_1), stride=1, padding=0, bias=False)  # for num_speaker

    def forward(self, x):
        # [B, 1, D, T] -> [B, 64, 18, 128] -> [B, 64*2^4, 1, 8]
        h = self.main(x)
        # [B, 1024, 1, 8] -> [B, 1, 1, 1]
        out_src = self.conv_dis(h)
        # [B, 1024, 1, 8] -> [B, 10, 1, 1]
        out_cls_spks = self.conv_clf_spks(h)
        return out_src, out_cls_spks.view(out_cls_spks.size(0), out_cls_spks.size(1))

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader = get_loader('vcc2018/mc_vcc2018/train', 16, 'train', num_workers=1)
    data_iter = iter(train_loader)
    version='v2'
    if version=='v1':
        G = Generator().to(device)
        D = Discriminator().to(device)
        for i in range(10):
            mc_real, spk_label_org, spk_acc_c_org, _,_,_,_ = next(data_iter)
            mc_real.unsqueeze_(1) # (B, D, T) -> (B, 1, D, T) for conv2d
            mc_real = mc_real.to(device)                         # Input mc.
            spk_label_org = spk_label_org.to(device)             # Original spk labels.
            spk_acc_c_org = spk_acc_c_org.to(device)             # Original spk acc conditioning.
            mc_fake = G(mc_real, spk_acc_c_org)
            print(mc_fake.size())
            out_src, out_cls_spks, out_cls_emos = D(mc_fake)
    elif version=='v2':
        G=GeneratorV2().to(device)
        D=DiscriminatorV2().to(device)
        for i in range(10):
            mc_real, spk_label_org, _, _,_,_, duo_cat = next(data_iter)
            mc_real.unsqueeze_(1) # (B, D, T) -> (B, 1, D, T) for conv2d
            mc_real = mc_real.to(device)                         # Input mc.
            duo_cat = duo_cat.to(device)
            mc_fake = G(mc_real, duo_cat)
            print(mc_fake.size())
            out_src, out_cls_spks = D(mc_fake, duo_cat)
            print(out_src.shape, out_cls_spks.shape)


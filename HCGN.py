# from tkinter import _Padding
from audioop import lin2adpcm
from optparse import TitledHelpFormatter
from cv2 import _OutputArray_DEPTH_MASK_16F
from sympy import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import numpy as np

# from Main import S

# from Main import Q
# from Main import Q1
# from Main import Q2

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

class GCNLayer(nn.Module):
    def __init__(self, input_dim: int, output_dim: int,A:torch.Tensor):
        super(GCNLayer, self).__init__()
        self.A = A
        self.BN = nn.BatchNorm1d(input_dim)
        self.Activition = nn.LeakyReLU()
        self.sigma1= torch.nn.Parameter(torch.tensor([0.1],requires_grad=True))
        # 第一层GCN
        self.GCN_liner_theta_1 =nn.Sequential(nn.Linear(input_dim, 256))
        self.DeFC =nn.Sequential(nn.Linear(256, 256))
        self.GCN_liner_theta_4 =nn.Sequential(nn.Linear(256+input_dim, 256))
        self.GCN_liner_theta_2 =nn.Linear(input_dim, input_dim)
        self.GCN_liner_theta_3 =nn.Linear(64, 64)
        self.GCN_liner_out_1 =nn.Sequential(nn.Linear(input_dim, output_dim))
        self.bn = nn.BatchNorm1d(output_dim)
        self.GCN_liner_out_11 =nn.Sequential(nn.Linear(output_dim, output_dim))
        self.GCN_liner_out_22 =nn.Sequential(nn.Linear(output_dim, output_dim))

        nodes_count=self.A.shape[0]
        self.I = torch.eye(nodes_count, nodes_count, requires_grad=False).to(device)
        self.mask=torch.ceil( self.A*0.00001)
        self.TFormer = TFormer(128)
        
    def A_to_D_inv(self, A: torch.Tensor):
        D = A.sum(1)
        D_hat = torch.diag(torch.pow(D, -0.5))
        return D_hat
    
    def forward(self, H, model='normal'):
        # # 方案一：minmax归一化
        # H = self.BN(H)
        # H_xx1= self.GCN_liner_theta_1(H)
        # A = torch.clamp(torch.sigmoid(torch.matmul(H_xx1, H_xx1.t())), min=0.1) * self.mask + self.I
        # if model != 'normal': A=torch.clamp(A,0.1) #This is a trick.
        # D_hat = self.A_to_D_inv(A)
        # A_hat = torch.matmul(D_hat, torch.matmul(A,D_hat))
        # output = torch.mm(A_hat, self.GCN_liner_out_1(H))
        # output = self.Activition(output)
        
        # # 方案二：softmax归一化 (加速运算)
        
        # print(H.shape)
        H = self.BN(H)     # [196,128]
        H_xx1= self.GCN_liner_theta_1(H)   # [196,256]

        # H_xx1= self.GCN_liner_theta_4(H_xx1) 
        # a = F.softmax(H_xx1, dim=1)
        # H_xx1 = H_xx1 + a
        # H_xx2= self.GCN_liner_theta_1(H) 
        # H_xx1 = H_xx1 + H_xx2
        # H_xx1 =H
        # distance = F.pairwise_distance(H_xx1, H_xx1, p=2)
        # std = torch.std(distance)

        e = torch.sigmoid(torch.matmul(H_xx1, H_xx1.t()))   # [196,196]
        zero_vec = -9e15 * torch.ones_like(e)
        # print(distance)
        A1 = torch.where(self.mask > 0, e, zero_vec)
        
        A = A1 + self.I
        if model != 'normal': A=torch.clamp(A,0.1) #This is a trick for the Indian Pines.

        A = F.softmax(A, dim=1)
        output1 = self.Activition(torch.mm(A, self.GCN_liner_out_1(H)))
        
        A1 = F.softmax(A1, dim=1)
        output2 = self.Activition(torch.mm(A1, self.GCN_liner_out_1(H)))

        output = torch.add(output1,output2)
        # output = torch.cat((output1,output2 ),dim=1)
        # output = self.GCN_liner_theta_2(output)
        # x1= self.GCN_liner_theta_3(x1)
        # print('2',x1.shape)
        # x1 = F.softmax(x1, dim=1)
        # output = x1 + output

        return output,A

class SpectralConv(nn.Module):
    '''
    Spectral-Spatial Convolution
    '''
    def __init__(self, in_ch, out_ch):
        super(SpectralConv, self).__init__()
        # self.depth_conv = nn.Conv2d(
        #     in_channels=out_ch,
        #     out_channels=out_ch,
        #     kernel_size=kernel_size,
        #     stride=1,
        #     padding=kernel_size//2,
        #     groups=out_ch,
        #     # dilation=2
        # )
        self.point_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            bias=False
        )
        self.Act1 = nn.LeakyReLU()
        self.Act2 = nn.LeakyReLU()
        self.BN=nn.BatchNorm2d(in_ch)
        # self.CAM = CAM(out_ch)
        # self.PAM = PAM_Module(out_ch)
        self.conv1 = nn.Conv3d(in_channels=1,out_channels=1,kernel_size=(3,1,1),padding=(1,0,0))
        self.conv2 = nn.Conv3d(in_channels=1,out_channels=1,kernel_size=(1,3,1),padding=(0,1,0))
        self.conv3 = nn.Conv3d(in_channels=1,out_channels=1,kernel_size=(1,1,3),padding=(0,0,1))
 

    def forward(self, input):
        out = self.point_conv(self.BN(input))
        out = self.Act1(out)
        out = self.depth_conv(out)
        out = self.Act2(out)
        b,c,h,w = out.shape
        out1 = self.conv1(out)
        out1 = self.Act1(out1)
        out2 = self.conv2(out2)
        out2 = self.Act1(out2)
        out3 = self.conv3(out3)
        out3 = self.Act1(out3)

        # out = out3
        out = out + out3 + out1 + out2
        out = out.reshape(b,c,h,w)
        

        # out = self.PAM(out)
        return out

class SSConv(nn.Module):
    '''
    Spectral-Spatial Convolution
    '''
    def __init__(self, in_ch, out_ch,kernel_size):
        super(SSConv, self).__init__()
        self.depth_conv = nn.Conv2d(
            in_channels=out_ch,
            out_channels=out_ch,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size//2,
            groups=out_ch,
            # dilation=2
        )
        self.point_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            bias=False
        )
        self.Act1 = nn.LeakyReLU()
        self.Act2 = nn.LeakyReLU()
        self.BN=nn.BatchNorm2d(in_ch)
        # self.CAM = CAM(out_ch)
        # self.PAM = PAM_Module(out_ch)
        self.conv1 = nn.Conv3d(in_channels=1,out_channels=1,kernel_size=(3,1,1),padding=(1,0,0))
        self.conv2 = nn.Conv3d(in_channels=1,out_channels=1,kernel_size=(1,3,1),padding=(0,1,0))
        self.conv3 = nn.Conv3d(in_channels=1,out_channels=1,kernel_size=(3,1,1),padding=(1,0,0))
        self.BN1=nn.BatchNorm3d(1)
        self.fc = nn.Linear(out_ch*3,out_ch)

    def forward(self, input):
 
        out = self.point_conv(self.BN(input))
        out1 = self.Act1(out)
        out = self.depth_conv(out1)
        out = self.Act2(out)
        # out = self.depth_conv(out1)
        # out = self.Act2(out)
        # out = self.PAM(out)
        return out

class SSConv1(nn.Module):
    '''
    Spectral-Spatial Convolution
    '''
    def __init__(self, in_ch, out_ch,kernel_size):
        super(SSConv1, self).__init__()
        self.depth_conv = nn.Conv2d(
            in_channels=out_ch,
            out_channels=out_ch,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size//2,
            groups=out_ch,
            # dilation=2
        )
        self.point_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            bias=False
        )
        self.Act1 = nn.LeakyReLU()
        self.Act2 = nn.LeakyReLU()
        self.BN=nn.BatchNorm2d(in_ch)
        # self.CAM = CAM(out_ch)
        # self.PAM = PAM_Module(out_ch)
        self.conv1 = nn.Conv3d(in_channels=1,out_channels=1,kernel_size=(3,1,1),padding=(1,0,0))
        self.conv2 = nn.Conv3d(in_channels=1,out_channels=1,kernel_size=(1,3,1),padding=(0,1,0))
        self.conv3 = nn.Conv3d(in_channels=1,out_channels=1,kernel_size=(3,1,1),padding=(1,0,0))
        self.BN1=nn.BatchNorm3d(1)
        self.fc = nn.Linear(out_ch*3,out_ch)

    def forward(self, input):
 
        out = self.point_conv(self.BN(input))
        out1 = self.Act1(out)
        out = self.depth_conv(out1)
        out = self.Act2(out)
        # out = self.depth_conv(out1)
        # out = self.Act2(out)
        # out = self.PAM(out)
        return out


# class PAM_Module(nn.Module):
#     def __init__(self, in_dim):
#         super(PAM_Module, self).__init__()
#         self.chanel_in = in_dim
#         self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
#         self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
#         self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
#         self.gamma = nn.Parameter(torch.zeros(1))

#         self.softmax = nn.Softmax(dim=-1)
#     def forward(self,x):

#         # C, channle, height, width = x.size()
#         # x = x.reshape(C, height, width, channle)   # [1,145,145,c]
#         # # print(x.size())
#         # proj_query = x.view(C, channle, height*width)
#         # proj_key = x.view(C, height*width, channle) #形状转换并交换维度
#         # energy = torch.bmm(proj_key,proj_query)
#         # energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
#         # attention = self.softmax(energy_new)   # [h,h]
#         # proj_value = x.view(C, height*width, channle)

#         # out = torch.bmm(attention,proj_value)
#         # out = out.view(C, height, width, channle)
#         # # print('out', out.shape)
#         # # print('x', x.shape)

#         # out = self.gamma*out + x  #C*H*W
#         # out = out.reshape(C,channle,height,width)


#         m_batchsize, C, height, width = x.size() 
#         proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
#         proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
#         energy = torch.bmm(proj_query, proj_key)
#         attention = self.softmax(energy)
#         proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)

#         out = torch.bmm(proj_value, attention.permute(0, 2, 1))
#         out = out.view(m_batchsize, C, height, width)

#         out = (self.gamma*out + x)

#         return out

# class CAM(nn.Module):
#     def __init__(self, out_ch):
#         super(CAM, self).__init__()
#         self.chanel_in = out_ch


#         self.gamma = nn.Parameter(torch.zeros(1))
#         self.softmax  = nn.Softmax(dim=-1)
#     def forward(self,x):
#         """
#             inputs :
#                 x : input feature maps( B X C X H X W)
#             returns :
#                 out : attention value + input feature
#                 attention: B X C X C
#         """
#         C, channle, height, width = x.size()
#         x = x.reshape(C, height, width, channle)   # [1,145,145,c]
#         # print(x.size())
#         proj_query = x.view(C, channle, height*width)
#         proj_key = x.view(C, height*width, channle) #形状转换并交换维度
#         energy = torch.bmm(proj_query, proj_key)
#         energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
#         attention = self.softmax(energy_new)   # [c,c]
#         proj_value = x.view(C, height*width, channle)

#         out = torch.bmm(proj_value,attention)
#         out = out.view(C, height, width, channle)
#         # print('out', out.shape)
#         # print('x', x.shape)

#         out = self.gamma*out + x  #C*H*W
#         out = out.reshape(C,channle,height,width)
#         return out

class CNNConv(nn.Module):
    def __init__(self, channel=128,out_channel=128):
        super(CNNConv, self).__init__()
        self.conv1 = nn.Conv2d(channel,out_channel,kernel_size=1)
        self.bn = nn.Sequential(nn.BatchNorm2d(channel,  eps=0.001, momentum=0.1, affine=True))
        self.bn1 = nn.Sequential(nn.BatchNorm2d(out_channel,  eps=0.001, momentum=0.1, affine=True))
        self.relu = nn.LeakyReLU()

        self.conv21 = nn.Conv2d(channel,out_channel,kernel_size=3,padding=1)
        self.conv22 = nn.Conv2d(out_channel,out_channel,kernel_size=3,padding=1)
        self.conv23 = nn.Conv2d(out_channel,out_channel,kernel_size=3,padding=1)

        self.conv3 = nn.Conv2d(channel,out_channel,kernel_size=1,padding=0)
    
    def forward(self, x):   # [1,128,145,145]
        
        x1 = self.relu(self.conv21(self.bn(x)))
        x2 = self.relu(self.conv21(self.bn(x)))
        x2 = self.relu(self.conv22(self.bn1(x2)))
        x2 = self.relu(self.conv23(self.bn1(x2)))
        x3 = self.relu(self.conv3(self.bn(x)))
    
        x = x+x1+x2+x3
     
        return x

class GELU(nn.Module):
    def __init__(self):
        super(GELU, self).__init__()

    def forward(self, x):
        return 0.5*x*(1+torch.tanh(np.sqrt(2/np.pi)*(x+0.044715*torch.pow(x,3))))

class Attention(nn.Module):

    def __init__(self, dim=64, heads=8, dropout=0.1):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5  # 1/sqrt(dim)

        self.to_qkv = nn.Linear(dim, dim * 3, bias=True)  # Wq,Wk,Wv for each vector, thats why *3
        # torch.nn.init.xavier_uniform_(self.to_qkv.weight)
        # torch.nn.init.zeros_(self.to_qkv.bias)

        self.nn1 = nn.Linear(dim, dim)
        # torch.nn.init.xavier_uniform_(self.nn1.weight)
        # torch.nn.init.zeros_(self.nn1.bias)
        self.do1 = nn.Dropout(dropout)
        self.a = torch.nn.Parameter(torch.zeros(1))
        self.b = torch.nn.Parameter(torch.zeros(1))
        self.GELU = GELU()

        # self.FC = nn.Linear()
        self.conv = nn.Conv2d(dim+1, dim+1, kernel_size=1)

    def forward(self, x, mask=None):       # x=[21025,64]
        n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)  # gets q = Q = Wq matmul x1, k = Wk mm x2, v = Wv mm x3
        q, k, v = map(lambda t: rearrange(t, 'n (h d) -> h n d', h=h), qkv)  # split into multi head attentions    [8,21025,8]
        dots = torch.einsum('hid,hjd->hij', q, k) * self.scale      #[8,21025,21025]
        mask_value = -torch.finfo(dots.dtype).max

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, float('-inf'))
            del mask

        attn = dots.softmax(dim=-1)  # follow the softmax,q,d,v equation in the paper

        out = torch.einsum('hij,hjd->hid', attn, v)  # product of v times whatever inside softmax  [8,21025,21025]*[8,21025,8] -> [8,21025,8]
        out = rearrange(out, 'h n d -> n (h d)')  # concat heads into one matrix, ready for next encoder block   [21025,64]
        out = self.nn1(out)
        out = self.do1(out)   
        return out

class MLP_Block(nn.Module):
    def __init__(self, dim=64, hidden_dim=8, dropout=0.1):
        super().__init__()
        self.Linear1 = nn.Linear(dim, hidden_dim)
        self.Linear2 = nn.Linear(hidden_dim, dim)
        self.GELU = GELU()
        self.Dropout = nn.Dropout(dropout)
    def forward(self, x):
        # print(x.shape)
        x = self.Linear1(x)    # [21025,64] -> [21025]
        x = self.GELU(x)
        x = self.Dropout(x)   
        x = self.Linear2(x)    # [21025,64]
        x = self.Dropout(x)

        return x

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout):
        super().__init__()
        self.attention = Attention(dim, heads=heads, dropout=dropout)
        self.norm = nn.LayerNorm(dim)
        self.mlp = MLP_Block(dim, mlp_dim, dropout=dropout)
        self.up = nn.Linear(64,128)

    def forward(self, x1, mask=None):
        # for attention, mlp in self.layers:
        identity = x1          # [21025,64]
        x1 = self.norm(x1)
        x1 = self.attention(x1, mask=mask)  # go to attention   [64, 65, 64]
        # print(x1.shape)
        x1 = x1 + identity
        x22 = self.norm(x1)
        x22 = self.mlp(x22)  # go to MLP_Block
        x = x22 + x1
        x = self.up(x)
        return x

class TFormer(nn.Module):
    def __init__(self, channel):
        super(TFormer, self).__init__()
        self.patch_to_embedding = nn.Linear(128, 64)
        self.pos_embedding1 = nn.Parameter(torch.empty(100, 64))
        self.pos_embedding2 = nn.Parameter(torch.empty(49, 64))
        self.pos_embedding3 = nn.Parameter(torch.empty(2304, 64))
        self.dropout = nn.Dropout(0.1)
        self.transformer = Transformer(dim=64, depth=1, heads=8, mlp_dim=8, dropout=0.1)


    def forward(self,x):
        # h,w,c = x.shape    # [145,145,128]
        # x = x.reshape(h*w,c)      # [21025,128]
        n,_ = x.shape
        x = self.patch_to_embedding(x)   # [145,145,128] -> [21025, 64]
        if x.shape[0]==100:
            x += self.pos_embedding1[:(n + 1)]     # [21025, 64]
        elif x.shape[0]==49:
            x += self.pos_embedding2[:(n + 1)]     # [21025, 64]
        elif x.shape[0]==2304:
            x += self.pos_embedding3[:(n + 1)]     # [21025, 64]
        x = self.dropout(x)
        x = self.transformer(x)

        return x

class Channel_att(nn.Module):
    def __init__(self, channel):
        super(Channel_att, self).__init__()
        self.FC1 = nn.Linear(channel,channel//2)
        self.FC2 = nn.Linear(channel//2,channel)
        self.sigmoid = nn.Sigmoid()
        self.bn = nn.BatchNorm1d(64)
    
    def forward(self,x):  # c, h, w
        x = x.squeeze(0)   # [128,145,145]
        h,w,c = x.permute([1, 2, 0]).shape    # [145,145,128]
        x = x.reshape(h*w,c)  # [21025,128]
        identity = x    # [21025,128]
        x = self.FC1(x)   # [21025,64]
        x = self.bn(x)
        x = self.FC2(x)    # [21025,128]
        x = self.sigmoid(x)
        return (x + identity).reshape(1,h,w,c).permute([0,3, 1, 2])

class HCGN(nn.Module):
    def __init__(self, height: int, width: int, changel: int, class_count: int, Q: torch.Tensor, A: torch.Tensor, Q1: torch.Tensor, A1: torch.Tensor,Q2: torch.Tensor, A2: torch.Tensor, model='normal'):
        super(HCGN, self).__init__()
        # 类别数,即网络最终输出通道数
        self.class_count = class_count  # 类别数
        # 网络输入数据大小
        self.channel = changel
        self.height = height
        self.width = width
        self.Q = Q
        self.A = A
        self.Q1 = Q1
        self.A1 = A1
        self.Q2 = Q2
        self.A2 = A2
        self.model=model
        self.norm_col_Q = Q / (torch.sum(Q, 0, keepdim=True))  # 列归一化Q
        self.norm_col_Q1 = Q1 / (torch.sum(Q1, 0, keepdim=True))  # 列归一化Q
        self.norm_col_Q2 = Q2 / (torch.sum(Q2, 0, keepdim=True))  # 列归一化Q
        
        layers_count=2

        self.TFormer = TFormer(128)
        self.channel_att = Channel_att(128)

        # Spectra Transformation Sub-Network
        self.CNN_denoise = nn.Sequential()
        for i in range(layers_count):
            if i == 0:
                self.CNN_denoise.add_module('CNN_denoise_BN'+str(i),nn.BatchNorm2d(self.channel))
                self.CNN_denoise.add_module('CNN_denoise_Conv'+str(i),nn.Conv2d(self.channel, 128, kernel_size=(1, 1)))
                self.CNN_denoise.add_module('CNN_denoise_Act'+str(i),nn.LeakyReLU())   
            else:
                self.CNN_denoise.add_module('CNN_denoise_BN'+str(i),nn.BatchNorm2d(128),)
                self.CNN_denoise.add_module('CNN_denoise_Conv' + str(i), nn.Conv2d(128, 128, kernel_size=(1, 1)))
                self.CNN_denoise.add_module('CNN_denoise_Act' + str(i), nn.LeakyReLU())
        
        #quzao
        self.CNNConv = CNNConv(128,128)

        # self.SpectralConv = nn.Sequential()
        # for i in range(layers_count):
        #     if i<layers_count-1:
        #         self.SpectralConv.add_module('SpectralConv'+str(i),SpectralConv(128, 128))
        #     else:
        #         self.SpectralConv.add_module('SpectralConv' + str(i), SpectralConv(128, 64))


        # Pixel-level Convolutional Sub-Network
        # self.CNN_Branch = nn.Sequential()
        # for i in range(layers_count):
        #     if i<layers_count-1:
        #         self.CNN_Branch.add_module('CNN_Branch'+str(i),SSConv(128, 128,kernel_size=5))
        #     else:
        #         self.CNN_Branch.add_module('CNN_Branch' + str(i), SSConv(128, 64, kernel_size=5))

        # cross fuse
        self.CNN_Branch11 = SSConv(128,128,kernel_size=1)
        self.CNN_Branch12 = SSConv(128,64,kernel_size=3)
        self.CNN_Branch21 = SSConv1(128,128,kernel_size=5)
        self.CNN_Branch22 = SSConv1(128,64,kernel_size=7)

        # self.CNN_Branch = SSConv(128,64,kernel_size=5)
        # self.CNN_Branch121 = SSConv(128,128,kernel_size=5)
        # self.CNN_Branch21 = SSConv1(128,128,kernel_size=3)
        # self.CNN_Branch22 = SSConv1(128,128,kernel_size=5)


        # self.CNN_Branch1 = nn.Sequential()
        # for i in range(layers_count):
        #     if i<layers_count-1:
        #         self.CNN_Branch1.add_module('CNN_Branch1'+str(i),SSConv1(128, 128,kernel_size=5))
        #     else:
        #         self.CNN_Branch1.add_module('CNN_Branch1' + str(i), SSConv1(128, 64, kernel_size=5))


        # Superpixel-level Graph Sub-Network
        self.GCN_Branch=nn.Sequential()
        for i in range(layers_count):
            if i<layers_count-1:
                self.GCN_Branch.add_module('GCN_Branch'+str(i),GCNLayer(128, 128, self.A))
            else:
                self.GCN_Branch.add_module('GCN_Branch' + str(i), GCNLayer(128, 64, self.A))

        # Superpixel-level Graph Sub-Network
        self.GCN_Branch1=nn.Sequential()
        for i in range(layers_count):
            if i<layers_count-1:
                self.GCN_Branch1.add_module('GCN_Branch'+str(i),GCNLayer(128, 128, self.A1))
            else:
                self.GCN_Branch1.add_module('GCN_Branch' + str(i), GCNLayer(128, 64, self.A1))
        
        # Superpixel-level Graph Sub-Network
        self.GCN_Branch2=nn.Sequential()
        for i in range(layers_count):
            if i<layers_count-1:
                self.GCN_Branch2.add_module('GCN_Branch'+str(i),GCNLayer(128, 128, self.A2))
            else:
                self.GCN_Branch2.add_module('GCN_Branch' + str(i), GCNLayer(128, 64, self.A2))


        # Softmax layer
        self.Softmax_linear =nn.Sequential(nn.Linear(128+128, self.class_count))

        self.cnnfc = nn.Conv2d(128,64,1)
    
    def forward(self, x: torch.Tensor):
        '''
        :param x: H*W*C
        :return: probability_map
        '''
        x = x.squeeze()

        h, w, c = x.shape
        
        # 先去除噪声
        noise = self.CNN_denoise(torch.unsqueeze(x.permute([2, 0, 1]), 0))
        noise =torch.squeeze(noise, 0).permute([1, 2, 0])
        clean_x=noise  #直连

        clean_x_flatten=clean_x.reshape([h * w, -1])
        superpixels_flatten = torch.mm(self.norm_col_Q.t(), clean_x_flatten)  # 低频部分
        superpixels_flatten1 = torch.mm(self.norm_col_Q1.t(), clean_x_flatten)  # 低频部分
        superpixels_flatten2 = torch.mm(self.norm_col_Q2.t(), clean_x_flatten)  # 低频部分
        hx = clean_x   # [145,145,128]

        # Tow GCN Branch
        H1 = superpixels_flatten1
        if self.model=='normal':
            for i in range(len(self.GCN_Branch1)): H1, _ = self.GCN_Branch1[i](H1)
        else:
            for i in range(len(self.GCN_Branch1)): H1, _ = self.GCN_Branch1[i](H1,model='smoothed')
        
        GCN_result1 = torch.matmul(self.Q1, H1)

        # Three GCN Branch
        H2 = superpixels_flatten2
        if self.model=='normal':
            for i in range(len(self.GCN_Branch2)): H2, _ = self.GCN_Branch2[i](H2)
        else:
            for i in range(len(self.GCN_Branch2)): H2, _ = self.GCN_Branch2[i](H2,model='smoothed')
        
        GCN_result2 = torch.matmul(self.Q2, H2)

        # GCN层 1 转化为超像素 x_flat 乘以 列归一化Q
        H = superpixels_flatten
        if self.model=='normal':
            for i in range(len(self.GCN_Branch)): H, _ = self.GCN_Branch[i](H)
        else:
            for i in range(len(self.GCN_Branch)): H, _ = self.GCN_Branch[i](H,model='smoothed')
        
        GCN_result = torch.matmul(self.Q, H)  # 这里self.norm_row_Q == self.Q


        # CNN与GCN分两条支路
        # CNN_result = self.CNNConv(torch.unsqueeze(hx.permute([2, 0, 1]), 0))
        CNN_result11 = self.CNN_Branch11(torch.unsqueeze(hx.permute([2, 0, 1]), 0))
        CNN_result21 = self.CNN_Branch21(torch.unsqueeze(hx.permute([2, 0, 1]), 0))
        # x1 = torch.unsqueeze(hx.permute([2, 0, 1]), 0)
        CNN_result = CNN_result11 + CNN_result21
        # CNN_result = self.CNN_Branch(CNN_result)
        
        # x2 = self.cnnfc(CNN_result)

        CNN_result12 = self.CNN_Branch12(CNN_result)
        CNN_result22 = self.CNN_Branch22(CNN_result)
        CNN_result = CNN_result12 + CNN_result22
        # CNN_result1 = self.CNN_Branch1(torch.unsqueeze(hx.permute([2, 0, 1]), 0))# spectral-spatial convolution
        CNN_result = torch.squeeze(CNN_result, 0).permute([1, 2, 0]).reshape([h * w, -1])
        # CNN_result1 = torch.squeeze(CNN_result1, 0).permute([1, 2, 0]).reshape([h * w, -1])
        CNN_result = torch.squeeze(CNN_result, 0).reshape([h * w, -1])


        # different of CNN 串联
        # CNN_result1 = self.CNN_Branch11(torch.unsqueeze(hx.permute([2, 0, 1]), 0))
        # x1 = torch.unsqueeze(hx.permute([2, 0, 1]), 0)
        # CNN_result1 = CNN_result1 + x1
        # CNN_result2 = self.CNN_Branch12(CNN_result1)
        # CNN_result2 = CNN_result1 + CNN_result2
        # CNN_result3 = self.CNN_Branch21(CNN_result2)
        # CNN_result3 = CNN_result3 + CNN_result2
        # CNN_result = self.CNN_Branch22(CNN_result3)
        # CNN_result = CNN_result + CNN_result3


        # CNN_result = torch.squeeze(CNN_result, 0).permute([1, 2, 0]).reshape([h * w, -1])
        # # CNN_result1 = torch.squeeze(CNN_result1, 0).permute([1, 2, 0]).reshape([h * w, -1])
        # CNN_result = torch.squeeze(CNN_result, 0).reshape([h * w, -1])



        # # 光谱分支
        # Spectral_result = self.SpectralConv(torch.unsqueeze(hx.permute([2, 0, 1]), 0))
        # Spectral_result = torch.squeeze(Spectral_result, 0).permute([1, 2, 0]).reshape([h * w, -1])

        GCN_result0 = torch.cat((GCN_result, GCN_result1, GCN_result2),dim=-1)
        
        # 两组特征融合(两种融合方式)
        Y = torch.cat([CNN_result,GCN_result0],dim=-1)
        Y = self.Softmax_linear(Y)
        Y = F.softmax(Y, -1)
        return Y, CNN_result,GCN_result,GCN_result1, GCN_result2


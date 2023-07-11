import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from convNd import convNd
from UpsampleNearest4D import UpsampleNearest4D

def weights_init(m):
    if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)

def convBlock4D(in_channels, group_size=8, kernel_size=3,activation = 'Relu'):
    
    if activation == 'Relu':
        block = nn.Sequential(
            nn.GroupNorm(num_groups=group_size, num_channels=in_channels),
            nn.ReLU(inplace=True),
            convNd(in_channels=in_channels, out_channels=in_channels, num_dims=4, kernel_size=kernel_size, stride=1,
                   padding=1,
                   kernel_initializer=lambda x: torch.nn.init.xavier_uniform_(x),
                   bias_initializer=lambda x: torch.nn.init.zeros_(x)),
            nn.GroupNorm(num_groups=group_size, num_channels=in_channels),
            nn.ReLU(inplace=True),
            convNd(in_channels=in_channels, out_channels=in_channels, num_dims=4, kernel_size=kernel_size, stride=1,
                   padding=1,
                   kernel_initializer=lambda x: torch.nn.init.xavier_uniform_(x),
                   bias_initializer=lambda x: torch.nn.init.zeros_(x))
        )
    elif activation == 'LeakyRelu':
        block = nn.Sequential(
            nn.GroupNorm(num_groups=group_size, num_channels=in_channels),
            nn.LeakyReLU(inplace=True),
            convNd(in_channels=in_channels, out_channels=in_channels, num_dims=4, kernel_size=kernel_size, stride=1,
                   padding=1,
                   kernel_initializer=lambda x: torch.nn.init.xavier_uniform_(x),
                   bias_initializer=lambda x: torch.nn.init.zeros_(x)),
            nn.GroupNorm(num_groups=group_size, num_channels=in_channels),
            nn.LeakyReLU(inplace=True),
            convNd(in_channels=in_channels, out_channels=in_channels, num_dims=4, kernel_size=kernel_size, stride=1,
                   padding=1,
                   kernel_initializer=lambda x: torch.nn.init.xavier_uniform_(x),
                   bias_initializer=lambda x: torch.nn.init.zeros_(x))
        )
    return block


def upsampleBlock4D(in_channels, out_channels, kernel_size=1):
    block = nn.Sequential(
        convNd(in_channels=in_channels, out_channels=out_channels, num_dims=4, kernel_size=kernel_size, stride=1,
               padding=0,
               kernel_initializer=lambda x: torch.nn.init.xavier_uniform_(x),
               bias_initializer=lambda x: torch.nn.init.zeros_(x)),
        UpsampleNearest4D(scale_factor=2)
    )
    return block


def lastBlock4D(in_channels, out_channels, kernel_size=1):
    block = nn.Sequential(
        convNd(in_channels=in_channels, out_channels=out_channels, num_dims=4, kernel_size=kernel_size, stride=1,
               padding=0,
               kernel_initializer=lambda x: torch.nn.init.xavier_uniform_(x),
               bias_initializer=lambda x: torch.nn.init.zeros_(x)),
        nn.Softmax(dim=1)
    )
    return block


def padAndSum4D(bypass, upsampled, pad=True):
    if pad:
        t_diff = bypass.shape[2] - upsampled.shape[2]
        z_diff = bypass.shape[3] - upsampled.shape[3]
        y_diff = bypass.shape[4] - upsampled.shape[4]
        x_diff = bypass.shape[5] - upsampled.shape[5]
        upsampled = F.pad(upsampled,
                (math.floor(x_diff/2), math.ceil(x_diff/2),
                 math.floor(y_diff/2), math.ceil(y_diff/2),
                 math.floor(z_diff/2), math.ceil(z_diff/2),
                 math.floor(t_diff/2), math.ceil(t_diff/2)))
    return torch.add(bypass, upsampled)


class UNet4DMyronenko(nn.Module):

    def __init__(self, in_channel, out_channel,activation):
        super(UNet4DMyronenko, self).__init__()
        working_channels = 8  # 8
        self.init_conv = convNd(in_channels=in_channel, out_channels=working_channels, num_dims=4, kernel_size=3, stride=1,
                                padding=1,
                                kernel_initializer=lambda x: torch.nn.init.xavier_uniform_(x),
                                bias_initializer=lambda x: torch.nn.init.zeros_(x))
        self.encode_block0 = convBlock4D(in_channels=working_channels, group_size=working_channels, kernel_size=3,activation=activation)
        self.encode_down1 = convNd(in_channels=working_channels,
                                   out_channels=working_channels*2, num_dims=4, kernel_size=3, stride=2,
                                   padding=1,
                                   kernel_initializer=lambda x: torch.nn.init.xavier_uniform_(x),
                                   bias_initializer=lambda x: torch.nn.init.zeros_(x))
        self.encode_block1 = convBlock4D(in_channels=working_channels*2, group_size=working_channels, kernel_size=3,activation=activation)
        self.encode_down2 = convNd(in_channels=working_channels*2,
                                   out_channels=working_channels*2*2, num_dims=4, kernel_size=3, stride=2,
                                   padding=1,
                                   kernel_initializer=lambda x: torch.nn.init.xavier_uniform_(x),
                                   bias_initializer=lambda x: torch.nn.init.zeros_(x))
        self.encode_block2 = convBlock4D(in_channels=working_channels*2*2, group_size=working_channels, kernel_size=3,activation=activation)
        self.decode_up1 = upsampleBlock4D(in_channels=working_channels*2*2, out_channels=working_channels*2, kernel_size=1)
        self.decode_block1 = convBlock4D(in_channels=working_channels*2, group_size=working_channels, kernel_size=3,activation=activation)
        self.decode_up0 = upsampleBlock4D(in_channels=working_channels*2, out_channels=working_channels, kernel_size=1)
        self.decode_block0 = convBlock4D(in_channels=working_channels, group_size=working_channels, kernel_size=3,activation=activation)
        self.decode_end = lastBlock4D(in_channels=working_channels, out_channels=out_channel, kernel_size=1)

    def forward(self, x):
        ic = self.init_conv(x)  # InitConv
        eb0 = self.encode_block0(ic)  # EncoderBlock0
        eb0 = torch.add(eb0, ic)
        ed1 = self.encode_down1(eb0)  # EncoderDown1
        eb10 = self.encode_block1(ed1)  # EncoderBlock1x2
        eb10 = torch.add(eb10, ed1)
        eb11 = self.encode_block1(eb10)  # EncoderBlock1x2
        eb11 = torch.add(eb11, eb10)
        ed2 = self.encode_down2(eb11)  # EncoderDown2
        eb20 = self.encode_block2(ed2)  # EncoderBlock2x4
        eb20 = torch.add(eb20, ed2)
        eb21 = self.encode_block2(eb20)  # EncoderBlock2x4
        eb21 = torch.add(eb21, eb20)
        eb22 = self.encode_block2(eb21)  # EncoderBlock2x4
        eb22 = torch.add(eb22, eb21)
        eb23 = self.encode_block2(eb22)  # EncoderBlock2x4
        eb23 = torch.add(eb23, eb22)
        du1 = self.decode_up1(eb23)  # DecoderUp1
        du1 = padAndSum4D(eb11, du1)
        db1 = self.decode_block1(du1)  # DecoderBlock1
        db1 = torch.add(db1, du1)
        du0 = self.decode_up0(db1)  # DecoderUp0
        du0 = padAndSum4D(eb0, du0)
        db0 = self.decode_block0(du0)  # DecoderBlock0
        db0 = torch.add(db0, du0)
        de = self.decode_end(db0)  # DecoderEnd
        return de


class UNet4DMyronenko_short(nn.Module):
    
    '''Deleted second downsampling and the four convolutional blocks in between'''

    def __init__(self, in_channel, out_channel,activation):
        super(UNet4DMyronenko_short, self).__init__()
        working_channels = 8  # 8
        self.init_conv = convNd(in_channels=in_channel, out_channels=working_channels, num_dims=4, kernel_size=3, stride=1,
                                padding=1,
                                kernel_initializer=lambda x: torch.nn.init.xavier_uniform_(x),
                                bias_initializer=lambda x: torch.nn.init.zeros_(x))
        self.encode_block0 = convBlock4D(in_channels=working_channels, group_size=working_channels, kernel_size=3,activation=activation)
        self.encode_down1 = convNd(in_channels=working_channels,
                                   out_channels=working_channels*2, num_dims=4, kernel_size=3, stride=2,
                                   padding=1,
                                   kernel_initializer=lambda x: torch.nn.init.xavier_uniform_(x),
                                   bias_initializer=lambda x: torch.nn.init.zeros_(x))
        self.encode_block1 = convBlock4D(in_channels=working_channels*2, group_size=working_channels, kernel_size=3,activation=activation)
        self.decode_block1 = convBlock4D(in_channels=working_channels*2, group_size=working_channels, kernel_size=3,activation=activation)
        self.decode_up0 = upsampleBlock4D(in_channels=working_channels*2, out_channels=working_channels, kernel_size=1)
        self.decode_block0 = convBlock4D(in_channels=working_channels, group_size=working_channels, kernel_size=3,activation=activation)
        self.decode_end = lastBlock4D(in_channels=working_channels, out_channels=out_channel, kernel_size=1)

    def forward(self, x):
        ic = self.init_conv(x)  # InitConv
        eb0 = self.encode_block0(ic)  # EncoderBlock0
        eb0 = torch.add(eb0, ic)
        ed1 = self.encode_down1(eb0)  # EncoderDown1
        eb10 = self.encode_block1(ed1)  # EncoderBlock1x2
        eb10 = torch.add(eb10, ed1)
        eb11 = self.encode_block1(eb10)  # EncoderBlock1x2
        eb11 = torch.add(eb11, eb10)
        db1 = self.decode_block1(eb11)  # DecoderBlock1
        db1 = torch.add(db1, eb11)
        du0 = self.decode_up0(db1)  # DecoderUp0
        du0 = padAndSum4D(eb0, du0)
        db0 = self.decode_block0(du0)  # DecoderBlock0
        db0 = torch.add(db0, du0)
        de = self.decode_end(db0)  # DecoderEnd
        return de
    

#########################################################################################
#                                                                                       #
#         4D UNET FROM ANTOINE VERSION 2 FOR COMPARISON WITH MYRONENKI                  #
#                                                                                       #
#######################################################################################

'''https://github.com/nikhilroxtomar/Deep-Residual-Unet/blob/master/Deep%20Residual%20UNet.ipynb'''

class BatchNorm4D(nn.Module):
    def __init__(self):
        super(BatchNorm4D, self).__init__()        
    def forward(self, x):       
        output_tensor6D = torch.empty_like(x)
        for c in range(0,x.shape[1]):         
            output_tensor6D[:,c,:,:,:,:]=(x[:,c,:,:,:,:]-torch.mean(x[:,c,:,:,:,:]))/torch.sqrt(torch.var(x[:,c,:,:,:,:],unbiased=False))
                      
        return output_tensor6D

def conv_block4D(in_channels, out_channels, kernel_size=3):
    block = nn.Sequential(            
            convNd(in_channels=in_channels,out_channels=out_channels//2, num_dims=4, kernel_size=kernel_size, stride=1,padding=1,
                                   kernel_initializer=lambda x: torch.nn.init.xavier_uniform_(x),
                                   bias_initializer=lambda x: torch.nn.init.zeros_(x)),
            nn.ReLU(inplace=True),
            convNd(in_channels=out_channels//2,out_channels=out_channels, num_dims=4, kernel_size=kernel_size, stride=1,padding=1,
                                   kernel_initializer=lambda x: torch.nn.init.xavier_uniform_(x),
                                   bias_initializer=lambda x: torch.nn.init.zeros_(x)),
            
            BatchNorm4D(),
            nn.ReLU(inplace=True)
    )
    return block


def upsample_block4D(in_channels, out_channels, kernel_size=3):
    block = nn.Sequential(            
            convNd(in_channels=in_channels+out_channels,out_channels=out_channels, num_dims=4, kernel_size=kernel_size, stride=1,padding=1,
                                   kernel_initializer=lambda x: torch.nn.init.xavier_uniform_(x),
                                   bias_initializer=lambda x: torch.nn.init.zeros_(x)),
            nn.ReLU(inplace=True),
            convNd(in_channels=out_channels,out_channels=out_channels, num_dims=4, kernel_size=kernel_size, stride=1,padding=1,
                                   kernel_initializer=lambda x: torch.nn.init.xavier_uniform_(x),
                                   bias_initializer=lambda x: torch.nn.init.zeros_(x)),
            BatchNorm4D(),
            nn.ReLU(inplace=True),
            UpsampleNearest4D(scale_factor=2)
    )
    return block

def last_block4D(in_channels, mid_channels, out_channels, kernel_size=3):
    block = nn.Sequential(
            convNd(in_channels=in_channels+mid_channels,out_channels=mid_channels, num_dims=4, kernel_size=kernel_size, stride=1,padding=1,
                                   kernel_initializer=lambda x: torch.nn.init.xavier_uniform_(x),
                                   bias_initializer=lambda x: torch.nn.init.zeros_(x)),
            BatchNorm4D(),
            nn.ReLU(inplace=True),
            convNd(in_channels=mid_channels, out_channels=mid_channels, num_dims=4, kernel_size=kernel_size, stride=1,padding=1,
                                   kernel_initializer=lambda x: torch.nn.init.xavier_uniform_(x),
                                   bias_initializer=lambda x: torch.nn.init.zeros_(x)),
            BatchNorm4D(),
            nn.ReLU(inplace=True),
            convNd(in_channels=mid_channels, out_channels=out_channels, num_dims=4, kernel_size=kernel_size, stride=1,padding=1,
                                   kernel_initializer=lambda x: torch.nn.init.xavier_uniform_(x),
                                   bias_initializer=lambda x: torch.nn.init.zeros_(x)),
            BatchNorm4D(),
            nn.Softmax(dim=1)
    )
    return block

def DownSample4D(in_channels, out_channels, kernel_size=3):    
     block = convNd(in_channels=in_channels, out_channels=out_channels, num_dims=4, kernel_size=kernel_size, stride=2,padding=1,
                                   kernel_initializer=lambda x: torch.nn.init.xavier_uniform_(x),
                                   bias_initializer=lambda x: torch.nn.init.zeros_(x))
     return block
 
def crop_and_concat4D(bypass, upsampled, crop=False):

    if crop:         
        t_diff = bypass.shape[2] - upsampled.shape[2]
        z_diff = bypass.shape[3] - upsampled.shape[3]
        y_diff = bypass.shape[4] - upsampled.shape[4]
        x_diff = bypass.shape[5] - upsampled.shape[5]

        upsampled = F.pad(upsampled,
                (math.floor(x_diff/2), math.ceil(x_diff/2),
                 math.floor(y_diff/2), math.ceil(y_diff/2),
                 math.floor(z_diff/2), math.ceil(z_diff/2),
                 math.floor(t_diff/2), math.ceil(t_diff/2)))
        
    return torch.cat([bypass, upsampled], dim=1)


class UNet4DAntoine2(nn.Module):

    def __init__(self, in_channel, out_channel):
        super(UNet4DAntoine2, self).__init__()  
        self.conv_encode1 = conv_block4D(in_channels=in_channel, out_channels=32)        
        self.Down_1 = DownSample4D(in_channels=32, out_channels=32)
        self.conv_encode2 = conv_block4D(in_channels=32, out_channels=64)        
        self.Down_2 = DownSample4D(in_channels=64, out_channels=64)
        self.conv_encode3 = conv_block4D(in_channels=64, out_channels=128)
        self.Down_3 = DownSample4D(in_channels=128, out_channels=128)
        self.conv_encode4 = conv_block4D(in_channels=128, out_channels=256)
        self.Down_4 = DownSample4D(in_channels=256, out_channels=256)

        self.bottleneck = nn.Sequential(
            convNd(in_channels=256, out_channels=256, num_dims=4, kernel_size=3, stride=1,padding=1,
                                   kernel_initializer=lambda x: torch.nn.init.xavier_uniform_(x),
                                   bias_initializer=lambda x: torch.nn.init.zeros_(x)),
            BatchNorm4D(),
            nn.ReLU(inplace=True),
            convNd(in_channels=256, out_channels=512, num_dims=4, kernel_size=3, stride=1,padding=1,
                                   kernel_initializer=lambda x: torch.nn.init.xavier_uniform_(x),
                                   bias_initializer=lambda x: torch.nn.init.zeros_(x)),
            BatchNorm4D(),
            nn.ReLU(inplace=True),
            UpsampleNearest4D(scale_factor=2)
        )

        self.conv_decode4 = upsample_block4D(in_channels=512, out_channels=256)
        self.conv_decode3 = upsample_block4D(in_channels=256, out_channels=128)
        self.conv_decode2 = upsample_block4D(in_channels=128, out_channels=64)
        self.final_layer = last_block4D(in_channels=64, mid_channels=32, out_channels=out_channel)

    def forward(self, x):
        encode_block1 = self.conv_encode1(x)
        encode_pool1 = self.Down_1(encode_block1)
        encode_block2 = self.conv_encode2(encode_pool1)
        encode_pool2 = self.Down_2(encode_block2)
        encode_block3 = self.conv_encode3(encode_pool2)
        encode_pool3 = self.Down_3(encode_block3)
        encode_block4 = self.conv_encode4(encode_pool3)
        encode_pool4 = self.Down_4(encode_block4)

        bottleneck = self.bottleneck(encode_pool4)

        decode_block4 = crop_and_concat4D(encode_block4, bottleneck, crop=True)
        cat_layer3 = self.conv_decode4(decode_block4)
        decode_block3 = crop_and_concat4D(encode_block3, cat_layer3, crop=True)
        cat_layer2 = self.conv_decode3(decode_block3)
        decode_block2 = crop_and_concat4D(encode_block2, cat_layer2, crop=True)
        cat_layer1 = self.conv_decode2(decode_block2)
        decode_block1 = crop_and_concat4D(encode_block1, cat_layer1, crop=True)

        final_layer = self.final_layer(decode_block1)

        return final_layer
    


#########################################################################################
#                                                                                       #
#                                         4D Res-UNET                                   #
#                                                                                       #
#######################################################################################

'''https://github.com/nikhilroxtomar/Deep-Residual-Unet/blob/master/Deep%20Residual%20UNet.ipynb'''

class bn_act(nn.Module):
    def __init__(self, act=True):
        super(bn_act, self).__init__()
        self.act = act    
        self.bn = BatchNorm4D()
        self.reluF = nn.ReLU(inplace=True)
    def forward(self, x):             
        out = self.bn(x)
        if self.act == True:             
            out = self.reluF(out)         
        return out 
        
        
class conv_block(nn.Module):
    def __init__(self,in_channels, out_channels,kernel_size=3,padding=1,stride=1):
        super(conv_block, self).__init__()
        
        self.convolution = convNd(in_channels=in_channels,out_channels=out_channels, num_dims=4, kernel_size=kernel_size, 
                        stride=stride,padding=padding,
                        kernel_initializer=lambda w: torch.nn.init.xavier_uniform_(w),
                        bias_initializer=lambda w: torch.nn.init.zeros_(w))
        self.batch_norm = bn_act() 
    def forward(self, x):
        bn =  self.batch_norm(x)  
        out = self.convolution(bn)
        
        return out      

            
class stem(nn.Module):
    def __init__(self,in_channels, out_channels,kernel_size=3,padding=1,stride=1):
        super(stem, self).__init__()        
        self.convolution = convNd(in_channels=in_channels,out_channels=out_channels, num_dims=4, kernel_size=kernel_size, 
                        stride=stride,padding=padding,
                        kernel_initializer=lambda w: torch.nn.init.xavier_uniform_(w),
                        bias_initializer=lambda w: torch.nn.init.zeros_(w))
        self.convolution1 = convNd(in_channels=in_channels,out_channels=out_channels, num_dims=4, kernel_size=1, 
                        stride=stride,padding=padding,
                        kernel_initializer=lambda w: torch.nn.init.xavier_uniform_(w),
                        bias_initializer=lambda w: torch.nn.init.zeros_(w))
        self.convBlock = conv_block(out_channels, out_channels,kernel_size=kernel_size,padding=padding,stride=stride)
        self.BnNorm = bn_act(act=False)
    def forward(self, x):
        out = self.convolution(x)
        out = self.convBlock(out)
        shortcut = self.convolution1(x)
        shortcut = self.BnNorm(shortcut)
        output = padAndSum4D(out,shortcut)
        # print("Out_Prev ",out.shape)
        # print("Out_skip_pad ",shortcut.shape)
        # output = padAndSum4D(shortcut,out)
        
        return output         
        
    
class residual_block(nn.Module):
    def __init__(self,in_channels, out_channels,kernel_size=3,padding=1,stride=1):
        super(residual_block, self).__init__()
        
        self.convolution1 = convNd(in_channels=in_channels,out_channels=out_channels, num_dims=4, kernel_size=1, 
                        stride=stride,padding=padding,
                        kernel_initializer=lambda w: torch.nn.init.xavier_uniform_(w),
                        bias_initializer=lambda w: torch.nn.init.zeros_(w))
        self.convBlock0 = conv_block(in_channels, out_channels,kernel_size=kernel_size,padding=padding,stride=stride)
        self.convBlock1 = conv_block(out_channels, out_channels,kernel_size=kernel_size,padding=padding,stride=1)
        self.BnNorm = bn_act(act=False)
    def forward(self, x):
        res = self.convBlock0(x)
        res = self.convBlock1(res)
        # print('x_shape ',x.shape)
        shortcut = self.convolution1(x)
        # print('shorcutConv1_shape ',shortcut.shape)
        shortcut = self.BnNorm(shortcut)        
        
        output = padAndSum4D(res,shortcut)
        
        return output      

    
class upsample_concat(nn.Module):
    def __init__(self):
        super(upsample_concat, self).__init__()    
        self.UpSample = UpsampleNearest4D(scale_factor=2)
    def forward(self, x,y):       
        u = self.UpSample(x)   
        # print('Upshape ',u.shape)
        # print('Skip ',y.shape)
        c = crop_and_concat4D(y,u,crop=True)
        # c = torch.cat([u, y], dim=1)
        return c     
  
    
class final_conv(nn.Module):
    def __init__(self,in_channels, out_channels,kernel_size=3,padding=1,stride=1):
        super(final_conv, self).__init__()
        
        self.convolution = convNd(in_channels=in_channels,out_channels=out_channels, num_dims=4, kernel_size=kernel_size, 
                        stride=stride,padding=padding,
                        kernel_initializer=lambda w: torch.nn.init.xavier_uniform_(w),
                        bias_initializer=lambda w: torch.nn.init.zeros_(w))
        self.sofmaxF = nn.Softmax(dim=1) 
    def forward(self, x):
        last_conv = self.convolution(x)
        out = self.sofmaxF(last_conv)
        return out   

    
class Residual_UNet4D(nn.Module):
    

    def __init__(self, in_channel, out_channel):
        super(Residual_UNet4D, self).__init__()  
        self.stem_block = stem(in_channel,64)
        self.residual_block_e0 = residual_block(64,128,stride=2)
        self.residual_block_e1 = residual_block(128,256,stride=2)
        self.boottle_neck = residual_block(256,512,stride=2)
          
        self.upsample_block = upsample_concat()
        self.residual_block_d0 = residual_block(512+256,256)
        self.residual_block_d1 = residual_block(256+128,128)
        self.residual_block_d2 = residual_block(128+64,64)
 

    def forward(self, x):
        init_stem = self.stem_block(x)
        e0 = self.residual_block_e0(init_stem)
        e1 = self.residual_block_e1(e0)
                        
        bn = self.boottle_neck(e1)
               
        u0 = self.upsample_block(bn,e1)
        d0 = self.residual_block_d0(u0)
        
        u1 = self.upsample_block(d0,e0)
        d1 = self.residual_block_d1(u1)
        
        u2 = self.upsample_block(d1,init_stem)
        d2 = self.residual_block_d2(u2)
          
        out  = self.final_block(d2)
    
        return out    


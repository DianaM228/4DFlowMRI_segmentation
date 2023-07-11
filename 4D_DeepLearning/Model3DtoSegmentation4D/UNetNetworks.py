import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn import init


def weights_init(m):
    if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)
            
def weights_init2(m):
    if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
        torch.manual_seed(10)
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.manual_seed(10)
            torch.nn.init.zeros_(m.bias)            


def conv_block(in_channels, out_channels, kernel_size=3):
    block = nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=out_channels//2, kernel_size=kernel_size, stride=1, padding=1, bias=True),
            nn.BatchNorm3d(num_features=out_channels//2),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=out_channels//2, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=1, bias=True),
            nn.BatchNorm3d(num_features=out_channels),
            nn.ReLU(inplace=True)
    )
    return block


def deconv_block(in_channels, out_channels, kernel_size=3):
    block = nn.Sequential(
            nn.Conv3d(in_channels=in_channels+out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=1, bias=True),
            nn.BatchNorm3d(num_features=out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=1, bias=True),
            nn.BatchNorm3d(num_features=out_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(in_channels=out_channels, out_channels=out_channels, kernel_size=2, stride=2, bias=True)
    )
    return block


def upsample_block(in_channels, out_channels, kernel_size=3):
    block = nn.Sequential(
            nn.Conv3d(in_channels=in_channels+out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=1, bias=True),
            nn.BatchNorm3d(num_features=out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=1, bias=True),
            nn.BatchNorm3d(num_features=out_channels),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
    )
    return block


def last_block(in_channels, mid_channels, out_channels, kernel_size=3):
    block = nn.Sequential(
            nn.Conv3d(in_channels=in_channels+mid_channels, out_channels=mid_channels, kernel_size=kernel_size, stride=1, padding=1, bias=True),
            nn.BatchNorm3d(num_features=mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=mid_channels, out_channels=mid_channels, kernel_size=kernel_size, stride=1, padding=1, bias=True),
            nn.BatchNorm3d(num_features=mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=mid_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=1, bias=True),
            nn.BatchNorm3d(num_features=out_channels),
            nn.Softmax(dim=1)
    )
    return block


def crop_and_concat(bypass, upsampled, crop=False):
    
    if crop:
        z_diff = bypass.shape[2] - upsampled.shape[2]
        y_diff = bypass.shape[3] - upsampled.shape[3]
        x_diff = bypass.shape[4] - upsampled.shape[4]
        upsampled = F.pad(upsampled,
                (math.floor(x_diff/2), math.ceil(x_diff/2),
                 math.floor(y_diff/2), math.ceil(y_diff/2),
                 
                 math.floor(z_diff/2), math.ceil(z_diff/2)))
    return torch.cat([bypass, upsampled], dim=1)


class UNet3DAntoine(nn.Module):

    def __init__(self, in_channel, out_channel):
        super(UNet3DAntoine, self).__init__()

        self.conv_encode1 = conv_block(in_channels=in_channel, out_channels=32)
        self.conv_maxpool1 = nn.MaxPool3d(kernel_size=2)
        self.conv_encode2 = conv_block(in_channels=32, out_channels=64)
        self.conv_maxpool2 = nn.MaxPool3d(kernel_size=2)
        self.conv_encode3 = conv_block(in_channels=64, out_channels=128)
        self.conv_maxpool3 = nn.MaxPool3d(kernel_size=2)
        self.conv_encode4 = conv_block(in_channels=128, out_channels=256)
        self.conv_maxpool4 = nn.MaxPool3d(kernel_size=2)

        self.bottleneck = nn.Sequential(
            nn.Conv3d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm3d(num_features=256),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm3d(num_features=512),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(in_channels=512, out_channels=512, kernel_size=2, stride=2, bias=True)
        )

        self.conv_decode4 = deconv_block(in_channels=512, out_channels=256)
        self.conv_decode3 = deconv_block(in_channels=256, out_channels=128)
        self.conv_decode2 = deconv_block(in_channels=128, out_channels=64)
        self.final_layer = last_block(in_channels=64, mid_channels=32, out_channels=out_channel)

    def forward(self, x):
        encode_block1 = self.conv_encode1(x)
        encode_pool1 = self.conv_maxpool1(encode_block1)
        encode_block2 = self.conv_encode2(encode_pool1)
        encode_pool2 = self.conv_maxpool2(encode_block2) 
        
        encode_block3 = self.conv_encode3(encode_pool2)        
        encode_pool3 = self.conv_maxpool3(encode_block3)
        encode_block4 = self.conv_encode4(encode_pool3)
        encode_pool4 = self.conv_maxpool4(encode_block4)
        
        bottleneck = self.bottleneck(encode_pool4)

        decode_block4 = crop_and_concat(encode_block4, bottleneck, crop=True)
        cat_layer3 = self.conv_decode4(decode_block4)
        decode_block3 = crop_and_concat(encode_block3, cat_layer3, crop=True)
        cat_layer2 = self.conv_decode3(decode_block3)
        decode_block2 = crop_and_concat(encode_block2, cat_layer2, crop=True)
        cat_layer1 = self.conv_decode2(decode_block2)
        decode_block1 = crop_and_concat(encode_block1, cat_layer1, crop=True)
        
        final_layer = self.final_layer(decode_block1)
        
        return final_layer


class UNet3DAntoine2(nn.Module):

    def __init__(self, in_channel, out_channel):
        super(UNet3DAntoine2, self).__init__()

        self.conv_encode1 = conv_block(in_channels=in_channel, out_channels=32)
        self.conv_maxpool1 = nn.MaxPool3d(kernel_size=2)
        self.conv_encode2 = conv_block(in_channels=32, out_channels=64)
        self.conv_maxpool2 = nn.MaxPool3d(kernel_size=2)
        self.conv_encode3 = conv_block(in_channels=64, out_channels=128)
        self.conv_maxpool3 = nn.MaxPool3d(kernel_size=2)
        self.conv_encode4 = conv_block(in_channels=128, out_channels=256)
        self.conv_maxpool4 = nn.MaxPool3d(kernel_size=2)

        self.bottleneck = nn.Sequential(
            nn.Conv3d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm3d(num_features=256),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm3d(num_features=512),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        )

        self.conv_decode4 = upsample_block(in_channels=512, out_channels=256)
        self.conv_decode3 = upsample_block(in_channels=256, out_channels=128)
        self.conv_decode2 = upsample_block(in_channels=128, out_channels=64)
        self.final_layer = last_block(in_channels=64, mid_channels=32, out_channels=out_channel)

    def forward(self, x):
        encode_block1 = self.conv_encode1(x)
        encode_pool1 = self.conv_maxpool1(encode_block1)
        encode_block2 = self.conv_encode2(encode_pool1)
        encode_pool2 = self.conv_maxpool2(encode_block2)
        encode_block3 = self.conv_encode3(encode_pool2)
        encode_pool3 = self.conv_maxpool3(encode_block3)
        encode_block4 = self.conv_encode4(encode_pool3)
        encode_pool4 = self.conv_maxpool4(encode_block4)

        bottleneck = self.bottleneck(encode_pool4)


        decode_block4 = crop_and_concat(encode_block4, bottleneck, crop=True)
        
        cat_layer3 = self.conv_decode4(decode_block4)
        decode_block3 = crop_and_concat(encode_block3, cat_layer3, crop=True)
        cat_layer2 = self.conv_decode3(decode_block3)
        decode_block2 = crop_and_concat(encode_block2, cat_layer2, crop=True)
        cat_layer1 = self.conv_decode2(decode_block2)
        decode_block1 = crop_and_concat(encode_block1, cat_layer1, crop=True)

        final_layer = self.final_layer(decode_block1)

        return final_layer

def weights_init_normal(m):
    classname = m.__class__.__name__
    #print(classname)
    if classname.find('Conv') != -1:
        init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_xavier(m):
    classname = m.__class__.__name__
    #print(classname)
    if classname.find('Conv') != -1:
        init.xavier_normal(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.xavier_normal(m.weight.data, gain=1)
    elif classname.find('BatchNorm') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    #print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    #print(classname)
    if classname.find('Conv') != -1:
        init.orthogonal(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.orthogonal(m.weight.data, gain=1)
    elif classname.find('BatchNorm') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def init_weights(net, init_type='normal'):
    #print('initialization method [%s]' % init_type)
    if init_type == 'normal':
        net.apply(weights_init_normal)
    elif init_type == 'xavier':
        net.apply(weights_init_xavier)
    elif init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        
        
        
class UnetGridGatingSignal3(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=(1,1,1), is_batchnorm=True):
        super(UnetGridGatingSignal3, self).__init__()

        if is_batchnorm:
            self.conv1 = nn.Sequential(nn.Conv3d(in_size, out_size, kernel_size, (1,1,1), (0,0,0)),
                                       nn.BatchNorm3d(out_size),
                                       nn.ReLU(inplace=True),
                                       )
        else:
            self.conv1 = nn.Sequential(nn.Conv3d(in_size, out_size, kernel_size, (1,1,1), (0,0,0)),
                                       nn.ReLU(inplace=True),
                                       )

        # initialise the blocks
        for m in self.children():
            init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        return outputs


class _GridAttentionBlockND(nn.Module):
    def __init__(self, in_channels, gating_channels, inter_channels=None, dimension=3, mode='concatenation',
                 sub_sample_factor=(2,2,2)):
        super(_GridAttentionBlockND, self).__init__()

        assert dimension in [2, 3]
        assert mode in ['concatenation', 'concatenation_debug', 'concatenation_residual']

        # Downsampling rate for the input featuremap
        if isinstance(sub_sample_factor, tuple): self.sub_sample_factor = sub_sample_factor
        elif isinstance(sub_sample_factor, list): self.sub_sample_factor = tuple(sub_sample_factor)
        else: self.sub_sample_factor = tuple([sub_sample_factor]) * dimension

        # Default parameter set
        self.mode = mode
        self.dimension = dimension
        self.sub_sample_kernel_size = self.sub_sample_factor

        # Number of channels (pixel dimensions)
        self.in_channels = in_channels
        self.gating_channels = gating_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            bn = nn.BatchNorm3d
            self.upsample_mode = 'trilinear'
        elif dimension == 2:
            conv_nd = nn.Conv2d
            bn = nn.BatchNorm2d
            self.upsample_mode = 'bilinear'
        else:
            raise NotImplementedError

        # Output transform
        self.W = nn.Sequential(
            conv_nd(in_channels=self.in_channels, out_channels=self.in_channels, kernel_size=1, stride=1, padding=0),
            bn(self.in_channels),
        )

        # Theta^T * x_ij + Phi^T * gating_signal + bias
        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=self.sub_sample_kernel_size, stride=self.sub_sample_factor, padding=0, bias=False)
        self.phi = conv_nd(in_channels=self.gating_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0, bias=True)
        self.psi = conv_nd(in_channels=self.inter_channels, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True)

        # Initialise weights
        for m in self.children():
            init_weights(m, init_type='kaiming')

        # Define the operation
        if mode == 'concatenation':
            self.operation_function = self._concatenation
        elif mode == 'concatenation_debug':
            self.operation_function = self._concatenation_debug
        elif mode == 'concatenation_residual':
            self.operation_function = self._concatenation_residual
        else:
            raise NotImplementedError('Unknown operation function.')


    def forward(self, x, g):
        '''
        :param x: (b, c, t, h, w)
        :param g: (b, g_d)
        :return:
        '''

        output = self.operation_function(x, g)
        return output

    def _concatenation(self, x, g):
        input_size = x.size()
        batch_size = input_size[0]
        assert batch_size == g.size(0)

        # theta => (b, c, t, h, w) -> (b, i_c, t, h, w) -> (b, i_c, thw)
        # phi   => (b, g_d) -> (b, i_c)
        theta_x = self.theta(x)
        theta_x_size = theta_x.size()

        # g (b, c, t', h', w') -> phi_g (b, i_c, t', h', w')
        #  Relu(theta_x + phi_g + bias) -> f = (b, i_c, thw) -> (b, i_c, t/s1, h/s2, w/s3)
        phi_g = F.upsample(self.phi(g), size=theta_x_size[2:], mode=self.upsample_mode)
        f = F.relu(theta_x + phi_g, inplace=True)

        #  psi^T * f -> (b, psi_i_c, t/s1, h/s2, w/s3)
        sigm_psi_f = torch.sigmoid(self.psi(f))

        # upsample the attentions and multiply
        sigm_psi_f = F.upsample(sigm_psi_f, size=input_size[2:], mode=self.upsample_mode)
        y = sigm_psi_f.expand_as(x) * x
        W_y = self.W(y)

        return W_y, sigm_psi_f


class GridAttentionBlock3D(_GridAttentionBlockND):
    def __init__(self, in_channels, gating_channels, inter_channels=None, mode='concatenation',
                 sub_sample_factor=(2,2,2)):
        super(GridAttentionBlock3D, self).__init__(in_channels,
                                                   inter_channels=inter_channels,
                                                   gating_channels=gating_channels,
                                                   dimension=3, mode=mode,
                                                   sub_sample_factor=sub_sample_factor,
                                                   )
        
        
class UNet3DAntoine2_Attention(nn.Module):

    def __init__(self, in_channel, out_channel):
        super(UNet3DAntoine2_Attention, self).__init__()

        self.conv_encode1 = conv_block(in_channels=in_channel, out_channels=32)
        self.conv_maxpool1 = nn.MaxPool3d(kernel_size=2)
        self.conv_encode2 = conv_block(in_channels=32, out_channels=64)
        self.conv_maxpool2 = nn.MaxPool3d(kernel_size=2)
        self.conv_encode3 = conv_block(in_channels=64, out_channels=128)
        self.conv_maxpool3 = nn.MaxPool3d(kernel_size=2)
        self.conv_encode4 = conv_block(in_channels=128, out_channels=256)
        self.conv_maxpool4 = nn.MaxPool3d(kernel_size=2)

        self.bottleneck = nn.Sequential(
            nn.Conv3d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm3d(num_features=256),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm3d(num_features=512),
            nn.ReLU(inplace=True)            
        )        
        
        
        self.gating = UnetGridGatingSignal3(512, 256, kernel_size=(1, 1, 1))

        # attention blocks
        self.attentionblock2 = GridAttentionBlock3D(in_channels=64, gating_channels=256,inter_channels=64)
        self.attentionblock3 = GridAttentionBlock3D(in_channels=128, gating_channels=256,inter_channels=128)
        self.attentionblock4 = GridAttentionBlock3D(256, gating_channels=256,inter_channels=256)
        
        self.UpCenter=nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        
        self.conv_decode4 = upsample_block(in_channels=512, out_channels=256)
        self.conv_decode3 = upsample_block(in_channels=256, out_channels=128)
        self.conv_decode2 = upsample_block(in_channels=128, out_channels=64)
        self.final_layer = last_block(in_channels=64, mid_channels=32, out_channels=out_channel)

    def forward(self, x):
        encode_block1 = self.conv_encode1(x)
        encode_pool1 = self.conv_maxpool1(encode_block1)
        encode_block2 = self.conv_encode2(encode_pool1)
        encode_pool2 = self.conv_maxpool2(encode_block2)
        encode_block3 = self.conv_encode3(encode_pool2)
        encode_pool3 = self.conv_maxpool3(encode_block3)
        encode_block4 = self.conv_encode4(encode_pool3)
        encode_pool4 = self.conv_maxpool4(encode_block4)

        bottleneck = self.bottleneck(encode_pool4)
        gating = self.gating(bottleneck)
        
        # Attention Mechanism
        g_conv4, att4 = self.attentionblock4(encode_block4, gating)
        g_conv3, att3 = self.attentionblock3(encode_block3, gating)
        g_conv2, att2 = self.attentionblock2(encode_block2, gating)
        
        Up_bottleneck = self.UpCenter(bottleneck)

        decode_block4 = crop_and_concat(g_conv4, Up_bottleneck, crop=True)
        
        cat_layer3 = self.conv_decode4(decode_block4)
        decode_block3 = crop_and_concat(g_conv3, cat_layer3, crop=True)
        cat_layer2 = self.conv_decode3(decode_block3)
        decode_block2 = crop_and_concat(g_conv2, cat_layer2, crop=True)
        cat_layer1 = self.conv_decode2(decode_block2)
        decode_block1 = crop_and_concat(encode_block1, cat_layer1, crop=True)

        final_layer = self.final_layer(decode_block1)

        return final_layer
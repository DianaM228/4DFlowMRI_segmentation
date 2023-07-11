import torch.nn as nn
import math


class UpsampleNearest4D(nn.Module):
    """Some Information about UpsampleNearest4D"""
    def __init__(self, scale_factor: int):
        super(UpsampleNearest4D, self).__init__()

        # ---------------------------------------------------------------------
        # Store constructor arguments
        # ---------------------------------------------------------------------
        self.scale_factor=scale_factor

    def forward(self, input):
        # B, C, T, Z, Y, X
        input_shape=input.shape
        assert len(input_shape) == 6, "Only works with 4D images, i.e. 6D tensors = B,C,T,Z,Y,X"
        output_shape = [input_shape[0], input_shape[1],
                        self.scale_factor * input_shape[2],
                        self.scale_factor * input_shape[3],
                        self.scale_factor * input_shape[4],
                        self.scale_factor * input_shape[5]]
        output = input.new_zeros(output_shape)
        # 4 for loops
        t_factor = input_shape[2] / output_shape[2]
       
        for t in range(output_shape[2]):
            old_t = math.floor(t * t_factor)
            output[:, :, t, :, :, :] = nn.functional.interpolate(input[:, :, old_t, :, :, :],
                                                                 size=(output_shape[3],output_shape[4],output_shape[5]))#
        return output











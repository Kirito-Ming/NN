""" Full assembly of the parts to form the complete network """

from unet_parts import *
import numpy as np
import torch
from torch.utils import checkpoint

class UNet(nn.Module):
    def __init__(self, inChannel, paraNum, parameter, imageSize, mode="bilinear") -> None:
        super().__init__()

        self.parameter = parameter
        self.mode = mode
        self.paraTensor = torch.ones(paraNum,imageSize//2,imageSize//2)
        for i in range(paraNum):
            self.paraTensor[i] *= parameter[i]
        
        self.input_1 = input_block(inChannel=inChannel, outChannel=32, midChannel=32)
        self.down_1 = down_block(inChannel=32+paraNum, outChannel=64, midChannel=64)
        self.down_2 = down_block(inChannel=64+paraNum, outChannel=128, midChannel=128)
        self.down_3 = down_block(inChannel=128+paraNum, outChannel=256, midChannel=256)
        self.down_4 = down_block(inChannel=256+paraNum, outChannel=512, midChannel=512)

        self.up_1 = up_block(inChannel=512,outChannel=256,midChannel=256)
        self.up_2 = up_block(inChannel=256,outChannel=128,midChannel=128)
        self.up_3 = up_block(inChannel=128,outChannel=64,midChannel=64)
        self.up_4 = up_block(inChannel=64,outChannel=32,midChannel=32)

        self.output_1 = output_block(inChannel=32,midChannel=12,shuffle=2)
    
    def forward(self, x):
        x1 = self.input_1(x)

        x2 = torch.cat((x1,self.paraTensor),dim=1)
        x2 = self.down_1(x2)

        self.paraTensor = self.scaledTensor(self.paraTensor,2)
        x3 = torch.cat((x2,self.paraTensor),dim=1)
        x3 = self.down_2(x3)

        self.paraTensor = self.scaledTensor(self.paraTensor,2)
        x4 = torch.cat((x3,self.paraTensor),dim=1)
        x4 = self.down_3(x4)

        self.paraTensor = self.scaledTensor(self.paraTensor,2)
        x5 = torch.cat((x4,self.paraTensor),dim=1)
        x5 = self.down_4(x5)

        xo = self.up_1(x5,x4,self.mode)
        xo = self.up_2(xo,x3,self.mode)
        xo = self.up_3(xo,x2,self.mode)
        xo = self.up_4(xo,x1,self.mode)

        xo = self.output_1(xo)

    def scaledTensor(self, t, scaleFactor):
        t = torch.narrow(t,1,0,len(t[0])//scaleFactor)
        t = torch.narrow(t,2,0,len(t[0][0])//scaleFactor)
        return t

    def use_checkpointing(self):
        self.input_1 = checkpoint.checkpoint(self.input_1)
        self.down_1 = checkpoint.checkpoint(self.down_1)
        self.down_2 = checkpoint.checkpoint(self.down_2)
        self.down_3 = checkpoint.checkpoint(self.down_3)
        self.down_4 = checkpoint.checkpoint(self.down_4)
        self.up_1 = checkpoint.checkpoint(self.up_1)
        self.up_2 = checkpoint.checkpoint(self.up_2)
        self.up_3 = checkpoint.checkpoint(self.up_3)
        self.up_4 = checkpoint.checkpoint(self.up_4)
        self.output_1 = checkpoint.checkpoint(self.output_1)
    

if __name__ == '__main__':
    parameter = torch.ones(4,8,8)
    print('\nparameter is \n',parameter)
    for i in range(len(parameter)):
        parameter[i] *= i
    print('\nparameter new is \n',parameter)
    unet = UNet(1,1,[],1)
    for i in range(3):
        parameter = unet.scaledTensor(parameter,2)
        print(f'\nparameter {i} is \n',parameter)

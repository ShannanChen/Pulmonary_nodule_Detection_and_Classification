# coding:utf8
from __future__ import print_function
#from .module import Module
import torch as t
import time
import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from layerincep import Inception_v1, Inception_v2, BasicConv, Deconv, SingleConv,res_conc_block

import torch
from torch import nn
from layers import *
from module import Module

config = {}
config['anchors'] =[5., 10., 20.]  #[ 10.0, 30.0, 60.]
config['chanel'] = 1
config['crop_size'] = [96, 96, 96]
config['stride'] = 4
config['max_stride'] = 16
config['num_neg'] = 800
config['th_neg'] = 0.02
config['th_pos_train'] = 0.5
config['th_pos_val'] = 1
config['num_hard'] = 2
config['bound_size'] = 12
config['reso'] = 1
config['sizelim'] = 2.5 #3 #6. #mm
config['sizelim2'] = 10 #30
config['sizelim3'] = 20 #40
config['aug_scale'] = True
config['r_rand_crop'] = 0.3
config['pad_value'] = 170
config['augtype'] = {'flip':True,'swap':False,'scale':True,'rotate':False}


config['augtype'] = {'flip':True,'swap':False,'scale':True,'rotate':False}
config['blacklist'] = ['868b024d9fa388b7ddab12ec1c06af38','990fbe3f0a1b53878669967b9afd1441','adc3bbc63d40f8761c59be10f1e504c3']



#######################zyc  res-#incep
class Net(Module):
    def __init__(self):
        super(Net, self).__init__()
        self.model_name = "seg"
        self.conv1 = BasicConv(1, 24)
        self.downsample1 = Inception_v1(24, 32)
        self.conv2 = res_conc_block(32, 32)
        self.downsample2 = Inception_v1(32, 64)
        self.conv3 = res_conc_block(64, 64)
        self.downsample3 = Inception_v1(64,64)#64, 128)
        self.conv4 = res_conc_block(64,64)#128, 128)
        self.downsample4 = Inception_v1(64,64)#64,64)#64,128)#128, 128)

        self.conv4_ = SingleConv(64, 64)#128, 128)
        self.incept4 = res_conc_block(64, 64)#64,64)#128, 128)
        self.deconv4 = Deconv(64, 64)#64,64)#128, 128)

        self.conv5 = SingleConv(128, 64)#128,64)#192, 128)
        self.incept5 = res_conc_block(64, 64)#64,64)#128, 128)
        self.deconv5 = Deconv(64, 64)#64,64)#128, 64)


        self.drop = nn.Dropout3d(p=0.5, inplace=False)    #default 0.5
        self.output = nn.Sequential(nn.Conv3d(131, 64, kernel_size=1),
                                    nn.ReLU(),
                                    #nn.Dropout3d(p = 0.2),
                                    nn.Conv3d(64, 5 * len(config['anchors']), kernel_size=1))


    def forward(self, x, coord):
        conv1 = self.conv1(x) #(96 96 96) # (64，64，64)

        down1 = self.downsample1(conv1) #(32,48*48*48) # (32,32,32)

        conv2 = self.conv2(down1)  # (32,32,32)

        down2 = self.downsample2(conv2)  # (16,16,16)

        conv3 = self.conv3(down2)  # (16,16,16)

        down3 = self.downsample3(conv3)  # (8,8,8)

        conv4 = self.conv4(down3)  # (8,8,8)

        down4 = self.downsample4(conv4)  # (4,4,4)
       # print('down4', down4.size())

        conv4_ = self.incept4(self.conv4_(down4))
        #print('conv4_', conv4_.size())

        up4 = self.deconv4(conv4_)  # (8,8,8)
        # print ('up4',up4.size())
        up4 = t.cat((up4, conv4), 1)
        # print('up4', up4.size())

        conv5 = self.incept5(self.conv5(up4))
        # print('conv5', conv5.size())
        up5 = self.deconv5(conv5)  # (16,16,16)
        # print('up5', up5.size())
        up5 = t.cat((up5, conv3,coord), 1)

        comb2 = self.drop(up5)
        out = self.output(comb2)
        size = out.size()
        out = out.view(out.size(0), out.size(1), -1)
        # out = out.transpose(1, 4).transpose(1, 2).transpose(2, 3).contiguous()
        out = out.transpose(1, 2).contiguous().view(size[0], size[2], size[3], size[4], len(config['anchors']), 5)
        # out = out.view(-1, 5)
        return out




def get_model():
    net = Net()
    loss = Loss(config['num_hard'])
    get_pbb = GetPBB(config)
    return config, net, loss, get_pbb

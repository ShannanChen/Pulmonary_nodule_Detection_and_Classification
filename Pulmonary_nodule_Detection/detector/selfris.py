# coding:utf8
'''
常用的层,比如inception block,residual block
'''

# coding:utf8
from __future__ import print_function
import argparse
import torch
import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.autograd import Variable



######################selayer#############################
class SELayer3D(nn.Module):
    def __init__(self, channel, reduction):
        super(SELayer3D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(channel // reduction, channel),
                nn.Sigmoid()
        )

    def forward(self, x):
        # print '##################'
        b, c, w, h ,d ,= x.size()
        x1=x.view(b, c,w,-1)

        y = self.avg_pool(x1).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        x2= (x1 * y).view(b, c, w, h ,d )


        return x2

###################################

class se_res_conc_block(nn.Module):
    # '''
    # 残差链接模块
    # 分支1：3*3，stride=1的卷积
    # 分支2:1*1，stride=1的卷积，3*3，stride=1的卷积
    # 分支3：1*1，stride=1的卷积，3*3，stride=1的卷积，3*3，stride=1的卷积
    # 分支1,2,3concat到一起，1*1，stride=1卷积
    # 最后在与input相加
    # '''
    def __init__(self, cin, cn, norm=True, relu=True):
        super(se_res_conc_block, self).__init__()
        self.activa = nn.Sequential()
        if norm: self.activa.add_module('norm', nn.BatchNorm3d(3 * cn))
        if relu: self.activa.add_module('relu', nn.ReLU(True))
        self.branch1 = nn.Conv3d(cin, cn, 3, padding=1)
        self.branch2 = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv3d(cin, cn, 1)),
            ('norm1', nn.BatchNorm3d(cn)),
            ('relu1', nn.ReLU(inplace=True)),
            ('conv2', nn.Conv3d(cn, cn, 3, stride=1, padding=1)),
        ]))
        self.branch3 = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv3d(cin, cn, 1)),
            ('norm1', nn.BatchNorm3d(cn)),
            ('relu1', nn.ReLU(inplace=True)),
            ('conv2', nn.Conv3d(cn, cn, 3, stride=1, padding=1)),
            ('norm2', nn.BatchNorm3d(cn)),
            ('relu2', nn.ReLU(inplace=True)),
            ('conv3', nn.Conv3d(cn, cn, 3, stride=1, padding=1)),
        ]))
        self.merge = nn.Conv3d(3 * cn, cin, 1, 1)

        self.se = SELayer3D(cn, 16)

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        result = torch.cat((branch1, branch2, branch3), 1)
        result = self.activa(result)
        #################se
        se = self.se(x)

        return x + self.merge(se)


# ################################


class Deconv(nn.Module):
    def __init__(self, cin, cout):
        super(Deconv, self).__init__()
        self.model = nn.Sequential(OrderedDict([
            ('deconv1', nn.ConvTranspose3d(cin, cout, 2, stride=2)),
            ('norm', nn.BatchNorm3d(cout)),
            ('relu', nn.ReLU(inplace=True)),
        ]))

    def forward(self, x):
        return self.model(x)


class SingleConv(nn.Module):
    def __init__(self, cin, cout, padding=1):
        super(SingleConv, self).__init__()
        self.padding = padding
        self.model = nn.Sequential(OrderedDict([
            ('conv1_1', nn.Conv3d(cin, cout, 3, padding=self.padding)),
            ('norm1_1', nn.BatchNorm3d(cout)),
            ('relu1_1', nn.ReLU(inplace=True)),
        ]))

    def forward(self, x):
        return self.model(x)


class BasicConv(nn.Module):
    def __init__(self, cin, cout):
        super(BasicConv, self).__init__()
        self.model = nn.Sequential(OrderedDict([
            ('conv1_1', nn.Conv3d(cin, cout, 3, padding=1)),
            ('norm1_1', nn.BatchNorm3d(cout)),
            ('relu1_1', nn.ReLU(inplace=True)),
            ('conv1_2', nn.Conv3d(cout, cout, 3, padding=1)),
            ('norm1_2', nn.BatchNorm3d(cout)),
            ('relu1_2', nn.ReLU(inplace=True)),
        ]))

    def forward(self, x):
        return self.model(x)


class res_conc_block2(nn.Module):
    # '''
    # 残差链接模块
    # 分支1：3*3，stride=1的卷积
    # 分支2:1*1，stride=1的卷积，3*3，stride=1的卷积
    # 分支3：1*1，stride=1的卷积，3*3，stride=1的卷积，3*3，stride=1的卷积
    # 分支1,2,3concat到一起，1*1，stride=1卷积
    # 最后在与input相加
    # '''
    def __init__(self, cin, cn, norm=True, relu=True):
        super(res_conc_block2, self).__init__()
        self.activa = nn.Sequential()
        if norm: self.activa.add_module('norm', nn.BatchNorm3d(cn))
        if relu: self.activa.add_module('relu', nn.ReLU(True))

        assert (cn % 2 == 0)
        cos = [cn / 2] * 2   #[32,32]
        self.activa = nn.Sequential()
        if norm: self.activa.add_module('norm', nn.BatchNorm3d(cn))
        if relu: self.activa.add_module('relu', nn.ReLU(True))


        self.branch1 = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv3d(cin, cos[0], 1)),
            ('norm1', nn.BatchNorm3d(cos[0])),
            ('relu1', nn.ReLU(inplace=True)),
            ('conv2', nn.Conv3d(cos[0], cos[0], 3, stride=1, padding=1)),
        ]))

        self.branch2 = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv3d(cin, cos[1], 1, stride=1)),
            ('norm1', nn.BatchNorm3d(cos[1])),
            ('relu1', nn.ReLU(inplace=True)),
            ('conv2', nn.Conv3d(cos[1], cos[1], 3, stride=1, padding=1)),
            ('norm2', nn.BatchNorm3d(cos[1])),
            ('relu2', nn.ReLU(inplace=True)),
            ('conv3', nn.Conv3d(cos[1], cos[1], 3, stride=1, padding=1)),
        ]))

        self.merge = nn.Conv3d(2*cos[1], cin, 1, 1)

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        result = torch.cat((branch1, branch2), 1)
        result = self.activa(result)
        result = self.merge(result)
        return x + result


class res_conc_block2down(nn.Module):

    def __init__(self, cin, cn, norm=True, relu=True):
        super(res_conc_block2down, self).__init__()
        self.activa = nn.Sequential()
        if norm: self.activa.add_module('norm', nn.BatchNorm3d(cn))
        if relu: self.activa.add_module('relu', nn.ReLU(True))

        assert (cn % 2 == 0)
        cos = [cn / 2] * 2
        self.activa = nn.Sequential()
        if norm: self.activa.add_module('norm', nn.BatchNorm3d(cn))
        if relu: self.activa.add_module('relu', nn.ReLU(True))

        self.branch1 = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv3d(cin, cos[0], 1)),
            ('norm1', nn.BatchNorm3d(cos[0])),
            ('relu1', nn.ReLU(inplace=True)),
            ('conv2', nn.Conv3d(cos[0], cos[0], 3, stride=2, padding=1)),
        ]))

        self.branch2 = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv3d(cin, cos[1], 1, stride=1)),
            ('norm1', nn.BatchNorm3d(cos[1])),
            ('relu1', nn.ReLU(inplace=True)),
            ('conv2', nn.Conv3d(cos[1], cos[1], 3, stride=1, padding=1)),
            ('norm2', nn.BatchNorm3d(cos[1])),
            ('relu2', nn.ReLU(inplace=True)),
            ('conv3', nn.Conv3d(cos[1], cos[1], 3, stride=2, padding=1)),
        ]))

        self.branch3 = nn.Sequential(OrderedDict([
            ('pool', nn.MaxPool3d(2)),
            ('conv', nn.Conv3d(cin, cn, 1, stride=1))
        ]))


        self.merge = nn.Conv3d(2*cos[1], cn, 1, 1)

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        result = torch.cat((branch1, branch2), 1)
        result = self.activa(result)
        return self.branch3(x) + self.merge(result)



class se_res_conc_block2(nn.Module):
    # '''
    # 残差链接模块
    # 分支1：3*3，stride=1的卷积
    # 分支2:1*1，stride=1的卷积，3*3，stride=1的卷积
    # 分支3：1*1，stride=1的卷积，3*3，stride=1的卷积，3*3，stride=1的卷积
    # 分支1,2,3concat到一起，1*1，stride=1卷积
    # 最后在与input相加
    # '''
    def __init__(self, cin, cn, norm=True, relu=True):
        super(se_res_conc_block2, self).__init__()
        self.activa = nn.Sequential()
        if norm: self.activa.add_module('norm', nn.BatchNorm3d(cn))
        if relu: self.activa.add_module('relu', nn.ReLU(True))

        assert (cn % 2 == 0)
        cos = [cn / 2] * 2   #[32,32]
        self.activa = nn.Sequential()
        if norm: self.activa.add_module('norm', nn.BatchNorm3d(cn))
        if relu: self.activa.add_module('relu', nn.ReLU(True))


        self.branch1 = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv3d(cin, cos[0], 1)),
            ('norm1', nn.BatchNorm3d(cos[0])),
            ('relu1', nn.ReLU(inplace=True)),
            ('conv2', nn.Conv3d(cos[0], cos[0], 3, stride=1, padding=1)),
        ]))

        self.branch2 = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv3d(cin, cos[1], 1, stride=1)),
            ('norm1', nn.BatchNorm3d(cos[1])),
            ('relu1', nn.ReLU(inplace=True)),
            ('conv2', nn.Conv3d(cos[1], cos[1], 3, stride=1, padding=1)),
            ('norm2', nn.BatchNorm3d(cos[1])),
            ('relu2', nn.ReLU(inplace=True)),
            ('conv3', nn.Conv3d(cos[1], cos[1], 3, stride=1, padding=1)),
        ]))

        self.merge = nn.Conv3d(2*cos[1], cin, 1, 1)
        self.se = SELayer3D(cn, 16)

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        result = torch.cat((branch1, branch2), 1)
        result = self.activa(result)
        #result = self.merge(result)
        #################se

        selayer = self.selayer(self.merge(result))

        return x + selayer


class se_res_conc_block2down(nn.Module):

    def __init__(self, cin, cn, norm=True, relu=True):
        super(se_res_conc_block2down, self).__init__()
        self.activa = nn.Sequential()
        if norm: self.activa.add_module('norm', nn.BatchNorm3d(cn))
        if relu: self.activa.add_module('relu', nn.ReLU(True))

        assert (cn % 2 == 0)
        cos = [cn / 2] * 2
        self.activa = nn.Sequential()
        if norm: self.activa.add_module('norm', nn.BatchNorm3d(cn))
        if relu: self.activa.add_module('relu', nn.ReLU(True))

        self.branch1 = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv3d(cin, cos[0], 1)),
            ('norm1', nn.BatchNorm3d(cos[0])),
            ('relu1', nn.ReLU(inplace=True)),
            ('conv2', nn.Conv3d(cos[0], cos[0], 3, stride=2, padding=1)),
        ]))

        self.branch2 = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv3d(cin, cos[1], 1, stride=1)),
            ('norm1', nn.BatchNorm3d(cos[1])),
            ('relu1', nn.ReLU(inplace=True)),
            ('conv2', nn.Conv3d(cos[1], cos[1], 3, stride=1, padding=1)),
            ('norm2', nn.BatchNorm3d(cos[1])),
            ('relu2', nn.ReLU(inplace=True)),
            ('conv3', nn.Conv3d(cos[1], cos[1], 3, stride=2, padding=1)),
        ]))

        self.branch3 = nn.Sequential(OrderedDict([
            ('pool', nn.MaxPool3d(2)),
            ('conv', nn.Conv3d(cin, cn, 1, stride=1))
        ]))


        self.merge = nn.Conv3d(2*cos[1], cn, 1, 1)
        self.se = SELayer3D(cn, 16)

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        result = torch.cat((branch1, branch2), 1)
        result = self.activa(result)
        #################se

        selayer = self.selayer(self.merge(result))



        return self.branch3(x) + selayer

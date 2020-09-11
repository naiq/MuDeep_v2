import torch
import cv2, math, copy
import torch.nn as nn
from torch.nn import init
import torch.utils.model_zoo as model_zoo
import torchvision
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
from collections import OrderedDict

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        init.constant_(m.bias.data, 0.0)


def load_network(net, model_path):
    net.load_state_dict(torch.load(model_path)['state_dict'])
    return net


# ########  MuDeep_v2  ########
# --------------------------------------
class MuDeep_v2(nn.Module):
    def __init__(self, num_class=751, num_scale=3, pretrain=True):
        super(MuDeep_v2, self).__init__()

        net = torchvision.models.resnet50(pretrained=pretrain)
        net.layer4[0].downsample[0].stride = (1,1)
        net.layer4[0].conv2.stride = (1,1)

        self.conv = nn.Sequential(OrderedDict([
            ('conv', net.conv1),
            ('bn', net.bn1),
            ('relu', net.relu),
            ('pool', net.maxpool)
        ]))
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3

        self.layer4_1 = copy.deepcopy(net.layer4[0])
        self.layer4_2 = nn.Sequential(copy.deepcopy(net.layer4[0]), copy.deepcopy(net.layer4[1]))
        self.layer4_3 = nn.Sequential(copy.deepcopy(net.layer4[0]), copy.deepcopy(net.layer4[1]), copy.deepcopy(net.layer4[2]))
        # self.layer4_4 = nn.Sequential(copy.deepcopy(net.layer4[0]), copy.deepcopy(net.layer4[1]), copy.deepcopy(net.layer4[2]), copy.deepcopy(net.layer4[2]))

        self.gap1 = nn.AdaptiveAvgPool2d((1,1))
        self.gap2 = nn.AdaptiveAvgPool2d((3,1))

        guidance = 512
        self.conv_all = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(2048*num_scale, guidance, kernel_size=1, stride=1, bias=False)),
            ('bn1', nn.BatchNorm2d(guidance)),
            ('relu1', nn.ReLU(True)),
        ]))

        bottle_neck = 512
        for i in range(num_scale):
            setattr(self, 'atten'+str(i+1), nn.Sequential(OrderedDict([
                ('feature1', nn.Sequential(OrderedDict([
                    ('conv1', nn.Conv2d(guidance, 2048, kernel_size=1, stride=1, bias=False)),
                    # ('bn1', nn.BatchNorm2d(2048)),
                    # ('relu1', nn.ReLU(True)),
                ]))),
                ('feature2', nn.Sequential(OrderedDict([
                    ('conv1', nn.Conv2d(guidance, 2048, kernel_size=1, stride=1, bias=False)),
                    # ('bn1', nn.BatchNorm2d(2048)),
                    # ('relu1', nn.ReLU(True)),
                ]))),
                ('softmax', nn.Softmax(dim=2)),
            ])))

            setattr(self, 'scale'+str(i+1), nn.Sequential(OrderedDict([
                ('feature', nn.Sequential(OrderedDict([
                    ('conv1', nn.Conv2d(2048, bottle_neck, kernel_size=1, stride=1, bias=False)),
                    ('bn', nn.BatchNorm2d(bottle_neck)),
                    ('relu', nn.ReLU(True)),
                    ])).apply(weights_init_kaiming)),
                ('classifier', nn.Sequential(OrderedDict([
                    ('dropout', nn.Dropout(p=0.5)),
                    ('fc', nn.Linear(bottle_neck, num_class)),
                ])).apply(weights_init_classifier)),
            ])))
            for j in range(3):
                setattr(self, 'scale'+str(i+1)+'_'+str(j+1), nn.Sequential(OrderedDict([
                ('feature', nn.Sequential(OrderedDict([
                    ('conv1', nn.Conv2d(2048, bottle_neck, kernel_size=1, stride=1, bias=False)),
                    ('bn', nn.BatchNorm2d(bottle_neck)),
                    ('relu', nn.ReLU(True)),
                    ])).apply(weights_init_kaiming)),
                ('classifier', nn.Sequential(OrderedDict([
                    ('dropout', nn.Dropout(p=0.5)),
                    ('fc', nn.Linear(bottle_neck, num_class)),
                ])).apply(weights_init_classifier)),
            ])))

        self.gamma1 = nn.Parameter(torch.zeros(1))
        self.gamma2 = nn.Parameter(torch.zeros(1))
        self.gamma3 = nn.Parameter(torch.zeros(1))
        # self.gamma4 = nn.Parameter(torch.zeros(1))

    def forward(self, x, test=False):
        x = self.conv(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        # multi-scale
        # #########################################
        x_1 = self.layer4_1(x)
        x_2 = self.layer4_2(x)
        x_3 = self.layer4_3(x)
        # x_4 = self.layer4_4(x)

        # attention
        # #########################################
        x_all = torch.cat((x_1, x_2, x_3), dim=1)
        x_all = self.conv_all(x_all)

        x1_fea1 = self.atten1.feature1(x_all).view(x_1.size(0),x_1.size(1),-1)       # B x C x (H*W)
        x1_fea2 = self.atten1.feature2(x_all).view(x_1.size(0),x_1.size(1),-1).permute(0,2,1)  # B x C x (H*W) --> B x (H*W) x C
        x1_att_ = torch.bmm(x1_fea1, x1_fea2)    # B x C x C
        x1_att  = self.atten1.softmax(x1_att_)   # B x C x C*
        x1_out = torch.bmm(x1_att, x_1.view(x_1.size(0),x_1.size(1),-1))
        x1_out = x1_out.view(x_1.size(0),x_1.size(1),x_1.size(2),x_1.size(3))
        x_1_new = self.gamma1 * x1_out + x_1
        # x_1_new = torch.sigmoid(self.gamma1) * x1_out + (1 - torch.sigmoid(self.gamma1)) * x_1
        # x_1_new = x1_out

        x2_fea1 = self.atten2.feature1(x_all).view(x_2.size(0),x_2.size(1),-1)   # B x C x (H*W)
        x2_fea2 = self.atten2.feature2(x_all).view(x_2.size(0),x_2.size(1),-1).permute(0,2,1) # B x C x (H*W) --> B x (H*W) x C
        x2_att_ = torch.bmm(x2_fea1, x2_fea2)    # B x C x C
        x2_att  = self.atten2.softmax(x2_att_)   # B x C x C*
        x2_out = torch.bmm(x2_att, x_2.view(x_2.size(0),x_2.size(1),-1))
        x2_out = x2_out.view(x_2.size(0),x_2.size(1),x_2.size(2),x_2.size(3))
        x_2_new = self.gamma2 * x2_out + x_2
        # x_2_new = torch.sigmoid(self.gamma2) * x2_out + (1 - torch.sigmoid(self.gamma2)) * x_2
        # x_2_new = x2_out

        x3_fea1 = self.atten3.feature1(x_all).view(x_3.size(0),x_3.size(1),-1)   # B x C x (H*W)
        x3_fea2 = self.atten3.feature2(x_all).view(x_3.size(0),x_3.size(1),-1).permute(0,2,1) # B x C x (H*W) --> B x (H*W) x C
        x3_att_ = torch.bmm(x3_fea1, x3_fea2)    # B x C x C
        x3_att  = self.atten3.softmax(x3_att_)   # B x C x C*
        x3_out = torch.bmm(x3_att, x_3.view(x_3.size(0),x_3.size(1),-1))
        x3_out = x3_out.view(x_3.size(0),x_3.size(1),x_3.size(2),x_3.size(3))
        x_3_new = self.gamma3 * x3_out + x_3
        # x_3_new = torch.sigmoid(self.gamma3) * x3_out + (1 - torch.sigmoid(self.gamma3)) * x_3
        # x_3_new = x3_out

        # x4_fea1 = self.atten4.feature1(x_all).view(x_4.size(0),x_4.size(1),-1)   # B x C x (H*W)
        # x4_fea2 = self.atten4.feature2(x_all).view(x_4.size(0),x_4.size(1),-1).permute(0,2,1) # B x C x (H*W) --> B x (H*W) x C
        # x4_att_ = torch.bmm(x4_fea1, x4_fea2)    # B x C x C
        # x4_att  = self.atten4.softmax(x4_att_)   # B x C x C*
        # x4_out = torch.bmm(x4_att, x_4.view(x_4.size(0),x_4.size(1),-1))
        # x4_out = x4_out.view(x_4.size(0),x_4.size(1),x_4.size(2),x_4.size(3))
        # x_4 = self.gamma4 * x4_out + x_4


        # global & local global avg pooling
        # #########################################
        fea_1_g = self.gap1(x_1_new)
        fea_1_l = self.gap2(x_1_new)

        fea_2_g = self.gap1(x_2_new)
        fea_2_l = self.gap2(x_2_new)

        fea_3_g = self.gap1(x_3_new)
        fea_3_l = self.gap2(x_3_new)

        # fea_4_g = self.gap1(x_4)
        # fea_4_l = self.gap2(x_4)

        # global and local features
        # #########################################
        fea_1_g_0 = self.scale1.feature(fea_1_g).squeeze()
        fea_1_l_0 = self.scale1_1.feature(fea_1_l[:,:,0,:].unsqueeze(dim=2)).squeeze()
        fea_1_l_1 = self.scale1_2.feature(fea_1_l[:,:,1,:].unsqueeze(dim=2)).squeeze()
        fea_1_l_2 = self.scale1_3.feature(fea_1_l[:,:,2,:].unsqueeze(dim=2)).squeeze()

        fea_2_g_0 = self.scale2.feature(fea_2_g).squeeze()
        fea_2_l_0 = self.scale2_1.feature(fea_2_l[:,:,0,:].unsqueeze(dim=2)).squeeze()
        fea_2_l_1 = self.scale2_2.feature(fea_2_l[:,:,1,:].unsqueeze(dim=2)).squeeze()
        fea_2_l_2 = self.scale2_3.feature(fea_2_l[:,:,2,:].unsqueeze(dim=2)).squeeze()

        fea_3_g_0 = self.scale3.feature(fea_3_g).squeeze()
        fea_3_l_0 = self.scale3_1.feature(fea_3_l[:,:,0,:].unsqueeze(dim=2)).squeeze()
        fea_3_l_1 = self.scale3_2.feature(fea_3_l[:,:,1,:].unsqueeze(dim=2)).squeeze()
        fea_3_l_2 = self.scale3_3.feature(fea_3_l[:,:,2,:].unsqueeze(dim=2)).squeeze()

        # fea_4_g_0 = self.scale4.feature(fea_4_g).squeeze()
        # fea_4_l_0 = self.scale4_1.feature(fea_4_l[:,:,0,:].unsqueeze(dim=2)).squeeze()
        # fea_4_l_1 = self.scale4_2.feature(fea_4_l[:,:,1,:].unsqueeze(dim=2)).squeeze()
        # fea_4_l_2 = self.scale4_3.feature(fea_4_l[:,:,2,:].unsqueeze(dim=2)).squeeze()

        # global and local outputs
        # #########################################
        out_1_g_0 = self.scale1.classifier(fea_1_g_0)
        out_1_l_0 = self.scale1_1.classifier(fea_1_l_0)
        out_1_l_1 = self.scale1_2.classifier(fea_1_l_1)
        out_1_l_2 = self.scale1_3.classifier(fea_1_l_2)

        out_2_g_0 = self.scale2.classifier(fea_2_g_0)
        out_2_l_0 = self.scale2_1.classifier(fea_2_l_0)
        out_2_l_1 = self.scale2_2.classifier(fea_2_l_1)
        out_2_l_2 = self.scale2_3.classifier(fea_2_l_2)

        out_3_g_0 = self.scale3.classifier(fea_3_g_0)
        out_3_l_0 = self.scale3_1.classifier(fea_3_l_0)
        out_3_l_1 = self.scale3_2.classifier(fea_3_l_1)
        out_3_l_2 = self.scale3_3.classifier(fea_3_l_2)

        # out_4_g_0 = self.scale4.classifier(fea_4_g_0)
        # out_4_l_0 = self.scale4_1.classifier(fea_4_l_0)
        # out_4_l_1 = self.scale4_2.classifier(fea_4_l_1)
        # out_4_l_2 = self.scale4_3.classifier(fea_4_l_2)

        # return x_1, x_2, x_3, x_1_new, x_2_new, x_3_new, self.gamma1, self.gamma2, self.gamma3, x1_att, x2_att, x3_att

        if test:
            return fea_1_g_0, fea_1_l_0, fea_1_l_1, fea_1_l_2, \
                   fea_2_g_0, fea_2_l_0, fea_2_l_1, fea_2_l_2, \
                   fea_3_g_0, fea_3_l_0, fea_3_l_1, fea_3_l_2
        else:
            return out_1_g_0, out_1_l_0, out_1_l_1, out_1_l_2, \
                   out_2_g_0, out_2_l_0, out_2_l_1, out_2_l_2, \
                   out_3_g_0, out_3_l_0, out_3_l_1, out_3_l_2, \
                   fea_1_g_0, fea_2_g_0, fea_3_g_0



if __name__ == "__main__":
    net = MuDeep_v2()
    print (net)
    print ('*****************')
    # for param in net.parameters():
    #     print (param)
    # print (net.state_dict().keys())
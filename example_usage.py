# Copyright (c) 2019, Adobe Inc. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike
# 4.0 International Public License. To view a copy of this license, visit
# https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode.

'''
将1000类输出变为9类
'''
import torch

# Run `bash weights/get_antialiased_models.sh` to get weights.

# filter_size = 2;
# filter_size = 3;
filter_size = 5;

# import models_lpf.alexnet
# model = models_lpf.alexnet.AlexNet(filter_size=filter_size)
# state_dict = model.state_dict()
# pre_state_dict = torch.load('weights/alexnet_lpf%i.pth.tar'%filter_size)['state_dict']
# pre_state_dict = {k: v for k, v in pre_state_dict.items() if (k != 'classifier.6.weight' and k != 'classifier.6.bias')}
#
# state_dict.update(pre_state_dict)
# torch.save({'state_dict':state_dict}, 'weights/alexnet_r_%i.pth.tar'%filter_size)
# model.load_state_dict(state_dict)

# import models_lpf.alexnet
# model = models_lpf.alexnet.AlexNet(filter_size=filter_size,use_stn=True)
# state_dict = model.state_dict()
# pre_state_dict = torch.load('weights/alexnet_lpf2_CNN01/model_best.pth.tar')['state_dict']
# pre_state_dict = {k: v for k, v in pre_state_dict.items() if k in state_dict}
#
# state_dict.update(pre_state_dict)
# torch.save({'state_dict':state_dict}, 'weights/alexnet_lpf2_stn.pth.tar')
# model.load_state_dict(state_dict)

# import models_lpf.mymodel
# model = models_lpf.mymodel.alexrotatenet(use_stn=True)
# state_dict = model.state_dict()
# pre_state_dict = torch.load('weights/baseline_CNN01/model_best.pth.tar')['state_dict']
# pre_state_dict = {k: v for k, v in pre_state_dict.items() if k in state_dict}
#
# state_dict.update(pre_state_dict)
# torch.save({'state_dict':state_dict}, 'weights/alexnet_stn_orig.pth.tar')
# model.load_state_dict(state_dict)

import models_lpf.mymodel
# model = models_lpf.mymodel.resrotate50(use_stn=True)
# state_dict = model.state_dict()
# pre_state_dict = torch.load('weights/resnet50_CNN_aug/model_best.pth.tar')['state_dict']
# pre_state_dict = {k: v for k, v in pre_state_dict.items() if k in state_dict}
#
# state_dict.update(pre_state_dict)
# torch.save({'state_dict':state_dict}, 'weights/resnet50_stn.pth.tar')
# model.load_state_dict(state_dict)
#
# model = models_lpf.mymodel.resrotate101(use_stn=True)
# state_dict = model.state_dict()
# pre_state_dict = torch.load('weights/resnet101_CNN_aug/model_best.pth.tar')['state_dict']
# pre_state_dict = {k: v for k, v in pre_state_dict.items() if k in state_dict}
#
# state_dict.update(pre_state_dict)
# torch.save({'state_dict':state_dict}, 'weights/resnet101_stn.pth.tar')
# model.load_state_dict(state_dict)

model = models_lpf.mymodel.vgg16_bn_rotate(use_stn=True)
state_dict = model.state_dict()
pre_state_dict = torch.load('weights/vgg_bn_CNN_aug/model_best.pth.tar')['state_dict']
pre_state_dict = {k: v for k, v in pre_state_dict.items() if k in state_dict}

state_dict.update(pre_state_dict)
torch.save({'state_dict':state_dict}, 'weights/vgg16_bn_stn0.pth.tar')
model.load_state_dict(state_dict)

# import models_lpf.resnet
# model = models_lpf.resnet.resnet101(filter_size=filter_size,use_stn=True)
# state_dict = model.state_dict()
# pre_state_dict = torch.load('weights/resnet101_lpf5_CNN_aug/model_best.pth.tar')['state_dict']
# pre_state_dict = {k: v for k, v in pre_state_dict.items() if k in state_dict}
#
# state_dict.update(pre_state_dict)
# torch.save({'state_dict':state_dict}, 'weights/resnet101_stn0_%i.pth.tar'%filter_size)
# model.load_state_dict(state_dict)
# model = models_lpf.resnet.resnet50(filter_size=filter_size)
# state_dict = model.state_dict()
# pre_state_dict = torch.load('weights/resnet50_lpf%i.pth.tar'%filter_size)['state_dict']
# pre_state_dict = {k: v for k, v in pre_state_dict.items() if (k != 'fc.weight' and k != 'fc.bias')}
#
# state_dict.update(pre_state_dict)
# torch.save({'state_dict':state_dict}, 'weights/resnet50_r_%i.pth.tar'%filter_size)
# model.load_state_dict(state_dict)
#
# model = models_lpf.resnet.resnet101(filter_size=filter_size)
# state_dict = model.state_dict()
# pre_state_dict = torch.load('weights/resnet101_lpf%i.pth.tar'%filter_size)['state_dict']
# pre_state_dict = {k: v for k, v in pre_state_dict.items() if (k != 'fc.weight' and k != 'fc.bias')}
#
# state_dict.update(pre_state_dict)
# torch.save({'state_dict':state_dict}, 'weights/resnet101_r_%i.pth.tar'%filter_size)
# model.load_state_dict(state_dict)
#
# import models_lpf.vgg
# model = models_lpf.vgg.vgg16(filter_size=filter_size)
# state_dict = model.state_dict()
# pre_state_dict = torch.load('weights/vgg16_lpf%i.pth.tar'%filter_size)['state_dict']
# pre_state_dict = {k: v for k, v in pre_state_dict.items() if (k != 'classifier.6.weight' and k != 'classifier.6.bias')}
#
# state_dict.update(pre_state_dict)
# torch.save({'state_dict': state_dict}, 'weights/vgg16_r_%i.pth.tar'%filter_size)
# model.load_state_dict(state_dict)
#
# model = models_lpf.vgg.vgg16_bn(filter_size=filter_size)
# state_dict = model.state_dict()
# pre_state_dict = torch.load('weights/vgg16_bn_lpf%i.pth.tar'%filter_size)['state_dict']
# pre_state_dict = {k: v for k, v in pre_state_dict.items() if (k != 'classifier.6.weight' and k != 'classifier.6.bias')}
#
# state_dict.update(pre_state_dict)
# torch.save({'state_dict': state_dict}, 'weights/vgg16_bn_r_%i.pth.tar'%filter_size)
# model.load_state_dict(state_dict)
#
# import models_lpf.mobilenet
# model = models_lpf.mobilenet.mobilenet_v2(filter_size=filter_size)
# state_dict = model.state_dict()
# pre_state_dict = torch.load('weights/mobilenet_v2_lpf%i.pth.tar'%filter_size)['state_dict']
# pre_state_dict = {k: v for k, v in pre_state_dict.items() if (k != 'classifier.0.weight' and k != 'classifier.0.bias')}
#
# state_dict.update(pre_state_dict)
# torch.save({'state_dict': state_dict}, 'weights/mobilenet_lpf_r_%i.pth.tar'%filter_size)
# model.load_state_dict(state_dict)

import models_lpf.mymodel
# output_num = [4, 2, 1]
# model = models_lpf.mymodel.rotatenet50(filter_size=filter_size,output_num=output_num,pool_type='max_pool')
# state_dict = model.state_dict()
# pre_state_dict = torch.load('weights/resnet50_r_%i.pth.tar'%filter_size)['state_dict']
# pre_state_dict = {k: v for k, v in pre_state_dict.items() if (k != 'fc.weight' and k != 'fc.bias')}
#
# state_dict.update(pre_state_dict)
# torch.save({'state_dict':state_dict}, 'weights/rotatenet50_r_%i.pth.tar'%filter_size)
# model.load_state_dict(state_dict)

# model = models_lpf.mymodel.alexrotatenet(filter_size=filter_size,output_num=output_num,pool_type='max_pool')
# state_dict = model.state_dict()
# pre_state_dict = torch.load('weights/alexnet_r_%i.pth.tar'%filter_size)['state_dict']
# pre_state_dict = {k: v for k, v in pre_state_dict.items() if k in state_dict}
#
# state_dict.update(pre_state_dict)
# torch.save({'state_dict':state_dict}, 'weights/alexrotatenet_r_%i.pth.tar'%filter_size)
# model.load_state_dict(state_dict)




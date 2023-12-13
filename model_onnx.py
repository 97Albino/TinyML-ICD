import torch
import torch.nn as nn
#from operations import *
from torch.autograd import Variable
#from utils import drop_path
from quantizer import *

stride1 = 17
stridemid =2

QUANTIZED = False
class SepConv(nn.Module):

  def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
    super(SepConv, self).__init__()
    if QUANTIZED:
      self.op = nn.Sequential(
        nn.ReLU(inplace=False),
        Conv2dLSQ(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in, bias=False),
        Conv2dLSQ(C_in, C_in, kernel_size=1, padding=0, bias=False),
        nn.BatchNorm2d(C_out, affine=affine), )
    else:
      self.op = nn.Sequential(
        nn.ReLU(inplace=False),
        nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in, bias=False),
        nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
        nn.BatchNorm2d(C_out, affine=affine), )

  def forward(self, x):
    return self.op(x)


class NetworkIEGM(nn.Module):

  def __init__(self):
    super(NetworkIEGM, self).__init__()
    num_classes = 2
    if QUANTIZED:

      C_curr = 5
      C_next = 7#stem_multiplier*C_curr

      self.stem = nn.Sequential(
        Conv2dLSQ(1, C_curr, kernel_size=(1, 17), stride=(1, stride1), padding=(0, 0), bias=False),
        Conv2dLSQ(C_curr, C_next, kernel_size=(1, 1), bias=False),
        nn.BatchNorm2d(C_next)
      )
      C = C_next

      self.layer1 = SepConv(C, C, (1,11),stride=(1,stridemid), padding=(0,5), affine=True)
      self.layer2 = SepConv(C, C, (1,11),stride=(1,stridemid), padding=(0,5), affine=True)
      self.layer3 = SepConv(C, C, (1,11),stride=(1,stridemid), padding=(0,5), affine=True)

      factor = 1
      self.global_pooling = nn.AdaptiveAvgPool2d((factor, 1))


      self.classifier2 = LinearLSQ(C_next * factor, num_classes)



    else:
      #num_classes = 2
      C_curr = 5
      C_next = 7  # stem_multiplier*C_curr

      self.stem = nn.Sequential(
        nn.Conv2d(1, C_curr, kernel_size=(1, 17), stride=(1, stride1), padding=(0, 0), bias=False),
        nn.Conv2d(C_curr,C_next, kernel_size=(1, 1), bias=False),
        nn.BatchNorm2d(C_next)
      )
      C = C_next
      self.layer1 = SepConv(C, C, (1,11),stride=(1,stridemid), padding=(0,5), affine=True)
      self.layer2 = SepConv(C, C, (1,11),stride=(1,stridemid), padding=(0,5), affine=True)
      self.layer3 = SepConv(C, C, (1,11),stride=(1,stridemid), padding=(0,5), affine=True)

      factor = 1
      self.global_pooling = nn.AdaptiveAvgPool2d((factor, 1))

      self.classifier2 = nn.Linear(C_next, num_classes)

  def forward(self, input):

    s0 =  self.stem(input)
    #print(s0.shape)

    s0 = self.layer1(s0)
    #print(s0.shape)
    s0 = self.layer2(s0)
    #print(s0.shape)
    s0 = self.layer3(s0)
    #print(s0.shape)

    out = self.global_pooling(s0)
    #print(s0.shape)
    out = out.view(1,7)

    logits = self.classifier2(out)
    #print(s0.shape)
    return logits


'''
class SepConv(nn.Module):

  def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
    super(SepConv, self).__init__()
    if QUANTIZED:
      self.op = nn.Sequential(
        nn.ReLU(inplace=False),
        Conv2dLSQ(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in, bias=False),
        Conv2dLSQ(C_in, C_in, kernel_size=1, padding=0, bias=False),
        nn.BatchNorm2d(C_out, affine=affine), )
    else:
      self.op = nn.Sequential(
        nn.ReLU(inplace=False),
        nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in, bias=False),
        nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
        nn.BatchNorm2d(C_out, affine=affine), )

  def forward(self, x):
    return self.op(x)


class NetworkIEGM(nn.Module):

  def __init__(self):
    super(NetworkIEGM, self).__init__()
    if QUANTIZED:
      num_classes=2
      C_curr = 5
      C_next = 7#stem_multiplier*C_curr

      self.stem = nn.Sequential(
        Conv2dLSQ(1, C_curr, kernel_size=(17, 1), stride=(17, 1), padding=(0, 0), bias=False),
        Conv2dLSQ(C_curr, C_next, kernel_size=(1, 1), bias=False),
        nn.BatchNorm2d(C_next)
      )
      C = C_next
      self.layer1 = SepConv(C, C, (11,1),stride=(2,1), padding=(5,0), affine=True)
      self.layer2 = SepConv(C, C, (11,1),stride=(2,1), padding=(5,0), affine=True)
      self.layer3 = SepConv(C, C, (11,1),stride=(2,1), padding=(5,0), affine=True)

      factor = 1
      self.global_pooling = nn.AdaptiveAvgPool2d((factor, 1))


      self.classifier2 = LinearLSQ(C_next * factor, num_classes)



    else:
      num_classes = 2
      C_curr = 5
      C_next = 7  # stem_multiplier*C_curr

      self.stem = nn.Sequential(
        nn.Conv2d(1, C_curr, kernel_size=(17, 1), stride=(17, 1), padding=(0, 0),bias=False),
        nn.Conv2d(C_curr,C_next, kernel_size=(1, 1), bias=False),
        nn.BatchNorm2d(C_next)
      )
      C = C_next
      self.layer1 = SepConv(C, C, (11, 1), stride=(2, 1), padding=(5, 0), affine=True)
      self.layer2 = SepConv(C, C, (11, 1), stride=(2, 1), padding=(5, 0), affine=True)
      self.layer3 = SepConv(C, C, (11, 1), stride=(2, 1), padding=(5, 0), affine=True)

      factor = 1
      self.global_pooling = nn.AdaptiveAvgPool2d((factor, 1))

      self.classifier2 = nn.Linear(C_next, num_classes)

  def forward(self, input):

    s0 =  self.stem(input)


    s0 = self.layer1(s0)
    s0 = self.layer2(s0)
    s0 = self.layer3(s0)

    out = self.global_pooling(s0)
    out = out.view(out.size(0),-1)
    logits = self.classifier2(out)
    return logits
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

try:
    from LovaszSoftmax.pytorch.lovasz_losses import lovasz_hinge
except ImportError:
    pass


class BCEDiceEdgeLoss(nn.Module):
    def __init__(self):
        super(BCEDiceEdgeLoss, self).__init__()

        self.para1 = torch.nn.Parameter(torch.tensor(0.5).cuda().requires_grad_())
        self.para2 = torch.nn.Parameter(torch.tensor(0.5).cuda().requires_grad_())
        self.para3 = torch.nn.Parameter(torch.tensor(0.5).cuda().requires_grad_())
        self.count = 0
    def forward(self, input, target):
        input_o = input[:,0:3,:,:,:]
        input_e = input[:, 3:, :, :, :]
        target_o =target[:,0:3,:,:,:]
        target_e = target[:, 3:, :, :, :]
        bce_e = F.binary_cross_entropy_with_logits(input_e, target_e)
        bce = F.binary_cross_entropy_with_logits(input_o, target_o)
        smooth = 1e-5
        input_o = torch.sigmoid(input_o)
        num = target.size(0)
        input_o = input_o.view(num, -1)
        target_o = target_o.view(num, -1)
        intersection = (input_o * target_o)
        dice = (2. * intersection.sum(1) + smooth) / (input_o.sum(1) + target_o.sum(1) + smooth)
        dice = 1 - dice.sum() / num

        # final_loss = 0.7*(dice+bce)+0.3+bce_e

        weighted_bce = torch.mul(bce, self.para1)+(1/self.para1)
        weighted_dice = torch.mul(dice, (self.para2)) + (1 / self.para2 )
        weighted_bce_e = torch.mul(bce_e, (self.para3))+(1/self.para3)

        final_loss = weighted_bce.add(weighted_dice).add(weighted_bce_e)

        self.count +=1
        if self.count%10 ==0:
        # print('p1=%.5f,p2=%.5f,p3=%.5f' % (self.para1, self.para2, self.para3))
            print('p1=%.5f,p2=%.5f,p3=%.5f' % (self.para1, self.para2, self.para3))
            self.count = 0
        return final_loss,self.para1,self.para2,self.para3
        # return final_loss

class BCEDiceLoss(nn.Module):
    def __init__(self):
        super(BCEDiceLoss, self).__init__()

    def forward(self, input, target):
        bce = F.binary_cross_entropy_with_logits(input, target)
        smooth = 1e-5
        input_o = torch.sigmoid(input)
        num = target.size(0)
        input_o = input_o.view(num, -1)
        target_o = target.view(num, -1)
        intersection = (input_o * target_o)
        dice = (2. * intersection.sum(1) + smooth) / (input_o.sum(1) + target_o.sum(1) + smooth)
        dice = 1 - dice.sum() / num
        return 0.5 * bce+0.5 *bce + dice

class LovaszHingeLoss(nn.Module):
    def __init__(self):
        super(LovaszHingeLoss, self).__init__()

    def forward(self, input, target):
        input = input.squeeze(1)
        target = target.squeeze(1)
        loss = lovasz_hinge(input, target, per_image=True)

        return loss

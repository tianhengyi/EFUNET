from torch import nn
from torch import cat
from torch import zeros
class pub(nn.Module):

    def __init__(self, in_channels, out_channels, batch_norm=True):
        super(pub, self).__init__()
        inter_channels = out_channels if in_channels > out_channels else out_channels//2

        layers = [
                    nn.Conv3d(in_channels, inter_channels, 3, stride=1, padding=1),     #3维卷积
                    nn.ReLU(True),
                    nn.Conv3d(inter_channels, out_channels, 3, stride=1, padding=1),
                    nn.ReLU(True)
                 ]
        if batch_norm:
            layers.insert(1, nn.BatchNorm3d(inter_channels))
            layers.insert(len(layers)-1, nn.BatchNorm3d(out_channels))
        self.pub = nn.Sequential(*layers)

    def forward(self, x):
        return self.pub(x)


class unet3dEncoder(nn.Module):

    def __init__(self, in_channels, out_channels, batch_norm=True):
        super(unet3dEncoder, self).__init__()
        self.pub = pub(in_channels, out_channels, batch_norm)      #两次卷积和Relu激活
        self.pool = nn.MaxPool3d(2, stride=2)           #2*2*2 步长为2的池化

    def forward(self, x):
        x = self.pub(x)
        return x,self.pool(x)


class unet3dUp(nn.Module):
    def __init__(self, in_channels, out_channels, batch_norm=True, sample=True):
        super(unet3dUp, self).__init__()
        self.pub = pub(in_channels//2+in_channels, out_channels, batch_norm)
        if sample:
            self.sample = nn.Upsample(scale_factor=2, mode='nearest')
        else:
            self.sample = nn.ConvTranspose3d(in_channels, in_channels, 2, stride=2)

    def forward(self, x, x1):
        x = self.sample(x)
        #c1 = (x1.size(2) - x.size(2)) // 2
        #c2 = (x1.size(3) - x.size(3)) // 2
        #x1 = x1[:, :, c1:-c1, c2:-c2, c2:-c2]
        x = cat((x, x1), dim=1)
        x = self.pub(x)
        return x

class unet3dUp_decoder(nn.Module):
    def __init__(self, in_channels_1, in_channels_2, out_channels, batch_norm=True, sample=True):
        super(unet3dUp_decoder, self).__init__()
        self.pub = pub(in_channels_1+in_channels_2, out_channels, batch_norm)
        if sample:
            self.sample = nn.Upsample(scale_factor=2, mode='nearest')
        else:
            self.sample = nn.ConvTranspose3d(in_channels, in_channels, 2, stride=2)

    def forward(self, x, x1):
        x = self.sample(x)
        #c1 = (x1.size(2) - x.size(2)) // 2
        #c2 = (x1.size(3) - x.size(3)) // 2
        #x1 = x1[:, :, c1:-c1, c2:-c2, c2:-c2]
        x = cat((x, x1), dim=1)
        x = self.pub(x)
        return x

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class cov_edge_guidance_model(nn.Module):
    def __init__(self, in_channels1, in_channels2,out_channels, batch_norm=True):
        super(cov_edge_guidance_model, self).__init__()
        self.pub1 = pub(in_channels1 + in_channels2, in_channels1 + in_channels2, batch_norm)
        self.cov1 = nn.Conv3d(in_channels1, in_channels1, 1)
        self.cov2 = nn.Conv3d(in_channels2, in_channels2, 1)
        self.pub2 = pub(in_channels1 + in_channels2, in_channels1 + in_channels2, batch_norm)
        self.cov3 = nn.Conv3d(in_channels1+in_channels2, out_channels, 1)#向后传的
        self.cov4 = nn.Conv3d(in_channels1 + in_channels2, 3, 1)#深度监督的
        self.sample = nn.Upsample(scale_factor=2, mode='nearest')
        self.se1 = SELayer(in_channels2 + in_channels2)  # 使用通道注意力模块
    def forward(self, x1, x2):
        x1= self.cov1(x1)
        x2 = self.sample(x2)
        x2 = self.cov2(x2)
        x = cat((x1, x2), dim=1)
        x = self.pub1(x)
        x = self.se1(x)  # 应用SELayer模块
        x = self.pub2(x)

        output1 = self.cov3(x)
        output2 = self.cov4(x)
        return output1,output2


class unet3d_conv(nn.Module):
    def __init__(self):# def __init__(self, args):
        super(unet3d_conv, self).__init__()    #使用super调用父类的初始化init
        init_channels = 4                 #输入为4通道，为MRI的四种模态图片
        class_nums = 3
        batch_norm = True
        sample = True

        self.en1 = unet3dEncoder(init_channels, 16, batch_norm)  # 第一次特征提取  输入为4通道 输出为64
        self.en2 = unet3dEncoder(16, 32, batch_norm)  # 池化后 进行第二次特征提取 输入64通道输出为128
        self.en3 = unet3dEncoder(32, 64, batch_norm)  # 池化后 进行第三次特征提取 输入128通道输出为256
        self.en4 = unet3dEncoder(64, 128, batch_norm)  # 池化后 进行第四次特征提取 输入256通道输出为512
        self.edge = cov_edge_guidance_model (16,32 ,8)
        self.up3 = unet3dUp_decoder(136,64, 64, batch_norm, sample)  # 反卷积后 进行第一次尺度还原 输入512通道输出为256
        self.up2 = unet3dUp_decoder(72,32, 32, batch_norm, sample)  # 反卷积后 进行第一次尺度还原 输入256通道输出为128
        self.up1 = unet3dUp_decoder(40,16, 16, batch_norm, sample)  # 反卷积后 进行第一次尺度还原 输入128通道输出为64
        self.con_laststage = pub(24, 16, batch_norm)
        self.con_last = nn.Conv3d(16, class_nums, 1)  # 最后一次 通过卷积输出通道变为3 为分割后的三种病灶区域
        self.pool = nn.MaxPool3d(2, stride=2)  # 2*2*2 步长为2的池化
    def forward(self, x):
        x1,x = self.en1(x)
        x2,x= self.en2(x)
        x3,x= self.en3(x)
        x4,_ = self.en4(x)
        x_edge,x_edge_deepsup = self.edge(x1,x2)
        x_edge_2 = self.pool(x_edge)
        x_edge_1 = self.pool(x_edge_2)
        x = self.up3(x4, x3)
        x = cat((x, x_edge_1), dim=1)
        x = self.up2(x, x2)
        x = cat((x, x_edge_2), dim=1)
        x = self.up1(x, x1)
        x = cat((x, x_edge), dim=1)
        x = self.con_laststage(x)
        out = self.con_last(x)
        out = cat((out, x_edge_deepsup), dim=1)
        return  out

    def _initialize_weights(self):                                #初始化网络权重
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_uniform(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

# 创建一个简单的CNN模型实例
model = unet3d_conv()
input_data = zeros(1, 4,64, 128, 128)  # batch_size=1
# 将空矩阵输入模型，获取输出
output = model(input_data)
# 输出模型的输出维度
print("模型输出维度:", output.size())
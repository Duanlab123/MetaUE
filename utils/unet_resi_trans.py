import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    """Basic Block for resnet 18 and resnet 34
    """

    # BasicBlock and BottleNeck block
    # have different output size
    # we use class attribute expansion
    # to distinct
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        # residual function
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion)
        )

        # shortcut
        self.shortcut = nn.Sequential()

        # the shortcut output dimension is not the same with residual function
        # use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))



class BottleNeck(nn.Module):
    expansion = 4
    def __init__(self, in_channels, middle_channels, stride=1):
        super().__init__()
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, middle_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(middle_channels, middle_channels, stride=stride, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(middle_channels, middle_channels * BottleNeck.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(middle_channels * BottleNeck.expansion),
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != middle_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, middle_channels * BottleNeck.expansion, stride=stride, kernel_size=1,
                          bias=False),
                nn.BatchNorm2d(middle_channels * BottleNeck.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))


class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.first = nn.Sequential(
            nn.Conv2d(in_channels, middle_channels, 3, padding=1),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU()
        )
        self.second = nn.Sequential(
            nn.Conv2d(middle_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        out = self.first(x)
        out = self.second(out)

        return out


class UResnet_trans(nn.Module):
    def __init__(self, block, layers, num_classes, input_channels=3):
        super().__init__()
        nb_filter = [32, 64, 128, 256, 512]

        self.in_channel = nb_filter[0]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = self._make_layer(block, nb_filter[1], layers[0], 1)
        self.conv2_0 = self._make_layer(block, nb_filter[2], layers[1], 1)
        self.conv3_0 = self._make_layer(block, nb_filter[3], layers[2], 1)
        self.conv4_0 = self._make_layer(block, nb_filter[4], layers[3], 1)

        self.conv3_1 = VGGBlock((nb_filter[3] + nb_filter[4]) * block.expansion, nb_filter[3],
                                nb_filter[3] * block.expansion)
        self.conv2_2 = VGGBlock((nb_filter[2] + nb_filter[3]) * block.expansion, nb_filter[2],
                                nb_filter[2] * block.expansion)
        self.conv1_3 = VGGBlock((nb_filter[1] + nb_filter[2]) * block.expansion, nb_filter[1],
                                nb_filter[1] * block.expansion)
        self.conv0_4 = VGGBlock(nb_filter[0] + nb_filter[1] * block.expansion, nb_filter[0], nb_filter[0])

        self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

    def _make_layer(self, block, middle_channel, num_blocks, stride):


        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channel, middle_channel, stride))
            self.in_channel = middle_channel * block.expansion
        return nn.Sequential(*layers)

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))


        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], 1))

        output = self.final(x0_4)
        return output
    def l1_loss(self, gt, gen):
        """
         Absolute Difference loss between gt and gen.

        Args:
            :param gt: The ground truth output tensor, same dimensions as 'gen'.
            :param gen: The predicted outputs.
            :return: Weighted loss float Tensor, it is scalar.
        """
        loss = nn.L1Loss()
        return loss(gt,gen)

    def mse_loss(self, gt, gen):
        """
        l2 loss between gt and gen

        :param gt: The ground truth output tensor, same dimensions as 'gen'.
        :param gen: The predicted outputs.
        :return: L2 loss
        """
        loss = nn.MSELoss()
        return loss(gt,gen)
    def l2_l1_loss(self, gt, gen, alpha=0.8):
        """
        Loss function mix l1_loss and l2_loss

        :param gt: The ground truth output tensor, same dimensions as 'gen'.
        :param gen: The predicted outputs.
        :param alpha: coefficient, default set as 0.8
        :return: Loss
        """
        l1 = self.l1_loss(gt, gen)
        l2 = self.mse_loss(gt, gen)

        return alpha * l2 + (1 - alpha) * l1



    def L_MM(self,I,J_gt, I_bar,I_tilde,J,c1=0.2,c2=0.8,c3=0,c4=0):
        l1 = self.l1_loss(J_gt,J)
        l2 = self.mse_loss(J_gt,J)
        l3 = self.mse_loss(I,I_tilde)
        l4 =  self.mse_loss(I_bar,I_tilde)
        return c1 * l1 + c2 * l2 +c3*l3+c4*l4
    def L_MM_new(self,x_spt,I_bar,y_spt,J, B_spt,A, t_spt,t_bar, c1=1,c2=2,c3=1,c4=1):
        l1 = self.l1_loss(x_spt,I_bar)
        l2 = self.mse_loss(x_spt,I_bar)
        l11=c3*l1+c4*l2
        l3 = self.l1_loss(y_spt,J)
        l4 =  self.mse_loss(y_spt,J)
        l12=c1*l3+c2*l4
        l5 = self.l1_loss(B_spt,A)
        l6 =  self.mse_loss(B_spt,A)
        l13=c1*l5+c2*l6
        l7 = self.l1_loss(t_spt,t_bar)
        l8 =  self.mse_loss(t_spt,t_bar)
        l14=c1*l7+c2*l8
        return l11+l12+l13+l14


    def L_trans(self,t_gt,t_bar,I_bar,I_tilde,c4=0.2,c5=0.8 ,c6=0):
        l1 = self.l1_loss(t_gt,t_bar)
        l2= self.mse_loss(t_gt,t_bar)
        l3 = self.mse_loss(I_bar,I_tilde)

        return c4 * l1  + c5 * l2 +c6*l3



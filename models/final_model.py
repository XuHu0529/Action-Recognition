import torch
import torch.nn as nn
from binarized_modules import  BinarizeLinear,BinarizeConv3d
from vit_pytorch.cct import cct_7


class BARNET(nn.Module):
    """
    模型输入为(batch,channel,time,h,w),(4,3,27,112,112)
    """
    def __init__(self):
        super(BARNET, self).__init__()
        # 输入参数定义
        self.input_shape = [4, 3, 27, 224, 224]
        self.batch_size=4
        self.binconv1=BinarizeConv3d(3, 64, kernel_size=(3, 7, 7), stride=(3, 1, 1),
                     padding=(1, 3, 3), bias=False)
        self.binconv2=BinarizeConv3d(64, 32, kernel_size=(3, 7, 7), stride=(3, 1, 1),
                     padding=(1, 3, 3), bias=False)
        self.binconv3 = BinarizeConv3d(32, 3, kernel_size=(3, 7, 7), stride=(3, 1, 1),
                                       padding=(1, 3, 3), bias=False)

        self.relu = nn.ReLU()

        self.cct = cct_7(
            img_size=224,
            n_conv_layers=1,
            kernel_size=7,
            stride=2,
            padding=3,
            pooling_kernel_size=3,
            pooling_stride=2,
            pooling_padding=1,
            num_classes=101,
            positional_embedding='learnable',  # ['sine', 'learnable', 'none']
        )

    def forward(self, x):
        h = self.relu(self.binconv1(x))
        h = self.relu(self.binconv2(h))
        h = self.relu(self.binconv3(h))
        print(h.size())
        h = torch.reshape(h, (self.batch_size, 3, 224, 224))

        probs = self.cct(h)

        return probs


if __name__ == '__main__':
    model = BARNET()
    input = torch.rand((4, 3, 27, 224, 224), requires_grad=True)
    output = model(input)
    label = torch.rand((4, 101))
    loss_func = nn.CrossEntropyLoss()
    loss = loss_func(output, label)
    loss.backward()


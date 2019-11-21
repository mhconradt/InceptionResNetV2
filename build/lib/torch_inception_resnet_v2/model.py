# declare the main model here
from torch import nn, randn
from torch_inception_resnet_v2.blocks.inception import InceptionA, InceptionB, InceptionC
from torch_inception_resnet_v2.blocks.reduction import ReductionA, ReductionB
from torch_inception_resnet_v2.blocks.stem import Stem
from torch_inception_resnet_v2.utils import Repeated


class InceptionResNetV2(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.stem = Stem()
        self.a_blocks = Repeated(InceptionA, 5)
        self.reduction_a = ReductionA()
        self.b_blocks = Repeated(InceptionB, 10)
        self.reduction_b = ReductionB()
        self.c_blocks = Repeated(InceptionC, 5)
        self.avg_pool = nn.AvgPool2d(kernel_size=8, stride=1, padding=0)
        self.dropout = nn.Dropout(0.8)
        self.fc = nn.Linear(1888, n_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.stem(x)
        x = self.a_blocks(x)
        x = self.reduction_a(x)
        x = self.b_blocks(x)
        x = self.reduction_b(x)
        x = self.c_blocks(x)
        x = self.avg_pool(x)
        x = self.dropout(x)
        x = self.fc(x.squeeze())
        x = self.softmax(x)
        return x


if __name__ == '__main__':
    model = InceptionResNetV2(101)
    X = randn((2, 3, 299, 299))
    print(model(X))

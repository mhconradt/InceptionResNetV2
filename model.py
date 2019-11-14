# declare the main model here
from torch import nn
from blocks.inception import InceptionA, InceptionB, InceptionC
from blocks.reduction import ReductionA, ReductionB
from blocks.stem import Stem
from utils import Repeated


class InceptionResNetV2(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.stem = Stem()
        self.a_blocks = Repeated(InceptionA, 5)
        self.reduction_a = ReductionA()
        self.b_blocks = Repeated(InceptionB, 10)
        self.reduction_b = ReductionB()
        self.c_blocks = Repeated(InceptionC, 5)
        self.avg_pool = nn.AvgPool2d(kernel_size=8, stride=1, padding=1)
        self.dropout = nn.Dropout(0.8)
        self.fc = nn.Linear(1888, n_classes)
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.stem(x)
        x = self.a_blocks(x)
        x = self.reduction_a(x)
        x = self.b_blocks(x)
        x = self.reduction_b(x)
        x = self.c_blocks(x)
        x = self.avg_pool(x)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.softmax(x)
        return x

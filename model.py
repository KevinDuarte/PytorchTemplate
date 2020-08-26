import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, input_dim=(28, 28, 1), n_classes=10):
        super(Model, self).__init__()
        h, w, ch = input_dim
        self.n_classes = n_classes

        self.conv1 = nn.Conv2d(ch, 32, (3, 3), (1, 1), padding=(1, 1))
        self.conv2 = nn.Conv2d(32, 32, (3, 3), (1, 1), padding=(1, 1))
        self.mp1 = nn.MaxPool2d((2, 2), (2, 2))

        self.conv3 = nn.Conv2d(32, 64, (3, 3), (1, 1), padding=(1, 1))
        self.conv4 = nn.Conv2d(64, 64, (3, 3), (1, 1), padding=(1, 1))
        self.mp2 = nn.MaxPool2d((2, 2), (2, 2))

        self.fc = nn.Linear(64*7*7, n_classes)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(-1)

    def forward(self, inputs):

        x = self.relu(self.conv1(inputs))
        x = self.relu(self.conv2(x))
        x = self.mp1(x)

        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.mp2(x)

        x = x.view(-1, 64*7*7)

        x = self.softmax(self.fc(x))

        return x

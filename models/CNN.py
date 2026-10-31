# models/cifar_cnn_gn.py
import torch.nn as nn
import torch.nn.functional as F


class CifarCNN(nn.Module):

    def __init__(self, num_classes=10, dropout_rate=0.5, num_groups=8):
        super(CifarCNN, self).__init__()
        self.num_groups = num_groups

        # Block 1: 32x32 -> 16x16
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.gn1 = nn.GroupNorm(num_groups, 32)  # 替换 BN
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.gn2 = nn.GroupNorm(num_groups, 64)
        self.pool1 = nn.MaxPool2d(2, 2)

        # Block 2: 16x16 -> 8x8
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.gn3 = nn.GroupNorm(num_groups, 128)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.gn4 = nn.GroupNorm(num_groups, 128)
        self.pool2 = nn.MaxPool2d(2, 2)

        # Block 3: 8x8 -> 4x4
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.gn5 = nn.GroupNorm(num_groups, 256)
        self.pool3 = nn.MaxPool2d(2, 2)

        # Classifier
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(dropout_rate / 2)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, x):
        # Block 1
        x = F.relu(self.gn1(self.conv1(x)))
        x = F.relu(self.gn2(self.conv2(x)))
        x = self.pool1(x)

        # Block 2
        x = F.relu(self.gn3(self.conv3(x)))
        x = F.relu(self.gn4(self.conv4(x)))
        x = self.pool2(x)

        # Block 3
        x = F.relu(self.gn5(self.conv5(x)))
        x = self.pool3(x)

        # Classifier
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x
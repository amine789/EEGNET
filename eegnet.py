import torch.nn as nn

class EEGNet(nn.Module):
    def __init__(self, input_shape=(1280, 2), num_classes=3, dropout_rate=0.5, kernel_length=64, F1=8, D=2, F2=16):
        super(EEGNet, self).__init__()

        timesteps, channels = input_shape
        #print('heeeeeeeeeeey ', input_shape, 'F1 ', F1, 'D: ', D, 'F2: ', F2, 'dropout_rate: ', dropout_rate)
        # Temporal convolution
        self.conv1 = nn.Conv2d(1, F1, (1, kernel_length), padding=(0, kernel_length // 2), bias=False)
        self.batchnorm1 = nn.BatchNorm2d(F1)

        # Depthwise convolution
        self.depthwise_conv = nn.Conv2d(F1, F1 * D, (channels, 1), groups=F1, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(F1 * D)
        self.activation1 = nn.ELU()
        self.avgpool1 = nn.AvgPool2d((1, 4))
        self.dropout1 = nn.Dropout(dropout_rate)

        # Separable convolution
        self.conv2 = nn.Conv2d(F1 * D, F2, (1, 16), padding=(0, 8), bias=False)
        self.batchnorm3 = nn.BatchNorm2d(F2)
        self.activation2 = nn.ELU()
        self.avgpool2 = nn.AvgPool2d((1, 8))
        self.dropout2 = nn.Dropout(dropout_rate)

        # Compute final feature size dynamically
        reduced_timepoints = timesteps // (4 * 8)
        self.final_feature_dim = F2 * reduced_timepoints
        self.fc = nn.Linear(self.final_feature_dim, num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1).unsqueeze(1)  # (batch, 1, channels, timesteps)

        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.depthwise_conv(x)
        

        x = self.batchnorm2(x)
        x = self.activation1(x)
        x = self.avgpool1(x)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = self.batchnorm3(x)
        x = self.activation2(x)
        x = self.avgpool2(x)

        x = self.dropout2(x)

        x = x.view(x.size(0), -1)  # Flatten dynamically
        x = self.fc(x)

        return x


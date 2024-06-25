import torch.nn as nn
import torch.nn.functional as F
import torch
#通道注意力机制
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1, stride=1, padding=0)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=1, stride=1, padding=0)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        channel_attention = self.sigmoid(avg_out + max_out)
        return x * channel_attention

#空间注意力机制
class SpatialAttention(nn.Module):
    def __init__(self,kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), "Kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_attention = self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))
        return x * spatial_attention

class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, spatial_kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size=spatial_kernel_size)

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x 
    
#------DenseNet model----
class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers):
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(nn.Sequential(
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, growth_rate, kernel_size=3, padding=1),
            ))
            in_channels += growth_rate

    def forward(self, x):
        features = [x]
        for layer in self.layers:
            new_features = layer(x)
            features.append(new_features)
            x = torch.cat(features, 1)
        return x

class TransitionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TransitionBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        return self.conv(x)

class DenseNet(nn.Module):
    def __init__(self, in_channels=64, growth_rate=32, num_layers=[6, 12, 24, 16]):
        super(DenseNet, self).__init__()

        self.initial_conv = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3)
        self.initial_bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        num_channels = 64
        self.dense_blocks = nn.ModuleList()
        self.transition_blocks = nn.ModuleList()
        for i, num_layers_in_block in enumerate(num_layers):
            dense_block = DenseBlock(num_channels, growth_rate, num_layers_in_block)
            self.dense_blocks.append(dense_block)
            num_channels += growth_rate * num_layers_in_block

            if i != len(num_layers) - 1:
                transition_block = TransitionBlock(num_channels, num_channels // 2)
                self.transition_blocks.append(transition_block)
                num_channels = num_channels // 2

        self.final_bn = nn.BatchNorm2d(512)

    def forward(self, x):
    
        x = self.initial_conv(x)
        x = self.initial_bn(x)
        x = self.relu(x)
        x = self.maxpool(x)
        for dense_block, transition_block in zip(self.dense_blocks, self.transition_blocks):
            x = dense_block(x)
            x = transition_block(x)
        x = self.final_bn(x)
        return x

    
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            SpatialAttention(),
    
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            CBAM(16),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            CBAM(32)
        )

    
    def forward(self, x):    
        x = self.encoder(x)
        return x


class Pose_Net(nn.Module):
    def __init__(self, num_classes=6):
        super(Pose_Net, self).__init__()
        self.num_classes = num_classes
        # Feature CNN
        self.CNN = SimpleCNN()
        self.denseNet = DenseNet()  
        self.regress = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        self.FC = nn.Sequential(
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, num_classes),
            
        )
    def forward(self, DRR,X_ray):
        DRR_feature=self.CNN(DRR) #提取DRR图像特征
        X_feature=self.CNN(X_ray) #提取X-ray图像特征        
        Cat_feature = torch.cat((DRR_feature, X_feature), dim=1)# 拼接
        feature = self.denseNet(Cat_feature) #融合DRR与X-ray特征        
        feature = self.regress(feature) #回归
        x = self.maxpool(feature)
        x = torch.flatten(x, 1) #特征展平
        Param = self.FC(x)   ##回归参数
        Param=torch.tanh(Param) #归一化
        return Param



class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [  nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features),
                        nn.ReLU(inplace=True),
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features)  ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)

class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_blocks=9):
        super(Generator, self).__init__()

        # Initial convolution block       
        model = [   nn.ReflectionPad2d(3),
                    nn.Conv2d(input_nc, 64, 7),
                    nn.InstanceNorm2d(64),
                    nn.ReLU(inplace=True) ]

        # Downsampling
        in_features = 64
        out_features = in_features*2
        for _ in range(2):
            model += [  nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features*2

        # Residual blocks
        for _ in range(n_residual_blocks):
            model += [ResidualBlock(in_features)]

        # Upsampling
        out_features = in_features//2
        for _ in range(2):
            model += [  nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features//2

        # Output layer
        model += [  nn.ReflectionPad2d(3),
                    nn.Conv2d(64, output_nc, 7),
                    nn.Tanh() ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self, input_nc):
        super(Discriminator, self).__init__()

        # A bunch of convolutions one after another
        model = [   nn.Conv2d(input_nc, 64, 4, stride=2, padding=1),
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(64, 128, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(128), 
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(128, 256, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(256), 
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(256, 512, 4, padding=1),
                    nn.InstanceNorm2d(512), 
                    nn.LeakyReLU(0.2, inplace=True) ]

        # FCN classification layer
        model += [nn.Conv2d(512, 1, 4, padding=1)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        x =  self.model(x)
        # Average pooling and flatten
        return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)



import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision import models
import math


class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, expansion=4, stride=1):
        super(Bottleneck, self).__init__()
        mid_channels = out_channels // expansion

        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)

        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)

        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class FC(nn.Module):
    def __init__(self, num_classes=3):
        super(FC, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(64 * 18 * 18, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        x = self.fc(x)
        return x


class FC_RotCenter(nn.Module):
    def __init__(self, num_classes=2):
        super(FC_RotCenter, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(64 * 18 * 18, 512),
            nn.Sigmoid(),
            nn.Linear(512, 128),
            nn.Sigmoid(),
            nn.Linear(128, 32),
            nn.Sigmoid(),
            nn.Linear(32, num_classes),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.fc(x)
        return x


class PosiFusionBlock(nn.Module):
    def __init__(self):
        super(PosiFusionBlock, self).__init__()
        self.regress = nn.Sequential(
            nn.Conv2d(1152, 1024, kernel_size=1, stride=1, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),

            nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),

            nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

        )

    def forward(self, x):
        x = self.regress(x)
        return x


class ViewFusionBlock(nn.Module):
    def __init__(self):
        super(ViewFusionBlock, self).__init__()
        self.fusion = nn.Sequential(

            nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

        )

    def forward(self, x):
        x = self.fusion(x)
        return x


class CrossAttentionFusion(nn.Module):
    def __init__(self, channels):
        super(CrossAttentionFusion, self).__init__()
        self.query_layer1 = nn.Linear(channels, channels)
        self.key_layer2 = nn.Linear(channels, channels)
        self.value_layer2 = nn.Linear(channels, channels)

        self.query_layer2 = nn.Linear(channels, channels)
        self.key_layer1 = nn.Linear(channels, channels)
        self.value_layer1 = nn.Linear(channels, channels)
        self.conv = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

        )

    def forward(self, feature_map1, feature_map2):
        B, C, H, W = feature_map1.size()

        # Flatten feature maps to (B, C, N) where N = H * W
        feature_map1_flat = feature_map1.view(B, C, -1)  # (B, C, N)
        feature_map2_flat = feature_map2.view(B, C, -1)  # (B, C, N)

        # Apply linear transformations
        query1 = self.query_layer1(feature_map1_flat.transpose(1, 2))  # (B, N, C)
        key2 = self.key_layer2(feature_map2_flat.transpose(1, 2))  # (B, N, C)
        value2 = self.value_layer2(feature_map2_flat.transpose(1, 2))  # (B, N, C)

        query2 = self.query_layer2(feature_map2_flat.transpose(1, 2))  # (B, N, C)
        key1 = self.key_layer1(feature_map1_flat.transpose(1, 2))  # (B, N, C)
        value1 = self.value_layer1(feature_map1_flat.transpose(1, 2))  # (B, N, C)

        # Compute cross-attention weights
        attention_weights_1_to_2 = torch.matmul(query1, key2.transpose(-2, -1)) / math.sqrt(C)  # (B, N, N)
        attention_weights_2_to_1 = torch.matmul(query2, key1.transpose(-2, -1)) / math.sqrt(C)  # (B, N, N)

        attention1 = torch.softmax(attention_weights_1_to_2, dim=-1)  # (B, N, N)
        attention2 = torch.softmax(attention_weights_2_to_1, dim=-1)  # (B, N, N)

        # Apply attention to value
        fused_feature1 = torch.matmul(attention1, value2)  # (B, N, C)
        fused_feature2 = torch.matmul(attention2, value1)  # (B, N, C)

        # Combine fused features
        fused_feature = torch.cat([fused_feature1, fused_feature2], dim=-1)  # (B, N, 2C)

        # Reshape to (B, 2C, H, W)
        fused_feature = fused_feature.transpose(1, 2).view(B, 2 * C, H, W)
        fused_feature = self.conv(fused_feature)

        return fused_feature


class RotCenterModel(nn.Module):
    def __init__(self):
        super(RotCenterModel, self).__init__()
        self.bottleneck1 = Bottleneck(512, 256)
        self.bottleneck2 = Bottleneck(256, 128)
        self.bottleneck3 = Bottleneck(128, 64)

    def forward(self, x):
        x = self.bottleneck1(x)
        x = self.bottleneck2(x)
        x = self.bottleneck3(x)

        return x


class Pose_Net(nn.Module):
    def __init__(self):
        super(Pose_Net, self).__init__()
        self.convDRRAp = nn.Conv2d(1, 3, kernel_size=3, padding=1)
        self.convXrayAp = nn.Conv2d(1, 3, kernel_size=3, padding=1)
        self.convDRRLat = nn.Conv2d(1, 3, kernel_size=3, padding=1)
        self.convXrayLat = nn.Conv2d(1, 3, kernel_size=3, padding=1)
        # Feature CNN
        # 加载预训练的 DenseNet-121 模型

        pretrained_densenetAp = models.densenet121(pretrained=True)
        pretrained_densenetLat = models.densenet121(pretrained=True)
        self.CNN_DRR_Ap = nn.Sequential(
            pretrained_densenetAp.features,  # 使用预训练模型的特征部分
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 512, kernel_size=1, padding=1),
        )
        self.CNN_X_Ap = nn.Sequential(
            pretrained_densenetAp.features,
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 512, kernel_size=1, padding=1),
        )
        self.CNN_DRR_Lat = nn.Sequential(
            pretrained_densenetLat.features,
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 512, kernel_size=1, padding=1),
        )
        self.CNN_X_Lat = nn.Sequential(
            pretrained_densenetLat.features,
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 512, kernel_size=1, padding=1),
        )
        self.rotcentermodelDrrAp = RotCenterModel()
        self.rotcentermodelDrrLat = RotCenterModel()
        self.rotcentermodelXrayAp = RotCenterModel()
        self.rotcentermodelXrayLat = RotCenterModel()

        self.FC_rotcenterDrrAp = FC_RotCenter()
        self.FC_rotcenterDrrLat = FC_RotCenter()
        self.FC_rotcenterXrayAp = FC_RotCenter()
        self.FC_rotcenterXrayLat = FC_RotCenter()

        self.posifusionblockAp = PosiFusionBlock()
        self.posifusionblockLat = PosiFusionBlock()
        self.viewfusion = ViewFusionBlock()

        # self.cross_attention = CrossAttentionFusion(channels=256)
        self.FC_R = FC()
        self.FC_T = FC()

    def forward(self, DRR_Ap, X_ray_Ap, DRR_Lat, X_ray_Lat):
        InputDRRAp = self.convDRRAp(DRR_Ap)
        InputXrayAp = self.convXrayAp(X_ray_Ap)
        InputDRRLat = self.convDRRLat(DRR_Lat)
        InputXrayLat = self.convXrayLat(X_ray_Lat)

        DRR_feature_Ap = self.CNN_DRR_Ap(InputDRRAp)
        X_feature_Ap = self.CNN_X_Ap(InputXrayAp)
        DRR_feature_Lat = self.CNN_DRR_Lat(InputDRRLat)
        X_feature_Lat = self.CNN_X_Lat(InputXrayLat)

        RotCenterFeatureDrrAp = self.rotcentermodelDrrAp(DRR_feature_Ap)
        RotCenterFeatureXrayAp = self.rotcentermodelXrayAp(X_feature_Ap)
        RotCenterFeatureDrrLat = self.rotcentermodelDrrLat(DRR_feature_Lat)
        RotCenterFeatureXrayLat = self.rotcentermodelXrayLat(X_feature_Lat)

        RotcenterDrrAp = self.FC_rotcenterDrrAp(torch.flatten(RotCenterFeatureDrrAp, 1))
        RotcenterXrayAp = self.FC_rotcenterXrayAp(torch.flatten(RotCenterFeatureXrayAp, 1))
        RotcenterDrrLat = self.FC_rotcenterDrrLat(torch.flatten(RotCenterFeatureDrrLat, 1))
        RotcenterXrayLat = self.FC_rotcenterXrayLat(torch.flatten(RotCenterFeatureXrayLat, 1))

        CatRotCenterCord = torch.cat((RotcenterDrrAp, RotcenterXrayAp, RotcenterDrrLat, RotcenterXrayLat), dim=1)

        # 拼接
        Cat_feature_Ap = torch.cat((DRR_feature_Ap, X_feature_Ap, RotCenterFeatureDrrAp, RotCenterFeatureXrayAp), dim=1)
        Cat_feature_Lat = torch.cat((DRR_feature_Lat, X_feature_Lat, RotCenterFeatureDrrLat, RotCenterFeatureXrayLat),
                                    dim=1)

        # 拼接两个视图的特征
        featureAp = self.posifusionblockAp(Cat_feature_Ap)
        featureLat = self.posifusionblockLat(Cat_feature_Lat)
        fused_feature = self.viewfusion(torch.cat((featureAp, featureLat), dim=1))

        """ DoubleFeature = torch.cat((featureAp, featureLat), dim=1)
        Feature=self.viewfusion(DoubleFeature)
        """
        x = torch.flatten(fused_feature, 1)
        R = self.FC_R(x)
        T = self.FC_T(x)
        Param = torch.cat((R, T), dim=1)
        return Param, CatRotCenterCord


class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [nn.ReflectionPad2d(1),
                      nn.Conv2d(in_features, in_features, 3),
                      nn.InstanceNorm2d(in_features),
                      nn.ReLU(inplace=True),
                      nn.ReflectionPad2d(1),
                      nn.Conv2d(in_features, in_features, 3),
                      nn.InstanceNorm2d(in_features)]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)


class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_blocks=9):
        super(Generator, self).__init__()

        # Initial convolution block
        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, 64, 7),
                 nn.InstanceNorm2d(64),
                 nn.ReLU(inplace=True)]

        # Downsampling
        in_features = 64
        out_features = in_features * 2
        for _ in range(2):
            model += [nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                      nn.InstanceNorm2d(out_features),
                      nn.ReLU(inplace=True)]
            in_features = out_features
            out_features = in_features * 2

        # Residual blocks
        for _ in range(n_residual_blocks):
            model += [ResidualBlock(in_features)]

        # Upsampling
        out_features = in_features // 2
        for _ in range(2):
            model += [nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                      nn.InstanceNorm2d(out_features),
                      nn.ReLU(inplace=True)]
            in_features = out_features
            out_features = in_features // 2

        # Output layer
        model += [nn.ReflectionPad2d(3),
                  nn.Conv2d(64, output_nc, 7),
                  nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class Discriminator(nn.Module):
    def __init__(self, input_nc):
        super(Discriminator, self).__init__()

        # A bunch of convolutions one after another
        model = [nn.Conv2d(input_nc, 64, 4, stride=2, padding=1),
                 nn.LeakyReLU(0.2, inplace=True)]

        model += [nn.Conv2d(64, 128, 4, stride=2, padding=1),
                  nn.InstanceNorm2d(128),
                  nn.LeakyReLU(0.2, inplace=True)]

        model += [nn.Conv2d(128, 256, 4, stride=2, padding=1),
                  nn.InstanceNorm2d(256),
                  nn.LeakyReLU(0.2, inplace=True)]

        model += [nn.Conv2d(256, 512, 4, padding=1),
                  nn.InstanceNorm2d(512),
                  nn.LeakyReLU(0.2, inplace=True)]

        # FCN classification layer
        model += [nn.Conv2d(512, 1, 4, padding=1)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        x = self.model(x)
        # Average pooling and flatten
        return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)



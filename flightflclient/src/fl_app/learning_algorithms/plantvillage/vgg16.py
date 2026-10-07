import torch
from torch import nn
from torchvision import models  # type: ignore


class ECALayer(nn.Module):
    """
    Constructs a ECA module.

    Args:
        channel: Number of channels of the input feature map
        kernel_size: Adaptive selection of kernel size

    """

    def __init__(self, kernel_size: int = 3) -> None:
        super().__init__()  # type: ignore
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(
            1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False
        )
        self.sigmoid = nn.Sigmoid()
        self.feature_maps: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # feature descriptor on the global spatial information
        y: torch.Tensor = self.avg_pool(x)

        # Two different branches of ECA module
        y_conv_t: torch.Tensor = self.conv(y.squeeze(-1).transpose(-1, -2))
        y_conv = y_conv_t.transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y_sigmoid: torch.Tensor = self.sigmoid(y_conv)

        self.feature_maps = x * y_sigmoid.expand_as(x)

        return self.feature_maps


class PlantVillageVGG16(nn.Module):
    """
    VGG16-based classification model
    Includes ECA (Efficient Channel Attention) modules
    in the convolutional blocks
    to enhance feature extraction.
    The model is pretrained on ImageNet.
    The last fully connected layer is replaced
    with a new one for the specific number of classes.

    Args:
        num_classes (int): Number of classes for the classification task.
        kernel_size (int): Kernel size for the ECA module.
        weights (str): Pretrained weights to use. Default is \
        "VGG16_Weights.IMAGENET1K_V1".

    """

    def __init__(
        self,
        num_classes: int = 10,
        kernel_size: int = 3,
        weights: str = "VGG16_Weights.IMAGENET1K_V1",
    ) -> None:
        super().__init__()  # type: ignore

        # Pretrained model
        vgg = models.vgg16(weights=weights)

        if not isinstance(vgg.features, nn.Sequential):
            raise TypeError(
                "The feature extractor of the VGG model is not a Sequential module."
            )

        # Freeze all layers except the feature extractor
        for param in vgg.features.parameters():
            param.requires_grad = True

        self.features = nn.Sequential(
            # Block 1
            *vgg.features[:5],  # Conv1-Conv2 + ReLU
            ECALayer(kernel_size),
            # Block 2
            *vgg.features[5:10],  # Conv3-Conv4 + ReLU
            ECALayer(kernel_size),
            # Block 3
            *vgg.features[10:17],  # Conv5-Conv7 + ReLU
            ECALayer(kernel_size),
            # Block 4
            *vgg.features[17:24],  # Conv8-Conv10 + ReLU
            ECALayer(kernel_size),
            # Block 5a
            *vgg.features[24:],  # Conv11-Conv13 + ReLU
            ECALayer(kernel_size),
        )

        # Classifier
        self.classifier = vgg.classifier
        self.classifier[-1] = nn.Linear(4096, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.features(x)
        y_f = torch.flatten(y, 1)
        return self.classifier(y_f)

    def get_last_eca_feature_maps(self) -> torch.Tensor | None:
        last_eca_layer = self.features[-1]
        if not isinstance(last_eca_layer, ECALayer):
            raise TypeError(
                "The last layer of the feature extractor is not an ECA layer."
            )
        # Get the last eca_layer feature maps
        return last_eca_layer.feature_maps

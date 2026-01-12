"""
PyTorch implementation of EEGNet for emotion classification.

Adapted from the original Keras/TensorFlow implementation to PyTorch.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EEGNet(nn.Module):
    """
    PyTorch implementation of EEGNet for emotion classification.
    
    Based on: http://iopscience.iop.org/article/10.1088/1741-2552/aace8c/meta
    """
    
    def __init__(self, nb_classes, Chans=64, Samples=128, 
                 dropoutRate=0.5, kernLength=64, F1=8, 
                 D=2, F2=16, norm_rate=0.25, dropoutType='Dropout'):
        """
        Initialize EEGNet model.
        
        Args:
            nb_classes: Number of classes to classify
            Chans: Number of channels/features
            Samples: Number of time samples per window
            dropoutRate: Dropout fraction
            kernLength: Length of temporal convolution in first layer
            F1: Number of temporal filters
            D: Number of spatial filters to learn within each temporal convolution
            F2: Number of pointwise filters (default: F1 * D)
            norm_rate: Max norm constraint rate
            dropoutType: 'Dropout' or 'SpatialDropout2D'
        """
        super(EEGNet, self).__init__()
        
        self.nb_classes = nb_classes
        self.Chans = Chans
        self.Samples = Samples
        
        if F2 is None:
            F2 = F1 * D
        
        # Block 1: Temporal convolution + Depthwise spatial convolution
        self.block1_conv = nn.Conv2d(1, F1, (1, kernLength), padding='same', bias=False)
        self.block1_bn1 = nn.BatchNorm2d(F1)
        self.block1_depthwise = nn.Conv2d(
            F1, F1 * D, (Chans, 1), 
            groups=F1,  # Depthwise convolution
            bias=False
        )
        self.block1_bn2 = nn.BatchNorm2d(F1 * D)
        self.block1_pool = nn.AvgPool2d((1, 4))
        
        # Block 2: Separable convolution
        self.block2_conv = nn.Conv2d(F1 * D, F2, (1, 16), padding='same', bias=False)
        self.block2_bn = nn.BatchNorm2d(F2)
        self.block2_pool = nn.AvgPool2d((1, 8))
        
        # Dropout
        self.use_spatial_dropout = (dropoutType == 'SpatialDropout2D')
        if self.use_spatial_dropout:
            self.dropout1 = nn.Dropout2d(dropoutRate)
            self.dropout2 = nn.Dropout2d(dropoutRate)
        else:
            self.dropout1 = nn.Dropout(dropoutRate)
            self.dropout2 = nn.Dropout(dropoutRate)
        
        self.dropout_rate = dropoutRate
        
        # Calculate flattened size
        # After block 1 pooling: Samples -> Samples // 4
        # After block 2 pooling: Samples // 4 -> (Samples // 4) // 8 = Samples // 32
        # Spatial dimension after depthwise conv: Chans -> 1
        # So final shape: (batch, F2, 1, Samples // 32)
        samples_after_pool = (Samples // 4) // 8
        self.flat_size = F2 * 1 * samples_after_pool
        
        # Classification layer
        self.dense = nn.Linear(self.flat_size, nb_classes)
        
        # Apply max norm constraint (handled during training)
        self.norm_rate = norm_rate
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, Chans, Samples, 1) or (batch, 1, Chans, Samples)
        """
        # Ensure input is (batch, 1, Chans, Samples)
        if x.dim() == 4 and x.shape[-1] == 1:
            # (batch, Chans, Samples, 1) -> (batch, 1, Chans, Samples)
            x = x.permute(0, 3, 1, 2)
        elif x.dim() == 3:
            # (batch, Chans, Samples) -> (batch, 1, Chans, Samples)
            x = x.unsqueeze(1)
        
        # Block 1
        x = self.block1_conv(x)
        x = self.block1_bn1(x)
        x = self.block1_depthwise(x)
        x = self.block1_bn2(x)
        x = F.elu(x)
        x = self.block1_pool(x)
        x = self.dropout1(x)
        
        # Block 2
        x = self.block2_conv(x)
        x = self.block2_bn(x)
        x = F.elu(x)
        x = self.block2_pool(x)
        x = self.dropout2(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Classification
        x = self.dense(x)
        x = F.softmax(x, dim=1)
        
        return x
    
    def extract_features(self, x):
        """
        Extract features before classification layer.
        
        Args:
            x: Input tensor of shape (batch, Chans, Samples, 1) or (batch, 1, Chans, Samples)
        
        Returns:
            Features tensor of shape (batch, flat_size) before classification
        """
        # Ensure input is (batch, 1, Chans, Samples)
        if x.dim() == 4 and x.shape[-1] == 1:
            # (batch, Chans, Samples, 1) -> (batch, 1, Chans, Samples)
            x = x.permute(0, 3, 1, 2)
        elif x.dim() == 3:
            # (batch, Chans, Samples) -> (batch, 1, Chans, Samples)
            x = x.unsqueeze(1)
        
        # Block 1
        x = self.block1_conv(x)
        x = self.block1_bn1(x)
        x = self.block1_depthwise(x)
        x = self.block1_bn2(x)
        x = F.elu(x)
        x = self.block1_pool(x)
        x = self.dropout1(x)
        
        # Block 2
        x = self.block2_conv(x)
        x = self.block2_bn(x)
        x = F.elu(x)
        x = self.block2_pool(x)
        x = self.dropout2(x)
        
        # Flatten (this is the feature vector)
        x = x.view(x.size(0), -1)
        
        return x


def count_parameters(model):
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


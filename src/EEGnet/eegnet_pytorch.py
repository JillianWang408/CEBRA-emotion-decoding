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
    
    Can optionally use self-attention to aggregate window-level predictions into trial-level predictions.
    """
    
    def __init__(self, nb_classes, Chans=64, Samples=128, 
                 dropoutRate=0.5, kernLength=64, F1=8, 
                 D=2, F2=16, norm_rate=0.25, dropoutType='Dropout',
                 use_attention=False, attention_heads=8, attention_dim=None):
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
            use_attention: If True, use self-attention to aggregate windows into trial-level predictions
            attention_heads: Number of attention heads (default: 8)
            attention_dim: Dimension for attention (default: flat_size if use_attention, else None)
        """
        super(EEGNet, self).__init__()
        
        self.nb_classes = nb_classes
        self.Chans = Chans
        self.Samples = Samples
        self.use_attention = use_attention
        
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
        
        # Self-attention mechanism (if enabled)
        if use_attention:
            # Use MultiheadAttention from PyTorch's transformer module
            attn_dim = attention_dim if attention_dim is not None else self.flat_size
            self.attention = nn.MultiheadAttention(
                embed_dim=attn_dim,
                num_heads=attention_heads,
                dropout=dropoutRate,
                batch_first=True  # (batch, seq_len, embed_dim)
            )
            self.attention_norm = nn.LayerNorm(attn_dim)
            self.attention_dropout = nn.Dropout(dropoutRate)
            
            # Projection layer if attention_dim != flat_size
            if attn_dim != self.flat_size:
                self.attention_proj = nn.Linear(self.flat_size, attn_dim)
            else:
                self.attention_proj = nn.Identity()
            
            # Positional encoding for temporal locality
            # We'll use learnable positional embeddings (max_seq_len=1000 should be enough)
            self.max_seq_len = 1000
            self.pos_encoding = nn.Parameter(torch.randn(1, self.max_seq_len, attn_dim) * 0.02)
            
            # Distance-based attention bias (learnable)
            # Creates a bias matrix that favors nearby windows
            self.use_distance_bias = True
            if self.use_distance_bias:
                # Learnable temperature parameter for distance weighting
                self.distance_temp = nn.Parameter(torch.tensor(1.0))
            
            # Final classification layer (operates on aggregated attention output)
            self.dense = nn.Linear(attn_dim, nb_classes)
        else:
            # Standard classification layer
            self.dense = nn.Linear(self.flat_size, nb_classes)
        
        # Apply max norm constraint (handled during training)
        self.norm_rate = norm_rate
    
    def forward(self, x, window_groups=None):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape:
               - Without attention: (batch, Chans, Samples, 1) or (batch, 1, Chans, Samples)
               - With attention: (batch * n_windows_per_trial, Chans, Samples, 1) where windows
                 from the same trial are consecutive in the batch
            window_groups: Optional tensor of shape (batch,) indicating which windows belong
                          to the same trial. Only used when use_attention=True.
                          Example: [0, 0, 0, 1, 1, 1] means first 3 windows are trial 0, next 3 are trial 1.
                          If None and use_attention=True, assumes all windows in batch belong to one trial.
        
        Returns:
            Output tensor:
            - (batch, nb_classes) - one prediction per window/timestep
            - With attention: windows are enhanced with context from other windows in the same trial
            - Without attention: windows are processed independently
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
        
        # Flatten to get window-level features
        x = x.view(x.size(0), -1)  # (batch, flat_size)
        
        if self.use_attention:
            # Apply self-attention to enhance window-level features with context from other windows
            # Project to attention dimension if needed
            x = self.attention_proj(x)  # (batch, attn_dim)
            attn_dim = x.size(1)  # Get attn_dim from the projected tensor shape
            
            # Group windows by trial
            if window_groups is None:
                # Assume all windows belong to one trial
                x_grouped = x.unsqueeze(0)  # (1, batch, attn_dim)
                original_indices = torch.arange(x.size(0), device=x.device)
                trial_lengths = [x.size(0)]
            else:
                # Group windows by trial
                unique_trials = torch.unique(window_groups, sorted=True)
                trial_features = []
                trial_indices = []  # Track original window indices
                trial_lengths = []
                
                # First pass: collect all trial lengths to find max
                for trial_id in unique_trials:
                    trial_mask = (window_groups == trial_id)
                    trial_windows = x[trial_mask]  # (n_windows_in_trial, attn_dim)
                    trial_lengths.append(trial_windows.size(0))
                    trial_indices.append(torch.where(trial_mask)[0])
                
                max_windows = max(trial_lengths) if trial_lengths else x.size(0)
                
                # Second pass: pad all trials to max_windows and concatenate
                for i, trial_id in enumerate(unique_trials):
                    trial_mask = (window_groups == trial_id)
                    trial_windows = x[trial_mask]  # (n_windows_in_trial, attn_dim)
                    seq_len = trial_windows.size(0)
                    
                    # Pad to max_windows if needed
                    if seq_len < max_windows:
                        padding = torch.zeros(max_windows - seq_len, attn_dim, device=x.device)
                        trial_windows = torch.cat([trial_windows, padding], dim=0)  # (max_windows, attn_dim)
                    
                    trial_features.append(trial_windows.unsqueeze(0))  # (1, max_windows, attn_dim)
                
                x_grouped = torch.cat(trial_features, dim=0)  # (n_trials, max_windows, attn_dim)
                original_indices = torch.cat(trial_indices)  # Flattened indices
            
            # Add positional encoding for temporal locality
            n_trials, max_windows_padded, attn_dim = x_grouped.shape
            
            # Get positional encodings for each trial (only for actual windows, not padding)
            pos_encoding_list = []
            for i, seq_len in enumerate(trial_lengths):
                pos_embed = self.pos_encoding[:, :seq_len, :]  # (1, seq_len, attn_dim)
                # Pad to max_windows_padded if needed
                if seq_len < max_windows_padded:
                    padding = torch.zeros(1, max_windows_padded - seq_len, attn_dim, device=x_grouped.device)
                    pos_embed = torch.cat([pos_embed, padding], dim=1)  # (1, max_windows_padded, attn_dim)
                pos_encoding_list.append(pos_embed)
            pos_encoding_grouped = torch.cat(pos_encoding_list, dim=0)  # (n_trials, max_windows_padded, attn_dim)
            
            # Add positional encoding to features
            x_grouped = x_grouped + pos_encoding_grouped
            
            # Create distance-based attention bias to favor nearby windows
            if self.use_distance_bias:
                # Create distance matrix: (max_windows_padded, max_windows_padded)
                positions = torch.arange(max_windows_padded, device=x_grouped.device, dtype=torch.float32)
                distance_matrix = torch.abs(positions.unsqueeze(0) - positions.unsqueeze(1))  # (max_windows_padded, max_windows_padded)
                distance_bias = -distance_matrix / (self.distance_temp + 1e-6)  # (max_windows_padded, max_windows_padded)
                distance_bias = distance_bias.unsqueeze(0).expand(n_trials, -1, -1)  # (n_trials, max_windows_padded, max_windows_padded)
                
                # Create mask to ignore padding positions in attention
                attention_mask = torch.zeros(n_trials, max_windows_padded, max_windows_padded, 
                                            device=x_grouped.device, dtype=torch.bool)
                for i, seq_len in enumerate(trial_lengths):
                    attention_mask[i, :seq_len, :seq_len] = True  # Only attend within real windows
            else:
                distance_bias = None
                attention_mask = None
            
            # Apply self-attention with distance bias
            # MultiheadAttention expects (batch, seq_len, embed_dim) with batch_first=True
            attn_out, attn_weights = self.attention(x_grouped, x_grouped, x_grouped)  # (n_trials, max_windows_padded, attn_dim)
            
            # Apply distance bias to attention weights if enabled
            if self.use_distance_bias and distance_bias is not None:
                # Create distance-weighted mask: nearby windows get higher weight
                distance_weights = torch.exp(distance_bias)  # (n_trials, max_windows_padded, max_windows_padded)
                
                # Mask out padding positions: set weights to 0 for padding
                if attention_mask is not None:
                    distance_weights = distance_weights * attention_mask.float()
                
                # Normalize so each window's attention sums to 1 (only over real windows)
                distance_weights = distance_weights / (distance_weights.sum(dim=-1, keepdim=True) + 1e-6)
                
                # Re-weight the attention output using distance weights
                attn_out_distance = torch.bmm(distance_weights, x_grouped)  # (n_trials, max_windows_padded, attn_dim)
                # Blend: 70% original attention, 30% distance-based
                attn_out = 0.7 * attn_out + 0.3 * attn_out_distance
            
            # Residual connection and layer norm
            x_attended = self.attention_norm(x_grouped + attn_out)  # (n_trials, max_windows_padded, attn_dim)
            x_attended = self.attention_dropout(x_attended)
            
            # Extract only real windows (remove padding) and flatten
            # Maintain per-window predictions, not aggregated to trial level
            real_windows = []
            for i, seq_len in enumerate(trial_lengths):
                real_windows.append(x_attended[i, :seq_len, :])  # (seq_len, attn_dim)
            x = torch.cat(real_windows, dim=0)  # (total_real_windows, attn_dim)
            
            # Reorder to match original batch order if needed
            if window_groups is not None:
                # Sort back to original order
                _, sort_idx = torch.sort(original_indices)
                x = x[sort_idx]  # (batch, attn_dim)
        
        # Classification
        x = self.dense(x)
        x = F.softmax(x, dim=1)
        
        return x
    
    def extract_features(self, x, window_groups=None):
        """
        Extract features before classification layer.
        
        Args:
            x: Input tensor of shape (batch, Chans, Samples, 1) or (batch, 1, Chans, Samples)
            window_groups: Optional tensor for grouping windows (only used if use_attention=True)
        
        Returns:
            Features tensor:
            - (batch, attn_dim or flat_size) - one feature vector per window/timestep
            - With attention: windows are enhanced with context from other windows in the same trial
            - Without attention: windows are processed independently
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
        x = x.view(x.size(0), -1)  # (batch, flat_size)
        
        if self.use_attention:
            # Project to attention dimension if needed
            x = self.attention_proj(x)  # (batch, attn_dim)
            attn_dim = x.size(1)  # Get attn_dim from the projected tensor shape
            
            # Group windows by trial
            if window_groups is None:
                # Assume all windows belong to one trial
                x_grouped = x.unsqueeze(0)  # (1, batch, attn_dim)
                original_indices = torch.arange(x.size(0), device=x.device)
                trial_lengths = [x.size(0)]
            else:
                # Group windows by trial
                unique_trials = torch.unique(window_groups, sorted=True)
                trial_features = []
                trial_indices = []  # Track original window indices
                trial_lengths = []
                
                # First pass: collect all trial lengths to find max
                for trial_id in unique_trials:
                    trial_mask = (window_groups == trial_id)
                    trial_windows = x[trial_mask]  # (n_windows_in_trial, attn_dim)
                    trial_lengths.append(trial_windows.size(0))
                    trial_indices.append(torch.where(trial_mask)[0])
                
                max_windows = max(trial_lengths) if trial_lengths else x.size(0)
                
                # Second pass: pad all trials to max_windows and concatenate
                for i, trial_id in enumerate(unique_trials):
                    trial_mask = (window_groups == trial_id)
                    trial_windows = x[trial_mask]  # (n_windows_in_trial, attn_dim)
                    seq_len = trial_windows.size(0)
                    
                    # Pad to max_windows if needed
                    if seq_len < max_windows:
                        padding = torch.zeros(max_windows - seq_len, attn_dim, device=x.device)
                        trial_windows = torch.cat([trial_windows, padding], dim=0)  # (max_windows, attn_dim)
                    
                    trial_features.append(trial_windows.unsqueeze(0))  # (1, max_windows, attn_dim)
                
                x_grouped = torch.cat(trial_features, dim=0)  # (n_trials, max_windows, attn_dim)
                original_indices = torch.cat(trial_indices)  # Flattened indices
            
            # Add positional encoding for temporal locality
            n_trials, max_windows_padded, attn_dim = x_grouped.shape
            
            # Get positional encodings for each trial (only for actual windows, not padding)
            pos_encoding_list = []
            for i, seq_len in enumerate(trial_lengths):
                pos_embed = self.pos_encoding[:, :seq_len, :]  # (1, seq_len, attn_dim)
                # Pad to max_windows_padded if needed
                if seq_len < max_windows_padded:
                    padding = torch.zeros(1, max_windows_padded - seq_len, attn_dim, device=x_grouped.device)
                    pos_embed = torch.cat([pos_embed, padding], dim=1)  # (1, max_windows_padded, attn_dim)
                pos_encoding_list.append(pos_embed)
            pos_encoding_grouped = torch.cat(pos_encoding_list, dim=0)  # (n_trials, max_windows_padded, attn_dim)
            
            # Add positional encoding to features
            x_grouped = x_grouped + pos_encoding_grouped
            
            # Create distance-based attention bias to favor nearby windows
            if self.use_distance_bias:
                # Create distance matrix: (max_windows_padded, max_windows_padded)
                positions = torch.arange(max_windows_padded, device=x_grouped.device, dtype=torch.float32)
                distance_matrix = torch.abs(positions.unsqueeze(0) - positions.unsqueeze(1))  # (max_windows_padded, max_windows_padded)
                distance_bias = -distance_matrix / (self.distance_temp + 1e-6)  # (max_windows_padded, max_windows_padded)
                distance_bias = distance_bias.unsqueeze(0).expand(n_trials, -1, -1)  # (n_trials, max_windows_padded, max_windows_padded)
                
                # Create mask to ignore padding positions in attention
                attention_mask = torch.zeros(n_trials, max_windows_padded, max_windows_padded, 
                                            device=x_grouped.device, dtype=torch.bool)
                for i, seq_len in enumerate(trial_lengths):
                    attention_mask[i, :seq_len, :seq_len] = True  # Only attend within real windows
            else:
                distance_bias = None
                attention_mask = None
            
            # Apply self-attention
            attn_out, _ = self.attention(x_grouped, x_grouped, x_grouped)  # (n_trials, max_windows_padded, attn_dim)
            
            # Apply distance weighting if enabled
            if self.use_distance_bias and distance_bias is not None:
                distance_weights = torch.exp(distance_bias)  # (n_trials, max_windows_padded, max_windows_padded)
                
                # Mask out padding positions: set weights to 0 for padding
                if attention_mask is not None:
                    distance_weights = distance_weights * attention_mask.float()
                
                distance_weights = distance_weights / (distance_weights.sum(dim=-1, keepdim=True) + 1e-6)
                attn_out_distance = torch.bmm(distance_weights, x_grouped)  # (n_trials, max_windows_padded, attn_dim)
                attn_out = 0.7 * attn_out + 0.3 * attn_out_distance
            
            # Residual connection and layer norm
            x_attended = self.attention_norm(x_grouped + attn_out)  # (n_trials, max_windows_padded, attn_dim)
            
            # Extract only real windows (remove padding) and flatten
            # Maintain per-window features, not aggregated to trial level
            real_windows = []
            for i, seq_len in enumerate(trial_lengths):
                real_windows.append(x_attended[i, :seq_len, :])  # (seq_len, attn_dim)
            x = torch.cat(real_windows, dim=0)  # (total_real_windows, attn_dim)
            
            # Reorder to match original batch order if needed
            if window_groups is not None:
                # Sort back to original order
                _, sort_idx = torch.sort(original_indices)
                x = x[sort_idx]  # (batch, attn_dim)
        
        return x


def count_parameters(model):
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


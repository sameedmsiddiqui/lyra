import numpy as np
import pandas as pd
import json
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from flashfftconv import FlashDepthWiseConv1d
from einops import rearrange
#from . import s4d 
from .s4d import S4D
from ..utils.rmsnorm import RMSNorm 

dropout_fn = nn.Dropout1d
#torch.backends.cudnn.benchmark = True
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Example configuration for Lyra, as used in our protein datasets
# usage: 
#       lyra_instance = Lyra(**lyra_example_proteins_config)
lyra_example_proteins_config = {
    "d_input": 20,
    "d_output": 1,
    "d_model": 64,
    "dropout": 0.2,
}


class PGC(nn.Module):
    """
    Parallel Gated Convolution module with FFT-based convolution.
    This module projects the input, applies a gated FFT convolution,
    and projects back to the original dimension.
    """
    def __init__(self, d_model, expansion_factor=1.0, dropout=0.0):
        super().__init__()
        self.d_model = d_model
        self.expansion_factor = expansion_factor
        self.dropout = dropout
        
        # Calculate expanded dimension
        expanded_dim = int(d_model * expansion_factor)
        
        # Input projection and normalization
        self.in_proj = nn.Linear(d_model, expanded_dim * 2)
        self.in_norm = RMSNorm(expanded_dim * 2, eps=1e-8)
        
        # Regular convolution for initialization
        self.conv = nn.Conv1d(expanded_dim, expanded_dim, kernel_size=3, padding=1, groups=expanded_dim)
        
        # Flash FFT Conv layer, passing weights and bias from self.conv
        self.flash_conv = FlashDepthWiseConv1d(expanded_dim, kernel_size=3, padding=1,
                                              weights=self.conv.weight, bias=self.conv.bias)
        
        # Output projection and normalization
        self.out_proj = nn.Linear(expanded_dim, d_model)
        self.out_norm = RMSNorm(d_model, eps=1e-8)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, u):
        # Input projection and normalization
        xv = self.in_norm(self.in_proj(u))
        
        # Split into x and v for gating
        x, v = xv.chunk(2, dim=-1)
        
        # Apply Flash FFT Convolution
        x_feature_mixed = self.flash_conv(rearrange(x, 'b t f -> b f t').contiguous())
        x_feature_mixed = rearrange(x_feature_mixed, 'b f t -> b t f')
        
        # Apply gating with v
        gated_output = v * x_feature_mixed
        
        # Output projection and normalization
        out = self.out_norm(self.out_proj(gated_output))
        
        return out


class Lyra(nn.Module):
    """
    Lyra neural network architecture combining PGC modules with S4D for sequence modeling.
    
    Architecture:
    1. Input encoding
    2. Two PGC modules with different expansion factors
    3. S4D layer with residual connection
    4. Global pooling and decoding
    
    Parameters:
    -----------
    d_input : int
        Dimension of input features
    d_output : int
        Dimension of output features
    d_model : int
        Internal model dimension
    d_state : int, default=64
        State dimension for S4D layer
    dropout : float, default=0.2
        Dropout rate
    transposed : bool, default=False
        Whether the input is transposed for S4D
    **kernel_args : 
        Additional arguments for S4D kernel
    """
    def __init__(self, d_input, d_output, d_model, d_state=64, dropout=0.2, transposed=False, **kernel_args):
        super().__init__()
        
        # Input encoding
        self.encoder = nn.Linear(d_input, d_model)
        
        # PGC modules with different expansion factors
        # First PGC compresses (0.25x), second PGC expands (2x)
        self.pgc1 = PGC(d_model, expansion_factor=0.25, dropout=dropout)
        self.pgc2 = PGC(d_model, expansion_factor=2, dropout=dropout)
        
        # S4D layer for sequence modeling
        self.s4d = S4D(d_model, d_state=d_state, dropout=dropout, transposed=transposed, **kernel_args)
        
        # Normalization and output layers
        self.norm = RMSNorm(d_model)
        self.decoder = nn.Linear(d_model, d_output)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, u):
        """
        Forward pass through the Lyra network.
        
        Parameters:
        -----------
        u : torch.Tensor
            Input tensor of shape [batch_size, sequence_length, d_input]
            
        Returns:
        --------
        x : torch.Tensor
            Output tensor of shape [batch_size, d_output]
        """
        # Encode input to model dimension
        x = self.encoder(u)
        
        # Apply PGC modules sequentially
        x = self.pgc1(x)  # Compress features
        x = self.pgc2(x)  # Expand features
        
        # Normalize for S4D
        z = self.norm(x)
        
        # Apply S4D with residual connection
        x = self.dropout(self.s4d(z)) + x
        
        # Global pooling over sequence dimension
        x = x.mean(dim=1)
        
        # Decode to output dimension
        x = self.decoder(x)
        
        return x


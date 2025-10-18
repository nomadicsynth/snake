import torch
import torch.nn as nn
import math

class PositionalEncoding2D(nn.Module):
    """
    2D sinusoidal positional encoding for grid-based inputs.
    Requires d_model % 4 == 0.
    """
    def __init__(self, d_model, height, width):
        super().__init__()
        assert d_model % 4 == 0, "d_model must be divisible by 4 for 2D positional encoding"
        
        self.height = height
        self.width = width
        self.d_model = d_model
        
        # Create row encodings (height, d_model//2)
        pe_row = torch.zeros(height, d_model // 2)
        position_row = torch.arange(0, height, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model // 2, 2, dtype=torch.float32) * (-math.log(10000.0) / (d_model // 2)))
        pe_row[:, 0::2] = torch.sin(position_row * div_term)
        pe_row[:, 1::2] = torch.cos(position_row * div_term)
        
        # Create column encodings (width, d_model//2)
        pe_col = torch.zeros(width, d_model // 2)
        position_col = torch.arange(0, width, dtype=torch.float32).unsqueeze(1)
        pe_col[:, 0::2] = torch.sin(position_col * div_term)
        pe_col[:, 1::2] = torch.cos(position_col * div_term)
        
        # Combine into 2D positional encoding (height, width, d_model)
        pe_2d = torch.zeros(height, width, d_model)
        for y in range(height):
            for x in range(width):
                pe_2d[y, x] = torch.cat([pe_row[y], pe_col[x]], dim=0)
        
        self.register_buffer("pe", pe_2d)

    def forward(self, x):
        # x: (B, S, d_model) where S = height * width
        b, s, d = x.shape
        assert s == self.height * self.width, f"Sequence length {s} does not match grid size {self.height}x{self.width}"
        assert d == self.d_model, f"Model dimension {d} does not match {self.d_model}"
        
        # Flatten 2D PE to match sequence
        pe_flat = self.pe.view(s, d).unsqueeze(0)  # (1, S, d_model)
        return x + pe_flat
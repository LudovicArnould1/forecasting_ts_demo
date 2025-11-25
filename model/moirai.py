"""Fixed Minimal MOIRAI implementation for time series forecasting.

This implementation follows the Moirai paper with:
- RoPE (Rotary Position Embeddings) applied to Q and K
- Any-variate attention with binary attention biases
- Proper handling of multivariate time series
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from pathlib import Path

# Add RevIN to path
sys.path.insert(0, str(Path(__file__).parent.parent / "RevIN"))
from RevIN import RevIN


class PatchEmbedding(nn.Module):
    """Embed time series patches into model dimension with adaptive patch size handling.
    
    This module can handle variable patch sizes by maintaining separate projection
    layers for each encountered patch size. This enables training on mixed-frequency
    time series data.
    
    Args:
        patch_sizes: List of patch sizes to support (e.g., [8, 16, 32, 64, 128, 256])
        d_model: Model dimension
        
    Examples:
        >>> embedding = PatchEmbedding([128, 256], d_model=128)
        >>> x1 = torch.randn(32, 64, 128)  # batch=32, patches=64, patch_size=128
        >>> x2 = torch.randn(32, 64, 256)  # batch=32, patches=64, patch_size=256
        >>> out1 = embedding(x1)  # Shape: (32, 64, 128)
        >>> out2 = embedding(x2)  # Shape: (32, 64, 128)
    """
    
    def __init__(self, patch_sizes: list[int], d_model: int):
        super().__init__()
        self.patch_sizes = sorted(patch_sizes)
        self.d_model = d_model
        
        # Create a separate projection layer for each patch size
        self.projections = nn.ModuleDict({
            str(ps): nn.Linear(ps, d_model) for ps in patch_sizes
        })
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project patches to embedding dimension.
        
        Automatically selects the appropriate projection based on input patch size.
        
        Args:
            x: Shape (batch, num_patches, patch_size)
            
        Returns:
            Shape (batch, num_patches, d_model)
            
        Raises:
            ValueError: If patch_size is not supported
        """
        patch_size = x.shape[-1]
        
        if str(patch_size) not in self.projections:
            raise ValueError(
                f"Unsupported patch_size={patch_size}. "
                f"Supported sizes: {self.patch_sizes}"
            )
        
        return self.projections[str(patch_size)](x)


def apply_rotary_pos_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary position embeddings to input tensor.
    
    RoPE rotates pairs of dimensions in the embedding space.
    
    Args:
        x: Input tensor of shape (..., seq_len, head_dim)
        cos: Cosine values of shape (seq_len, head_dim)  
        sin: Sine values of shape (seq_len, head_dim)
        
    Returns:
        Tensor with rotary embeddings applied
    """
    # Split into two halves for rotation
    # x1 contains dimensions [0, 2, 4, ...], x2 contains [1, 3, 5, ...]
    x1 = x[..., ::2]  # Even indices
    x2 = x[..., 1::2]  # Odd indices
    
    # cos and sin are duplicated, so we only need half
    cos_half = cos[..., ::2]
    sin_half = sin[..., ::2]
    
    # Apply rotation: 
    # [x1*cos - x2*sin, x1*sin + x2*cos]
    rotated_x1 = x1 * cos_half - x2 * sin_half
    rotated_x2 = x1 * sin_half + x2 * cos_half
    
    # Interleave back
    rotated = torch.stack([rotated_x1, rotated_x2], dim=-1).flatten(-2)
    
    return rotated


class RotaryPositionEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) as described in the Moirai paper.
    
    RoPE encodes relative position information by rotating query and key vectors
    in the complex plane.
    """
    
    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        # Precompute frequency values
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        
        # Precompute cos and sin for efficiency
        self._precompute_freqs(max_seq_len)
    
    def _precompute_freqs(self, seq_len: int):
        """Precompute cosine and sine values."""
        t = torch.arange(seq_len, device=self.inv_freq.device).float()
        freqs = torch.outer(t, self.inv_freq)  # (seq_len, dim//2)
        
        # Duplicate frequencies to match head_dim
        emb = torch.cat([freqs, freqs], dim=-1)  # (seq_len, dim)
        
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)
        self.cached_seq_len = seq_len
    
    def forward(self, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get cosine and sine values for the given sequence length.
        
        Args:
            seq_len: Length of the sequence
            
        Returns:
            Tuple of (cos, sin) tensors of shape (seq_len, dim)
        """
        if seq_len > self.cached_seq_len:
            self._precompute_freqs(seq_len)
        
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


class AnyVariateAttention(nn.Module):
    """Any-variate attention with binary attention biases as described in Moirai.
    
    This attention mechanism handles arbitrary numbers of variates by:
    1. Flattening multivariate sequences into a single sequence
    2. Using RoPE for temporal position encoding
    3. Using binary attention biases to distinguish between variates
    
    Equation (2) from paper:
    Aij,mn = (W^Q xi,m)^T Ri-j (W^K xj,n) + u(1) * 1{m=n} + u(2) * 1{m≠n}
    
    where i,j are time indices, m,n are variate indices
    """
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        # Q, K, V projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        # Binary attention biases for any-variate attention
        # u(1) for same-variate (m=n), u(2) for cross-variate (m≠n)
        self.u_same = nn.Parameter(torch.zeros(num_heads))
        self.u_cross = nn.Parameter(torch.zeros(num_heads))
        
        # RoPE
        self.rope = RotaryPositionEmbedding(self.head_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
    def forward(
        self,
        x: torch.Tensor,
        variate_ids: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass with any-variate attention.
        
        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
               where seq_len = num_variates * num_patches
            variate_ids: Variate indices of shape (batch, seq_len)
                        e.g., [0,0,0,1,1,1,2,2,2] for 3 variates with 3 patches each
            mask: Optional attention mask (batch, seq_len)
            
        Returns:
            Output tensor of shape (batch, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape
        
        # Project to Q, K, V
        q = self.q_proj(x)  # (batch, seq_len, d_model)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape for multi-head attention: (batch, num_heads, seq_len, head_dim)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Apply RoPE to Q and K
        cos, sin = self.rope(seq_len)
        cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, head_dim)
        sin = sin.unsqueeze(0).unsqueeze(0)
        
        q = apply_rotary_pos_emb(q, cos, sin)
        k = apply_rotary_pos_emb(k, cos, sin)
        
        # Compute attention scores: Q @ K^T
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        # Shape: (batch, num_heads, seq_len, seq_len)
        
        # Add binary attention biases based on variate indices
        # Create binary bias matrix: u(1) if m=n, u(2) if m≠n
        variate_ids_expanded = variate_ids.unsqueeze(-1)  # (batch, seq_len, 1)
        variate_ids_transposed = variate_ids.unsqueeze(-2)  # (batch, 1, seq_len)
        
        # Binary mask: 1 where same variate, 0 where different
        same_variate_mask = (variate_ids_expanded == variate_ids_transposed).float()
        # Shape: (batch, seq_len, seq_len)
        
        # Compute binary bias for each head
        # u_same for same variate, u_cross for different variates
        binary_bias = (
            same_variate_mask.unsqueeze(1) * self.u_same.view(1, -1, 1, 1) +
            (1 - same_variate_mask).unsqueeze(1) * self.u_cross.view(1, -1, 1, 1)
        )
        # Shape: (batch, num_heads, seq_len, seq_len)
        
        attn_scores = attn_scores + binary_bias
        
        # Apply padding mask if provided
        if mask is not None:
            # mask shape: (batch, seq_len) - 0 for padding, 1 for valid
            # Convert to attention mask: -inf for padding positions
            attn_mask = (1.0 - mask.unsqueeze(1).unsqueeze(2)) * -1e9
            attn_scores = attn_scores + attn_mask
        
        # Softmax and dropout
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        
        # Apply attention to values
        output = torch.matmul(attn_probs, v)
        # Shape: (batch, num_heads, seq_len, head_dim)
        
        # Reshape back
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        # Final projection
        output = self.out_proj(output)
        
        return output


class TransformerBlock(nn.Module):
    """Transformer block with any-variate attention and FFN."""
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        
        # Any-variate attention
        self.attention = AnyVariateAttention(d_model, num_heads, dropout)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        
        # Layer norms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        x: torch.Tensor,
        variate_ids: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass with residual connections.
        
        Args:
            x: Shape (batch, seq_len, d_model)
            variate_ids: Variate indices (batch, seq_len)
            mask: Optional attention mask
            
        Returns:
            Shape (batch, seq_len, d_model)
        """
        # Self-attention with residual
        attn_out = self.attention(x, variate_ids, mask)
        x = self.norm1(x + self.dropout(attn_out))
        
        # FFN with residual
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        
        return x


class ForecastingHead(nn.Module):
    """Prediction head that outputs direct value predictions for each patch.
    
    Handles variable patch sizes by maintaining separate projection layers.
    
    Args:
        d_model: Model dimension
        patch_sizes: List of patch sizes to support
    """
    
    def __init__(self, d_model: int, patch_sizes: list[int]):
        super().__init__()
        self.patch_sizes = sorted(patch_sizes)
        self.d_model = d_model
        
        # Create separate projections for each patch size
        self.projections = nn.ModuleDict({
            str(ps): nn.Linear(d_model, ps) for ps in patch_sizes
        })
        
    def forward(self, x: torch.Tensor, patch_size: int) -> torch.Tensor:
        """Predict values directly.
        
        Args:
            x: Shape (batch, num_patches, d_model)
            patch_size: Size of patches to predict
            
        Returns:
            predictions: Shape (batch, num_patches, patch_size)
            
        Raises:
            ValueError: If patch_size is not supported
        """
        if str(patch_size) not in self.projections:
            raise ValueError(
                f"Unsupported patch_size={patch_size}. "
                f"Supported sizes: {self.patch_sizes}"
            )
        
        return self.projections[str(patch_size)](x)


class MinimalMOIRAI(nn.Module):
    """MOIRAI-style forecasting model with any-variate attention.
    
    This implementation follows the Moirai paper:
    - Flattens multivariate time series into a single sequence
    - Uses RoPE for temporal position encoding
    - Uses binary attention biases to distinguish between variates
    - Supports multiple patch sizes for mixed-frequency training
    
    Args:
        patch_sizes: List of patch sizes to support (e.g., [128, 256])
                    If single int provided, will be converted to list
        d_model: Model dimension
        num_heads: Number of attention heads
        num_layers: Number of transformer blocks
        d_ff: Feed-forward dimension
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        patch_sizes: int | list[int],
        d_model: int,
        num_heads: int,
        num_layers: int = 2,
        d_ff: int | None = None,
        dropout: float = 0.1,
        revin_affine: bool = True,
    ):
        super().__init__()
        
        if d_ff is None:
            d_ff = 4 * d_model
        
        # Handle both single patch_size (backward compat) and list
        if isinstance(patch_sizes, int):
            patch_sizes = [patch_sizes]
        
        self.patch_sizes = sorted(patch_sizes)
        self.d_model = d_model
        
        # RevIN normalization (applied per variate, one instance per patch size)
        # RevIN expects input shape: (..., seq_len, num_features)
        # where num_features is the patch_size in our case
        self.revin_layers = nn.ModuleDict({
            str(ps): RevIN(num_features=ps, eps=1e-5, affine=revin_affine)
            for ps in patch_sizes
        })
        
        # Adaptive patch embedding (handles multiple patch sizes)
        self.embedding = PatchEmbedding(patch_sizes, d_model)
        
        # Transformer blocks with any-variate attention
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # Forecasting head (handles multiple patch sizes)
        self.head = ForecastingHead(d_model, patch_sizes)
        
    def forward(
        self,
        context: torch.Tensor,
        context_mask: torch.Tensor | None = None,
        prediction_length: int | None = None,
        target: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with RevIN normalization (everything stays normalized).
        
        Args:
            context: Shape (batch, num_variates, context_len, patch_size)
            context_mask: Shape (batch, context_len) - 1 for valid, 0 for padding
            prediction_length: Number of patches to predict
            target: Optional target for training, shape (batch, num_variates, pred_len, patch_size)
                   Will be normalized using same statistics as context
            
        Returns:
            If target is None:
                predictions: Shape (batch, num_variates, pred_len, patch_size) - NORMALIZED
            If target is provided:
                (predictions, normalized_target): Both normalized with same RevIN stats
        """
        batch_size, num_variates, context_len, patch_size = context.shape
        
        # Get the appropriate RevIN layer for this patch size
        if str(patch_size) not in self.revin_layers:
            raise ValueError(
                f"Unsupported patch_size={patch_size}. "
                f"Supported sizes: {self.patch_sizes}"
            )
        revin = self.revin_layers[str(patch_size)]
        
        # Step 1: Normalize each variate independently using RevIN
        # Reshape to (batch * num_variates, context_len, patch_size)
        x = context.reshape(batch_size * num_variates, context_len, patch_size)
        x = revin(x, mode='norm')
        
        # Step 2: Embed patches
        x = self.embedding(x)  # (batch * num_variates, context_len, d_model)
        
        # Step 3: Flatten variates into single sequence
        # Reshape to (batch, num_variates * context_len, d_model)
        x = x.reshape(batch_size, num_variates, context_len, self.d_model)
        x = x.transpose(1, 2)  # (batch, context_len, num_variates, d_model)
        x = x.reshape(batch_size, context_len * num_variates, self.d_model)
        
        # Step 4: Create variate IDs for binary attention bias
        # Pattern: [0,1,2,0,1,2,0,1,2,...] for 3 variates
        variate_ids = torch.arange(num_variates, device=x.device).repeat(context_len)
        variate_ids = variate_ids.unsqueeze(0).expand(batch_size, -1)
        # Shape: (batch, context_len * num_variates)
        
        # Step 5: Create flattened mask
        # Expand mask to all variates
        if context_mask is not None:
            # context_mask: (batch, context_len)
            # Expand to (batch, context_len * num_variates)
            flat_mask = context_mask.unsqueeze(2).expand(-1, -1, num_variates)
            flat_mask = flat_mask.reshape(batch_size, context_len * num_variates)
        else:
            flat_mask = None
        
        # Step 6: Apply transformer layers with any-variate attention
        for layer in self.layers:
            x = layer(x, variate_ids, flat_mask)
        
        # Step 7: Reshape back to (batch, num_variates, context_len, d_model)
        x = x.reshape(batch_size, context_len, num_variates, self.d_model)
        x = x.transpose(1, 2)  # (batch, num_variates, context_len, d_model)
        
        # Step 8: Generate predictions (still in normalized space)
        all_predictions = []
        
        for v in range(num_variates):
            x_v = x[:, v, :, :]  # (batch, context_len, d_model)
            
            if prediction_length is not None and prediction_length > 0:
                # Use last valid embedding to predict future
                if context_mask is not None:
                    last_valid_idx = context_mask.sum(dim=1).long() - 1
                    last_embed = x_v[torch.arange(batch_size), last_valid_idx]
                else:
                    last_embed = x_v[:, -1, :]
                
                # Repeat for prediction length
                x_pred = last_embed.unsqueeze(1).repeat(1, prediction_length, 1)
            else:
                x_pred = x_v
            
            # Predict values (in normalized space)
            pred = self.head(x_pred, patch_size)
            all_predictions.append(pred)
        
        # Stack variates: (batch, num_variates, pred_len, patch_size)
        predictions = torch.stack(all_predictions, dim=1)
        
        # If target is provided, normalize it using the same RevIN statistics
        if target is not None:
            # Target shape: (batch, num_variates, pred_len, patch_size)
            # Reshape to (batch * num_variates, pred_len, patch_size)
            target_flat = target.reshape(batch_size * num_variates, -1, patch_size)
            
            # Normalize using the SAME statistics that were stored during context normalization
            # This is crucial - we use the same mean/std from context to normalize target
            target_normalized = revin(target_flat, mode='norm')
            
            # Reshape back to (batch, num_variates, pred_len, patch_size)
            target_normalized = target_normalized.reshape_as(target)
            
            return predictions, target_normalized
        
        # Return normalized predictions only (for inference)
        return predictions


def mse_loss(
    predictions: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Mean Squared Error loss.
    
    Args:
        predictions: Predicted values
        target: Ground truth
        mask: Optional mask (1 for valid, 0 for padding)
        
    Returns:
        Scalar loss
    """

    squared_error = (predictions - target) ** 2
    
    if mask is not None:
        # Expand mask to match squared_error shape
        while mask.ndim < squared_error.ndim:
            mask = mask.unsqueeze(-1)
        squared_error = squared_error * mask
        # Add small epsilon to denominator
        return squared_error.sum() / (mask.sum() + 1e-8)
    else:
        return squared_error.mean()


"""
Embedding networks that compress the Hα photon-count maps into a summary
vector for the NPE density estimator.

The current pipeline uses DensityEmbeddingPerFrameTemporalAdv (a shared 2D CNN
per frame followed by a 1D temporal convolution), built with
make_temporal_embedding. The single-frame and 3D-convolution variants are
kept for older trained posteriors and experiments.

WARNING: the trained posteriors are pickled and reference these classes by
name (originally as module `helper`, see posterior_io.load_posterior). Do not
rename the classes or their attributes.
"""

# neural networks
import torch.nn as nn


def get_activation(name):
    """Activation module by name ('relu', 'leaky_relu', 'gelu'); ReLU otherwise."""
    if name == 'relu':
        return nn.ReLU()
    elif name == 'leaky_relu':
        return nn.LeakyReLU(0.1)
    elif name == 'gelu':
        return nn.GELU()
    else:
        return nn.ReLU()


def _conv_output_length(input_length, kernel, stride, padding, dilation=1):
    """Output length of a convolution along one axis."""
    return (input_length + 2 * padding - dilation * (kernel - 1) - 1) // stride + 1


# -------------------------------------------------------------
# ============== ↓ Single-frame embeddings (old) ↓ ============
# -------------------------------------------------------------


class DensityEmbedding(nn.Module):
    """Fixed three-block CNN for a single 64x64 frame, embedding size 128."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
        )

    def forward(self, x):
        # (B, 64, 64) -> (B, 1, 64, 64)
        x = x.unsqueeze(1)
        return self.conv(x)


class DensityEmbeddingComplex(nn.Module):
    """Tunable CNN for a single square frame; each block halves the size."""

    def __init__(self,
                 input_shape=(64, 64),
                 num_conv_blocks=3,
                 base_channels=16,
                 kernel_size=3,
                 use_batchnorm=True,
                 dropout=0.0,
                 fc_dim=128,
                 activation='relu'):
        """
        Args:
            input_shape: (H, W) of the single-channel image (square).
            num_conv_blocks: Number of stride-2 conv blocks.
            base_channels: Channels of the first block; doubled per block
                (capped at 512).
            kernel_size: Convolution kernel size (3 or 5).
            use_batchnorm: Add batch norm after each convolution.
            dropout: Dropout probability before the output.
            fc_dim: Embedding size.
            activation: 'relu', 'leaky_relu' or 'gelu'.
        """
        super().__init__()
        assert input_shape[0] == input_shape[1], "code assumes square images; adapt if not."
        self.activation = get_activation(activation)
        padding = kernel_size // 2

        convs = []
        in_channels = 1
        channels = base_channels
        for _ in range(num_conv_blocks):
            convs.append(nn.Conv2d(in_channels, channels, kernel_size=kernel_size, stride=2, padding=padding))
            if use_batchnorm:
                convs.append(nn.BatchNorm2d(channels))
            convs.append(self.activation)
            in_channels = channels
            channels = min(channels * 2, 512)
        self.conv = nn.Sequential(*convs)

        height = input_shape[0] // (2**num_conv_blocks)
        width = input_shape[1] // (2**num_conv_blocks)
        self.flatten = nn.Flatten()

        self.head = nn.Sequential(
            nn.Linear(in_channels * height * width, fc_dim),
            self.activation,
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        )

    def forward(self, x):
        """Embed x of shape (B, H, W) or (B, C, H, W) into (B, fc_dim)."""
        if x.dim() == 3:
            x = x.unsqueeze(1)
        elif x.dim() == 4 and x.shape[1] != 1:
            x = x.mean(dim=1, keepdim=True)
        x = self.conv(x)
        x = self.flatten(x)
        x = self.head(x)
        return x


def make_embedding_from_trial(trial, input_shape=(64, 64)):
    """
    Single-frame embedding with hyperparameters from an optuna trial (old
    single-frame study).

    Args:
        trial: The optuna trial.
        input_shape: (H, W) of the frames.

    Returns:
        A DensityEmbeddingComplex.
    """
    num_conv_blocks = trial.suggest_categorical("num_conv_blocks", [2, 3])
    fc_dim = trial.suggest_categorical("fc_dim", [64, 128])

    return DensityEmbeddingComplex(
        input_shape=input_shape,
        num_conv_blocks=num_conv_blocks,
        kernel_size=3,
        use_batchnorm=False,
        dropout=0.0,
        fc_dim=fc_dim,
        activation='relu',
    )

# -------------------------------------------------------------
# ============== ↑ Single-frame embeddings (old) ↑ ============
# -------------------------------------------------------------


# -------------------------------------------------------------
# ================= ↓ Temporal embeddings ↓ ===================
# -------------------------------------------------------------


class DensityEmbeddingComplexTemporal(nn.Module):
    """Spatio-temporal 3D-convolution embedding for inputs of shape (T, H, W)."""

    def __init__(
        self,
        input_shape=(10, 64, 64),
        num_conv_blocks=3,
        base_channels=16,
        kernel_size=3,
        use_batchnorm=True,
        dropout=0.0,
        fc_dim=128,
        activation='relu',
        temporal_ks=None,
        temporal_stride=1,
    ):
        """
        Args:
            input_shape: (T, H, W), spatially square.
            num_conv_blocks: Number of Conv3d blocks; each halves H and W.
            base_channels: Channels of the first block; doubled per block
                (capped at 512).
            kernel_size: Spatial kernel size.
            use_batchnorm: Add batch norm after each convolution.
            dropout: Dropout probability before the output.
            fc_dim: Embedding size.
            activation: 'relu', 'leaky_relu' or 'gelu'.
            temporal_ks: Temporal kernel size; defaults to 3 (1 if T < 3).
            temporal_stride: Temporal stride (1 preserves T).
        """
        super().__init__()
        assert len(input_shape) == 3, "input_shape should be (T, H, W)"
        num_frames, height, width = input_shape
        assert height == width, "this implementation assumes square spatial dims (H == W)"
        self.activation = get_activation(activation)

        if temporal_ks is None:
            temporal_ks = 3 if num_frames >= 3 else 1
        temporal_padding = temporal_ks // 2
        spatial_padding = kernel_size // 2

        # The data is treated as a single channel for Conv3d.
        convs = []
        in_channels = 1
        channels = base_channels
        for _ in range(num_conv_blocks):
            convs.append(
                nn.Conv3d(
                    in_channels,
                    channels,
                    kernel_size=(temporal_ks, kernel_size, kernel_size),
                    stride=(temporal_stride, 2, 2),
                    padding=(temporal_padding, spatial_padding, spatial_padding),
                )
            )
            if use_batchnorm:
                convs.append(nn.BatchNorm3d(channels))
            convs.append(self.activation)
            in_channels = channels
            channels = min(channels * 2, 512)

        self.conv = nn.Sequential(*convs)

        frames_out, height_out, width_out = num_frames, height, width
        for _ in range(num_conv_blocks):
            frames_out = _conv_output_length(frames_out, temporal_ks, temporal_stride, temporal_padding)
            height_out = _conv_output_length(height_out, kernel_size, 2, spatial_padding)
            width_out = _conv_output_length(width_out, kernel_size, 2, spatial_padding)

        if height_out < 1 or width_out < 1 or frames_out < 1:
            raise ValueError("num_conv_blocks or temporal settings too large for given input dims")

        self.flatten = nn.Flatten()
        self.head = nn.Sequential(
            nn.Linear(in_channels * frames_out * height_out * width_out, fc_dim),
            self.activation,
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        )

    def forward(self, x):
        """
        Embed x of shape (B, T, H, W), (B, 1, T, H, W) or (B, C, T, H, W)
        (channels are mean-pooled) into (B, fc_dim).
        """
        if x.dim() == 4:
            x = x.unsqueeze(1)
        elif x.dim() == 5 and x.shape[1] != 1:
            x = x.mean(dim=1, keepdim=True)

        x = self.conv(x)
        x = self.flatten(x)
        x = self.head(x)
        return x


class DensityEmbeddingPerFrameTemporalAdv(nn.Module):
    """
    Factorised spatio-temporal embedding: a 2D CNN shared across frames gives
    a per-frame vector, a 1D temporal convolution aggregates the frames, and a
    linear head produces the fc_dim output.
    """

    def __init__(
        self,
        input_shape=(10, 64, 64),
        num_conv_blocks=3,
        base_channels=16,
        kernel_size=3,
        use_batchnorm=False,
        dropout=0.0,
        fc_dim=128,
        activation='relu',
        temporal_pool_output=1,
        per_frame_dim=None,
        temporal_num_layers=2,
        temporal_stride_first=1,
    ):
        """
        Args:
            input_shape: (T, H, W), spatially square.
            num_conv_blocks: Number of stride-2 Conv2d blocks per frame.
            base_channels: Channels of the first block; doubled per block
                (capped at 512).
            kernel_size: Spatial kernel size.
            use_batchnorm: Add batch norm after each convolution.
            dropout: Dropout probability before the output.
            fc_dim: Embedding size.
            activation: 'relu', 'leaky_relu' or 'gelu'.
            temporal_pool_output: Number of temporal bins kept by the final
                adaptive pooling (1 = global average).
            per_frame_dim: Optional projection size of the per-frame vector.
            temporal_num_layers: Number of temporal Conv1d layers.
            temporal_stride_first: Stride of the first temporal layer.
        """
        super().__init__()
        assert len(input_shape) == 3, "input_shape should be (T, H, W)"
        _, height, width = input_shape
        assert height == width, "this implementation assumes square spatial dims (H == W)"

        self.activation = get_activation(activation)
        padding = kernel_size // 2

        # The 2D frame encoder is shared across all frames.
        convs = []
        in_channels = 1
        channels = base_channels
        for _ in range(num_conv_blocks):
            convs.append(nn.Conv2d(in_channels, channels, kernel_size=kernel_size, stride=2, padding=padding))
            if use_batchnorm:
                convs.append(nn.BatchNorm2d(channels))
            convs.append(self.activation)
            in_channels = channels
            channels = min(channels * 2, 512)
        self.frame_encoder = nn.Sequential(*convs)
        self.frame_encoder_out_channels = in_channels

        # Global spatial pooling reduces each frame to one vector.
        self.spatial_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.temporal_pool_output = int(temporal_pool_output)

        if per_frame_dim is None:
            per_frame_dim = self.frame_encoder_out_channels
        self.per_frame_dim = per_frame_dim
        if per_frame_dim != self.frame_encoder_out_channels:
            self.per_frame_proj = nn.Linear(self.frame_encoder_out_channels, per_frame_dim)
        else:
            self.per_frame_proj = None

        layers = []
        for layer_index in range(int(temporal_num_layers)):
            stride = int(temporal_stride_first) if layer_index == 0 else 1
            layers.append(nn.Conv1d(per_frame_dim, per_frame_dim, kernel_size=3, padding=1, stride=stride))
            layers.append(self.activation)
        self.temporal_conv = nn.Sequential(*layers)

        self.temporal_pool = nn.AdaptiveAvgPool1d(self.temporal_pool_output)

        self.head = nn.Sequential(
            nn.Linear(per_frame_dim * self.temporal_pool_output, fc_dim),
            self.activation,
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        )

    def forward(self, x):
        """
        Embed x of shape (B, T, H, W), (B, 1, T, H, W) or (B, C, T, H, W)
        (channels are mean-pooled) into (B, fc_dim).
        """
        if x.dim() == 5 and x.shape[1] != 1:
            x = x.mean(dim=1)
        elif x.dim() == 5 and x.shape[1] == 1:
            x = x.squeeze(1)
        elif x.dim() == 4:
            pass
        else:
            raise ValueError(f"Unexpected input dims {x.shape}; expected (B, T, H, W) or (B, C, T, H, W)")

        batch_size, num_frames, height, width = x.shape
        frames = x.reshape(batch_size * num_frames, 1, height, width)

        # Encode every frame: (B*T, C_enc).
        features = self.frame_encoder(frames)
        features = self.spatial_pool(features)
        features = features.view(batch_size * num_frames, self.frame_encoder_out_channels)

        if self.per_frame_proj is not None:
            features = self.per_frame_proj(features)

        # Convolve along time: (B, per_frame_dim, T) -> (B, per_frame_dim, P).
        features = features.view(batch_size, num_frames, self.per_frame_dim)
        features = features.permute(0, 2, 1).contiguous()
        features = self.temporal_conv(features)
        features = self.temporal_pool(features)
        features = features.view(batch_size, -1)

        return self.head(features)


def make_temporal_embedding(
    num_conv_blocks,
    fc_dim,
    base_channels,
    temporal_pool_output,
    temporal_num_layers,
    temporal_stride_first,
    input_shape=(10, 64, 64),
):
    """
    Build the per-frame temporal embedding with the tuned architecture
    hyperparameters; the remaining settings are fixed (no dropout, kernel
    size 3, no batch norm, ReLU).

    Args:
        num_conv_blocks: Number of Conv2d blocks per frame.
        fc_dim: Embedding size.
        base_channels: Channels of the first Conv2d block.
        temporal_pool_output: Number of temporal bins after pooling.
        temporal_num_layers: Number of temporal Conv1d layers.
        temporal_stride_first: Stride of the first temporal layer.
        input_shape: (T, H, W) of the observations.

    Returns:
        A DensityEmbeddingPerFrameTemporalAdv.
    """
    return DensityEmbeddingPerFrameTemporalAdv(
        input_shape=input_shape,
        num_conv_blocks=num_conv_blocks,
        base_channels=base_channels,
        kernel_size=3,
        use_batchnorm=False,
        dropout=0.0,
        fc_dim=fc_dim,
        activation='relu',
        temporal_pool_output=temporal_pool_output,
        temporal_num_layers=temporal_num_layers,
        temporal_stride_first=temporal_stride_first,
    )

# -------------------------------------------------------------
# ================= ↑ Temporal embeddings ↑ ===================
# -------------------------------------------------------------

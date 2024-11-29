from typing import Optional, List

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def initialize_weights(m):
    if isinstance(m, nn.Linear):
        # Xavier/Glorot initialization for linear layers
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

    elif isinstance(m, nn.Conv1d):
        # Kaiming/He initialization for convolutional layers
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

    elif isinstance(m, nn.LayerNorm):
        # Standard initialization for LayerNorm
        nn.init.constant_(m.weight, 1.0)
        nn.init.constant_(m.bias, 0)

    elif isinstance(m, nn.MultiheadAttention):
        # Initialize attention layers with a smaller scale factor
        if hasattr(m, 'in_proj_weight') and m.in_proj_weight is not None:
            # Combined weight matrix for query, key, value projections
            nn.init.xavier_uniform_(m.in_proj_weight, gain=0.1)
        if hasattr(m, 'out_proj.weight'):
            # Output projection
            nn.init.xavier_uniform_(m.out_proj.weight, gain=0.1)

        # Initialize biases to zero
        if hasattr(m, 'in_proj_bias') and m.in_proj_bias is not None:
            nn.init.constant_(m.in_proj_bias, 0.)
        if hasattr(m, 'out_proj.bias'):
            nn.init.constant_(m.out_proj.bias, 0.)


class ResidualBlock(nn.Module):
    def __init__(self, hidden_size: int, dropout: float = 0.3):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.block(x))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 512):
        super(PositionalEncoding, self).__init__()
        self.dropout: nn.Dropout = nn.Dropout(p=dropout)

        # Create a long enough 'pe' matrix with values dependent on pos and i
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )  # (d_model/2,)

        pe[:, 0::2] = torch.sin(position * div_term)  # Apply sin to even indices in the array
        pe[:, 1::2] = torch.cos(position * div_term)  # Apply cos to odd indices in the array

        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
        self.register_buffer("pe", pe)  # Register as buffer to avoid updating during training

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, d_model = x.size()
        if seq_len > self.pe.size(1):
            # Dynamically expand positional encodings if sequence length exceeds max_len
            pe_extended = self._generate_extended_pe(seq_len, d_model)
            x = x + pe_extended[:, :seq_len, :]
        else:
            x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)

    def _generate_extended_pe(self, seq_len: int, d_model: int) -> torch.Tensor:
        pe = torch.zeros(seq_len, d_model, device=self.pe.device, dtype=self.pe.dtype)
        position = torch.arange(0, seq_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: (1, seq_len, d_model)
        return pe


class TransformerEncoder(nn.Module):
    def __init__(
            self,
            input_size: int,
            hidden_size: int = 256,
            num_layers: int = 2,
            num_heads: int = 8,
            dropout: float = 0.1,
    ):
        super(TransformerEncoder, self).__init__()
        self.input_projection = nn.Linear(input_size, hidden_size)
        self.pos_encoder = PositionalEncoding(hidden_size, dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(
            self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x = self.input_projection(x)
        x = self.pos_encoder(x)
        x = self.transformer(x, src_key_padding_mask=mask)

        return x


class SequenceClassifier(nn.Module):
    def __init__(
            self,
            input_size: int,
            hidden_size: int = 256,
            num_layers: int = 2,
            num_heads: int = 8,
            dropout: float = 0.3,
            kernel_sizes: Optional[List[int]] = None,
    ):
        super(SequenceClassifier, self).__init__()
        self.transformer = TransformerEncoder(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
        )

        # Multi-scale feature aggregation using Conv1d with different kernel sizes
        if kernel_sizes is None:
            kernel_sizes = [3, 5, 7]
        self.pooling_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(
                        in_channels=hidden_size,
                        out_channels=hidden_size // 2,
                        kernel_size=k,
                        padding=k // 2,
                    )
                )
                for k in kernel_sizes
            ]
        )

        # Classifier head
        classifier_input = (hidden_size // 2) * len(kernel_sizes)
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            ResidualBlock(hidden_size, dropout),
            nn.Linear(hidden_size, 1),
        )
        self.apply(initialize_weights)

    def forward(
            self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        encoded = self.transformer(x, mask=mask)

        # Multi-scale feature extraction
        pooled_features = []
        x_conv = encoded.transpose(1, 2)  # Shape: (batch_size, hidden_size, seq_len)
        for conv_layer in self.pooling_layers:
            pooled = F.adaptive_max_pool1d(conv_layer(x_conv), 1).squeeze(-1)
            pooled_features.append(pooled)

        # Concatenate pooled features
        combined_features = torch.cat(pooled_features, dim=-1)
        logits = self.classifier(combined_features).squeeze(-1)

        return logits


class ProfileClassifier(nn.Module):
    """Profile classifier that uses a pretrained sequence classifier to analyze author profiles"""

    def __init__(self, sequence_classifier: SequenceClassifier, threshold: float = 0.5,
                 aggregation: str = 'mean'):
        super().__init__()
        self.sequence_classifier = sequence_classifier
        self.threshold = threshold
        self.aggregation = aggregation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (num_conversations, seq_length, num_features)

        Returns:
            Tensor of shape (1,) containing profile-level predictions
        """
        # Get sequence-level predictions
        with torch.no_grad():
            sequence_logits = self.sequence_classifier(x)
            sequence_probs = torch.sigmoid(sequence_logits)

        # Aggregate sequence predictions
        if self.aggregation == 'mean':
            profile_prob = sequence_probs.mean()
        elif self.aggregation == 'median':
            profile_prob = sequence_probs.median()
        elif self.aggregation == 'mean_vote':
            profile_prob = (sequence_probs > self.threshold).float().mean()
        elif self.aggregation == 'total_vote':
            profile_prob = (sequence_probs > self.threshold).float().sum()
        else:
            raise ValueError(f"Invalid aggregation method: {self.aggregation}")

        return profile_prob

from itertools import chain, repeat

import torch
from torch import nn
from torch.nn import functional as F

from AttentionLayer import AttentionLayer


class MLPBinaryClassifier(nn.Module):
    def __init__(self, in_features, hidden_layers, units_per_layer, dropout=0.2, threshold=0.5, positive_weight=1, negative_weight=1, focal_alpha=0.25, focal_gamma=2.0, *args,
                 **kwargs):
        super().__init__(*args, **kwargs)

        self.model = nn.Sequential(
            *[nn.Linear(in_features, units_per_layer),
              nn.ReLU(),
              nn.Dropout(dropout), ],
            *chain(*repeat(
                [
                    nn.Linear(units_per_layer, units_per_layer),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                ], hidden_layers - 1)),
            nn.Linear(units_per_layer, 1),
            nn.Sigmoid()
        )
        self.hidden_layers = hidden_layers
        self.default_threshold = threshold
        self.negative_weight = negative_weight
        self.positive_weight = positive_weight
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

    def forward(self, x):
        if self.hidden_layers == 0:
            return x
        return self.model(x)

    def loss_function(self, output, target):
        # self.bce_loss_function(output, target)
        return self.focal_loss_function(output, target)

    def bce_loss_function(self, output, target):
        output = output.squeeze(1)
        return F.binary_cross_entropy(output, target.float(), weight=torch.where(target == 1,
                                                                                 self.positive_weight * torch.ones_like(output),
                                                                                 self.negative_weight * torch.ones_like(output))
                                      )

    def focal_loss_function(self, output, target):
        output = output.squeeze(1)
        target = target.float()

        bce_loss = F.binary_cross_entropy(output, target, reduction='none')

        pt = output * target + (1 - output) * (1 - target)
        focal_weight = (1 - pt) ** self.focal_gamma
        alpha_t = self.focal_alpha * target + (1 - self.focal_alpha) * (1 - target)
        focal_loss = alpha_t * focal_weight * bce_loss
        return focal_loss.mean()

    def predict(self, prob, threshold=None):
        if threshold is None:
            threshold = self.default_threshold
        with torch.no_grad():
            return (prob >= threshold).T.float()


class CustomModel(nn.Module):
    def __init__(self, vocab_size, context_length=128, n_heads=4, embed_dim=128, expand=1, hidden_layers=1, attention_layers=1, focal_alpha=0.25, focal_gamma=2.0, dropout=0.2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.context_length = context_length
        self.vocab_size = vocab_size

        self.token_embedding = nn.Embedding(self.vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(context_length, embed_dim)
        self.model = nn.Sequential(
            *chain(*repeat(
                [
                    AttentionLayer(embed_dim=embed_dim, n_heads=n_heads),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                ], attention_layers))
        )
        self.linear = MLPBinaryClassifier(embed_dim, hidden_layers=hidden_layers, units_per_layer=int(embed_dim * expand), focal_alpha=focal_alpha, focal_gamma=focal_gamma, dropout=dropout, *args,
                                          **kwargs)
        self.device = "cpu"

    def to(self, device):
        self.device = device
        return super().to(device)

    def forward(self, params):
        x, x_mask = params
        x, x_mask = x.to(self.device), x_mask.to(self.device)
        f1 = self.token_embedding.forward(x) + self.position_embedding(torch.arange(0, x.shape[1], device=x.device))
        f2 = self.model(f1)
        f3 = self.linear.forward((f1 + f2)[:, 0, :])
        return f3

    def loss_function(self, output, target):
        return self.linear.loss_function(output, target)

    def predict(self, prob, threshold=None):
        return self.linear.predict(prob, threshold=threshold)

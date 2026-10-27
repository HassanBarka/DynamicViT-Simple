import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------
# Attention masking (Eq. 9, 10, 11 de l'article)
# ---------------------------------------------------------------
def masked_softmax(attn_scores, policy, eps=1e-6):
    """
    attn_scores : P = QK^T / sqrt(C)          [B, H, N, N]
    policy      : masque de décision D_hat    [B, N]  (valeurs 0/1)

    G_ij = 1 si i == j, sinon D_hat_j          (Eq. 10)
    A_ij = exp(P_ij) G_ij / sum_k exp(P_ik) G_ik   (Eq. 11)
    """
    B, H, N, _ = attn_scores.shape
    policy = policy.reshape(B, 1, 1, N).to(attn_scores.dtype)
    eye = torch.eye(N, device=attn_scores.device,
                    dtype=attn_scores.dtype).view(1, 1, N, N)

    # policy_j si i != j ; 1 si i == j (auto-boucle)
    G = policy + (1.0 - policy) * eye

    # Stabilité numérique : on retire le max avant l'exponentielle
    attn_scores = attn_scores - attn_scores.amax(dim=-1, keepdim=True)
    attn = attn_scores.exp() * G
    return attn / (attn.sum(dim=-1, keepdim=True) + eps)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim=128, num_heads=4, dropout=0.1):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)

    def forward(self, x, policy=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(
            B, N, 3, self.num_heads, self.head_dim
        ).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn_scores = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        if policy is None:
            attn = torch.softmax(attn_scores, dim=-1)
        else:
            attn = masked_softmax(attn_scores, policy)

        attn = self.attn_dropout(attn)
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj_dropout(self.proj(out))


class TransformerBlock(nn.Module):
    def __init__(self, dim=128, num_heads=4, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mlp_ratio, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, policy=None):
        x = x + self.attn(self.norm1(x), policy)
        x = x + self.mlp(self.norm2(x))
        return x


# ---------------------------------------------------------------
# Module de prédiction local-global (Eq. 1 à 5)
# ---------------------------------------------------------------
class PredictionModule(nn.Module):
    def __init__(self, dim):
        super().__init__()
        c_prime = dim // 2  # C' = C / 2

        # MLP de projection (Eq. 1 et 2)
        self.mlp_in = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, c_prime),
            nn.GELU(),
        )
        # MLP de décision (Eq. 5) : [local, global] (dim) -> 2 classes
        self.mlp_out = nn.Sequential(
            nn.Linear(2 * c_prime, c_prime),
            nn.GELU(),
            nn.Linear(c_prime, c_prime // 2),
            nn.GELU(),
            nn.Linear(c_prime // 2, 2),
        )

    def forward(self, x, d_hat):
        """
        x     : tokens patchs (sans CLS)        [B, N, C]
        d_hat : décision courante D_hat          [B, N, 1]
        retourne les logits [B, N, 2] (0 = supprimer, 1 = conserver)
        """
        B, N, _ = x.shape

        z_local = self.mlp_in(x)                                  # Eq. 1

        # Eq. 2-3 : moyenne pondérée par D_hat sur les tokens restants
        z_global = (z_local * d_hat).sum(dim=1, keepdim=True)
        z_global = z_global / d_hat.sum(dim=1, keepdim=True).clamp(min=1e-6)

        z = torch.cat([z_local, z_global.expand(-1, N, -1)], dim=-1)  # Eq. 4
        return self.mlp_out(z)                                    # Eq. 5 (avant softmax)


# ---------------------------------------------------------------
# Modèle
# ---------------------------------------------------------------
class SmallViT(nn.Module):
    def __init__(
        self,
        num_classes=10,
        dim=128,
        num_heads=4,
        depth=8,
        keep_ratio=0.7,                 # rho : ratios des étages = rho, rho^2, rho^3
        use_pruning=True,
        prune_locations=(2, 4, 6),      # élagage AVANT les blocs d'indice 2, 4, 6
        gumbel_tau=1.0,
    ):
        super().__init__()
        self.keep_ratio = keep_ratio
        self.use_pruning = use_pruning
        self.prune_locations = list(prune_locations)
        self.gumbel_tau = gumbel_tau

        # Image 32x32 -> 8x8 = 64 patchs
        self.patch_embed = nn.Conv2d(3, dim, kernel_size=4, stride=4)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 65, dim))
        self.pos_drop = nn.Dropout(0.1)

        self.blocks = nn.ModuleList(
            [TransformerBlock(dim, num_heads) for _ in range(depth)]
        )

        if use_pruning:
            self.predictors = nn.ModuleList(
                [PredictionModule(dim) for _ in self.prune_locations]
            )

        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, images):
        B = images.shape[0]

        x = self.patch_embed(images).flatten(2).transpose(1, 2)  # [B, 64, C]
        N = x.shape[1]
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.pos_drop(x + self.pos_embed)

        # D_hat : décision cumulative (initialisée à 1, Eq. 6)
        d_hat = torch.ones(B, N, 1, device=x.device, dtype=x.dtype)
        policy = None
        decisions = []   # D_hat après chaque étage (entraînement)
        kept = []        # nombre de tokens gardés par étage (inférence)

        for i, block in enumerate(self.blocks):
            if self.use_pruning and i in self.prune_locations:
                s = self.prune_locations.index(i)
                logits_s = self.predictors[s](x[:, 1:], d_hat)

                if self.training:
                    # Eq. 7 : Gumbel-Softmax (straight-through), indice 1 = conserver
                    d = F.gumbel_softmax(
                        logits_s, tau=self.gumbel_tau, hard=True, dim=-1
                    )[..., 1:2]
                    d_hat = d_hat * d                       # Eq. 6 : D_hat <- D_hat * D
                    ones = torch.ones(B, 1, 1, device=x.device, dtype=x.dtype)
                    policy = torch.cat([ones, d_hat], dim=1).squeeze(-1)
                    decisions.append(d_hat)
                else:
                    # Eq. 17 : on garde exactement m_s = floor(rho^s * N) tokens
                    pi_keep = torch.softmax(logits_s, dim=-1)[..., 1]
                    m = int(math.floor(N * self.keep_ratio ** (s + 1)))
                    m = max(1, min(m, pi_keep.shape[1]))
                    idx = pi_keep.topk(m, dim=1).indices.sort(dim=1).values
                    patches = torch.gather(
                        x[:, 1:], 1,
                        idx.unsqueeze(-1).expand(-1, -1, x.shape[-1]),
                    )
                    x = torch.cat([x[:, :1], patches], dim=1)
                    d_hat = torch.ones(B, m, 1, device=x.device, dtype=x.dtype)
                    kept.append(m)

            x = block(x, policy)

        x = self.norm(x)
        logits = self.head(x[:, 0])
        tokens = x[:, 1:]   # train : 64 tokens ; éval : tokens restants

        aux = {"decisions": decisions, "kept": kept}
        return logits, tokens, aux
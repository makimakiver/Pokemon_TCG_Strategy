"""Solver net: a pointer policy over options + value head.

Deck-agnostic by construction — it scores each option from that option's runtime
features (``encode.featurize``), so the action space is variable-length and the
same weights pilot any deck. A learned STOP embedding lets the policy close
variable-length selections (the env's micro-step STOP action).

The scripted ``option_prior`` (encode.option_prior_scores, surfaced as the last
option feature) is added to the logits as a residual whose weight is **annealed
to 0** over training, so the net starts near scripted strength and then surpasses
it (ledger §1, plan §3.2).

Requires torch -> runs in the Docker engine image (add torch via rl/requirements.txt).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from rl.config import CONFIG

# Last option feature is the normalized scripted prior (see encode.OPTION_FEATURE_NAMES).
PRIOR_FEATURE_INDEX = CONFIG.option_feat_dim - 1


class PointerPolicy(nn.Module):
    def __init__(self, cfg=CONFIG):
        super().__init__()
        G, Fdim, H = cfg.global_feat_dim, cfg.option_feat_dim, cfg.hidden_dim
        self.global_encoder = nn.Sequential(
            nn.Linear(G, H), nn.ReLU(), nn.Linear(H, H))
        self.option_encoder = nn.Sequential(
            nn.Linear(Fdim, H), nn.ReLU(), nn.Linear(H, H))
        self.stop_embed = nn.Parameter(torch.zeros(H))
        self.value_head = nn.Sequential(
            nn.Linear(H, H), nn.ReLU(), nn.Linear(H, 1))
        self._scale = 1.0 / math.sqrt(H)
        nn.init.normal_(self.stop_embed, std=0.02)

    def forward(self, global_feat, option_feats, mask, prior_weight: float = 0.0):
        """Single-state forward.

        Args:
            global_feat: [G]
            option_feats: [N, Fdim]  (N may be 0)
            mask: [N+1]  (last slot = STOP)
            prior_weight: residual weight on the scripted prior
        Returns:
            logits: [N+1], value: scalar tensor
        """
        query = self.global_encoder(global_feat)                  # [H]
        N = option_feats.shape[0]
        if N > 0:
            opt_emb = self.option_encoder(option_feats)           # [N, H]
            all_emb = torch.cat([opt_emb, self.stop_embed.unsqueeze(0)], dim=0)  # [N+1, H]
            prior = option_feats[:, PRIOR_FEATURE_INDEX]          # [N]
            prior = torch.cat([prior, prior.new_zeros(1)], dim=0)  # STOP prior = 0
            pooled = opt_emb.mean(dim=0)
        else:
            all_emb = self.stop_embed.unsqueeze(0)                # [1, H]
            prior = torch.zeros(1, device=global_feat.device)
            pooled = torch.zeros_like(query)

        logits = (all_emb @ query) * self._scale + prior_weight * prior
        logits = logits.masked_fill(mask < 0.5, float("-inf"))
        value = self.value_head(query + pooled).squeeze(-1)
        return logits, value

    # --- convenience: numpy obs -> action ---------------------------------
    @torch.no_grad()
    def act(self, obs: dict, prior_weight: float = 0.0, greedy: bool = False):
        g = torch.as_tensor(obs["global"], dtype=torch.float32)
        o = torch.as_tensor(obs["options"], dtype=torch.float32)
        m = torch.as_tensor(obs["mask"], dtype=torch.float32)
        logits, value = self.forward(g, o, m, prior_weight)
        if torch.isinf(logits).all():
            return obs["n_options"], 0.0, value.item()   # only STOP legal
        dist = torch.distributions.Categorical(logits=logits)
        action = int(logits.argmax()) if greedy else int(dist.sample())
        return action, float(dist.log_prob(torch.tensor(action))), value.item()

    def evaluate(self, obs: dict, action: int, prior_weight: float = 0.0):
        """Differentiable log-prob, entropy, value for a stored transition."""
        g = torch.as_tensor(obs["global"], dtype=torch.float32)
        o = torch.as_tensor(obs["options"], dtype=torch.float32)
        m = torch.as_tensor(obs["mask"], dtype=torch.float32)
        logits, value = self.forward(g, o, m, prior_weight)
        dist = torch.distributions.Categorical(logits=logits)
        a = torch.tensor(min(action, logits.shape[0] - 1))
        return dist.log_prob(a), dist.entropy(), value


def prior_weight_at(update: int, cfg=CONFIG) -> float:
    """Linear anneal of the scripted-prior residual weight from start -> end."""
    if cfg.prior_anneal_updates <= 0:
        return cfg.prior_weight_end
    frac = min(1.0, update / cfg.prior_anneal_updates)
    return cfg.prior_weight_start + frac * (cfg.prior_weight_end - cfg.prior_weight_start)


def save(policy: PointerPolicy, path):
    torch.save(policy.state_dict(), path)


def load(path, cfg=CONFIG) -> PointerPolicy:
    p = PointerPolicy(cfg)
    p.load_state_dict(torch.load(path, map_location=cfg.device))
    p.eval()
    return p

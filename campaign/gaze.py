"""H2-G token layout and every-layer information barriers (True = blocked)."""
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F

BINS = 224

def quantize_gaze(uv, valid):
    if uv.shape != (len(valid), 2):
        raise ValueError('Expected normalized gaze [B,2] and validity [B]')
    observed = uv[valid.bool()]
    if not torch.isfinite(observed).all() or ((observed < 0) | (observed > 1)).any():
        raise ValueError('Valid gaze must be finite normalized camera coordinates')
    # Invalid values never reach integer conversion or embedding lookup.
    safe = torch.where(valid.bool()[:, None], uv, 0.)
    return (safe * BINS).floor().long().clamp(0, BINS - 1)

def gaze_ce(logits, labels, valid):
    per_coordinate = F.cross_entropy(logits.float().reshape(-1, BINS),
                                     labels.reshape(-1), reduction='none').reshape(-1, 2)
    return torch.where(valid[:, None], per_coordinate, 0.).sum() / (2 * valid.sum()).clamp_min(1)

@dataclass(frozen=True)
class Layout:
    prefix: int
    context: int
    queries: int

    @property
    def gaze(self): return slice(self.prefix, self.prefix + 3)
    @property
    def action(self): return slice(self.prefix + 3 + self.context, self.length)
    @property
    def length(self): return self.prefix + 3 + self.context + self.queries

    def mask(self, valid, heads, condition=True):
        b, p, g, a, n = len(valid), self.prefix, self.prefix + 3, self.action.start, self.length
        mask = torch.ones((b, n, n), dtype=torch.bool, device=valid.device)
        mask[:, :p, :p] = False                    # clean + task only
        for q in range(p, g): mask[:, q, :q+1] = False  # shifted causal gaze
        mask[:, g:a, :a] = False                   # policy observations
        mask[:, a:, :] = False                     # action-conditioning queries
        allow = valid.bool() if condition else torch.zeros_like(valid, dtype=torch.bool)
        mask[:, g:, p:g] = ~allow[:, None, None]    # also blocks indirect gaze paths
        # Invalid gaze queries still read the clean prefix; never all-masked.
        return mask.repeat_interleave(heads, dim=0)

class GazeTokens(nn.Module):
    def __init__(self, width=256):
        super().__init__()
        self.task = nn.Parameter(torch.randn(1, 1, width) * .02)
        self.bos = nn.Parameter(torch.randn(1, 1, width) * .02)
        self.x = nn.Embedding(BINS, width)
        self.y = nn.Embedding(BINS, width)
        self.classifier = nn.Linear(width, BINS)

    def forward(self, labels):
        return torch.cat((self.bos.expand(len(labels), -1, -1),
                          self.x(labels[:, 0])[:, None], self.y(labels[:, 1])[:, None]), dim=1)

    def logits(self, tokens, layout):
        # BOS -> x; x -> y. The y target is never read by either predictor.
        return self.classifier(tokens[:, layout.prefix:layout.prefix+2])

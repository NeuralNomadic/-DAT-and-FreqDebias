"""Training methods. Each method implements `train_step` (one optimizer update)."""
from .standard import Standard
from .pgd_at import PGDAT
from .trades import TRADES
from .base_at import BaseAT
from .fd import FreqDebias
from .dat import DAT

METHODS = {
    "standard": Standard,   # E0
    "pgd_at": PGDAT,        # E1
    "trades": TRADES,       # E2
    "base_at": BaseAT,      # E3
    "fd": FreqDebias,       # E4, E5, A1-A4
    "dat": DAT,             # E6, A5, A6
}


def build_method(cfg, model, device, extra=None):
    return METHODS[cfg["method"]](cfg, model, device, extra or {})


def method_needs_partner(cfg) -> bool:
    """Whether the dataset must supply a same-class partner image (E4/E5/A1-A4 and A6)."""
    return cfg["method"] == "fd" or (cfg["method"] == "dat" and cfg.get("generator", "learned") == "partner")

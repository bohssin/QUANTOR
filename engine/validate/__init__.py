"""Fold geometry and validation. Plan §12."""
from .folds import (
    Fold,
    WalkForwardResult,
    anchored_folds,
    attribute_trades,
    purge_bars_for,
    purged_kfold,
    rolling_folds,
    train_mask,
    walk_forward,
)

__all__ = ["Fold", "WalkForwardResult", "anchored_folds", "attribute_trades",
           "purge_bars_for", "purged_kfold", "rolling_folds", "train_mask",
           "walk_forward"]

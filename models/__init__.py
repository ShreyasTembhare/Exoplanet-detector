"""Vetter model architectures.

  * :class:`ResNet1DClassifier` -- legacy single-tower model (Phase 0+).
  * :class:`TwoTowerResNet1D`   -- Phase 4 two-tower architecture.
  * :class:`ExoMinerVetter`     -- Phase 6 multi-input late-fusion model.
  * :class:`TransformerDetector` -- Phase 7 raw-light-curve transformer.
"""

from .exominer import (
    SCALAR_FEATURE_NAMES,
    ExoMinerVetter,
    make_exominer_inputs,
)
from .resnet1d import (
    ResNet1DClassifier,
    TwoTowerResNet1D,
    load_checkpoint,
    make_two_channel,
    make_two_tower_inputs,
)
from .transformer_detector import (
    TransformerDetector,
    make_token_inputs,
)

__all__ = [
    "ResNet1DClassifier", "TwoTowerResNet1D",
    "make_two_channel", "make_two_tower_inputs", "load_checkpoint",
    "ExoMinerVetter", "SCALAR_FEATURE_NAMES", "make_exominer_inputs",
    "TransformerDetector", "make_token_inputs",
]

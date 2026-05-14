from __future__ import annotations

from v2v_rt.config import load_config


def test_rtx3090_accepts_static_background_path() -> None:
    config = load_config("configs/rtx3090.yaml")

    assert config.models is not None
    assert config.models.static_background_path == "assets/backgrounds/rtx3090_static_background.png"
    assert config.models.segmentation_model_selection == 1
    assert config.models.segmentation_mask_threshold == 0.12
    assert config.models.segmentation_feather_sigma == 1.2
    assert config.models.segmentation_temporal_smoothing == 0.22
    assert config.models.identity_anchor_enabled is True
    assert config.models.identity_anchor_strength == 0.85
    assert config.models.identity_anchor_rolling_window == 12

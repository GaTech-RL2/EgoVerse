"""Stage 2 smoke: Qwen/T5 language conditioning wiring.

1. Pure-logic test of HPT._build_prompts (no model download).
2. Real QwenPooledEncoder / QwenPerTokenEncoder compute_latent shape check
   (downloads Qwen/Qwen3-Embedding-0.6B on first run).
"""
import torch
from omegaconf import OmegaConf
from egomimic.algo.hpt.algo import HPT


def test_build_prompts():
    class Stub:
        pass
    s = Stub()
    s.annotation_key = "annotations"
    s.annotation_sampling_mode = "first"
    s.default_prompt = "GO"
    out = HPT._build_prompts(s, {"annotations": [["a", "b"], [], ["c"]]}, 3)
    assert out == ["a", "GO", "c"], out
    # missing key -> all default
    s.annotation_key = None
    assert HPT._build_prompts(s, {}, 4) == ["GO"] * 4
    # random mode stays within the per-item annotation list
    s.annotation_key = "annotations"
    s.annotation_sampling_mode = "random"
    out = HPT._build_prompts(s, {"annotations": [["x", "y", "z"]]}, 1)
    assert out[0] in ("x", "y", "z"), out
    print("OK _build_prompts logic (first/random/default-fallback)")


def _specs(latent=16, dim=256):
    # init_cross_attn reads these keys directly (init_domain_stem passes
    # stem_spec[modality].specs.cross_attn, i.e. the flat cross_attn block).
    return OmegaConf.create(
        {
            "crossattn_latent": latent,
            "crossattn_heads": 8,
            "crossattn_dim_head": 64,
            "crossattn_modality_dropout": 0.1,
            "modality_embed_dim": dim,
        }
    )


def test_qwen_stems():
    from egomimic.models.stems.text_encoders import (
        QwenPooledEncoder,
        QwenPerTokenEncoder,
    )

    prompts = ["push the T block to the goal", ""]
    for cls in (QwenPooledEncoder, QwenPerTokenEncoder):
        enc = cls(output_dim=256, freeze=True, dtype="float32")
        enc.init_cross_attn(_specs())
        with torch.no_grad():
            tok = enc.compute_latent(prompts)
        assert tok.shape == (2, 16, 256), (cls.__name__, tuple(tok.shape))
        print(f"OK {cls.__name__}.compute_latent -> {tuple(tok.shape)}")


if __name__ == "__main__":
    test_build_prompts()
    test_qwen_stems()
    print("STAGE2 QWEN SMOKE: ALL OK")

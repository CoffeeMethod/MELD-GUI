"""Checkpoint-format compatibility that does not need the real weights.

v5 checkpoints store one prototype per family; v7+ store one per generator and
map generators to families. Both must load into the same module, and the
family logit must be the nearest member generator's logit, as in the
reference meld.py.
"""

from __future__ import annotations

import json

import pytest
import torch

from meld_gui.core.canonical import canonicalize
from meld_gui.core.thresholds import ThresholdTable, stratum_label

transformers = pytest.importorskip("transformers")


def _write_checkpoint(tmp_path, *, v7: bool):
    from safetensors.torch import save_file
    from transformers import AutoConfig, AutoModel

    # A tiny ModernBERT in the same config.json shape the real checkpoint uses.
    # layer_types must contain both attention kinds: with only one, transformers
    # 5.x adds flat rope keys and then rejects its own rope_parameters.
    (tmp_path / "config.json").write_text(json.dumps({
        "architectures": ["ModernBertForMaskedLM"], "model_type": "modernbert",
        "vocab_size": 64, "hidden_size": 32, "intermediate_size": 64,
        "num_hidden_layers": 2, "num_attention_heads": 2,
        "max_position_embeddings": 64, "global_attn_every_n_layers": 3,
        "local_attention": 8, "layer_types": ["full_attention", "sliding_attention"],
        "pad_token_id": 0, "bos_token_id": 1, "eos_token_id": 2,
        "cls_token_id": 1, "sep_token_id": 2, "transformers_version": "5.8.0",
        "rope_parameters": {
            "full_attention": {"rope_theta": 160000.0, "rope_type": "default"},
            "sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"},
        },
    }))
    backbone = AutoModel.from_config(AutoConfig.from_pretrained(tmp_path))

    r, n_fam, n_ops = 4, 3, 2
    gen_family = [0, 0, 1, 2, 2] if v7 else None
    n_proto = len(gen_family) if v7 else n_fam
    torch.manual_seed(0)
    state = {f"backbone.{k}": v.contiguous() for k, v in backbone.state_dict().items()}
    state.update({
        "style_proj.weight": torch.randn(r, 32),
        "style_ln.weight": torch.ones(r),
        "style_ln.bias": torch.zeros(r),
        "human_anchors": torch.randn(2, r),
        "log_tau": torch.tensor(0.0),
        "op_protos": torch.randn(n_ops, r),
        "op_bias": torch.zeros(n_ops),
        ("gen_protos" if v7 else "family_protos"): torch.randn(n_proto, r),
        ("gen_bias" if v7 else "family_bias"): torch.randn(n_proto),
    })
    save_file(state, str(tmp_path / "model.safetensors"))
    cfg = {
        "style_rank": r, "backbone_hidden_size": 32, "n_human_anchors": 2,
        "n_families": n_fam, "families": ["a", "b", "c"], "n_ops": n_ops,
        "ops": ["x", "y"], "rho": 0.25, "tau_agg": 1.0, "max_length": 16,
        "score_offsets": {"overall": {"fpr_0.01": 1.0, "fpr_0.05": 0.0, "fpr_0.1": -1.0}},
    }
    if v7:
        cfg.update({"generator_family": gen_family, "null_reduction": "max",
                    "scoring_contract": {"window_overlap": "none"}, "cap_tokens": 64})
    (tmp_path / "meld_config.json").write_text(json.dumps(cfg))
    return state


@pytest.mark.parametrize("v7", [False, True])
def test_both_checkpoint_formats_load(tmp_path, v7):
    from meld_gui.core.model import Meld

    _write_checkpoint(tmp_path, v7=v7)
    model = Meld(tmp_path)
    n_proto = 5 if v7 else 3
    assert tuple(model.gen_protos.shape) == (n_proto, 4)
    assert int(model.gen_member.sum()) == n_proto
    ids = torch.tensor([[1, 5, 6, 7, 2]])
    mask = torch.ones_like(ids)
    special = torch.tensor([[1, 0, 0, 0, 1]])
    (row,) = model.forward_batch(ids, mask, special)
    assert row.family_logits.shape == (5, 3)
    assert torch.isfinite(row.family_logits).all()


def test_family_logit_is_nearest_member_generator(tmp_path):
    from meld_gui.core.model import Meld

    _write_checkpoint(tmp_path, v7=True)
    model = Meld(tmp_path)
    u = torch.randn(1, 3, 4)
    tau = model.log_tau.exp()
    gen = -tau * model._sqdist(u, model.gen_protos) + model.gen_bias.view(1, 1, -1)
    expected = torch.stack(
        [gen[..., [0, 1]].amax(-1), gen[..., [2]].amax(-1), gen[..., [3, 4]].amax(-1)], -1
    )
    got = torch.where(model.gen_member, gen.unsqueeze(-1), gen.new_tensor(float("-inf"))).amax(2)
    assert torch.allclose(got, expected)


def test_canonicalize_strips_markdown_and_folds_characters():
    raw = "# Title\n\n**Bold** \u2014 text with \u0430 cyrillic\u200b letter.\n- item"
    out = canonicalize(raw)
    assert out == "Title\n\nBold, text with a cyrillic letter. item"
    assert canonicalize(out) == out  # idempotent


def test_threshold_table_ignores_extra_keys_and_labels_new_strata():
    table = ThresholdTable({
        "overall": {"fpr_0.01": 3.9, "fpr_0.05": 2.3, "fpr_0.1": 0.7},
        "overall_pooled": {"fpr_0.01": 1.6, "fpr_0.05": -2.0, "fpr_0.1": -3.6},
        "fallback_policy": "max_strata",
        "strata": {
            "student_essay": {"n": 10, "fpr_0.01": 1.4, "fpr_0.05": -1.9, "fpr_0.1": -3.3},
            "broken": {"n": 1},
        },
    })
    assert table.strata == ["overall", "student_essay"]
    assert table.options()[1]["label"] == "Student essay"
    assert stratum_label("brand_new_type") == "Brand new type"

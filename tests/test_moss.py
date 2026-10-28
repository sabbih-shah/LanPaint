"""CPU regression tests for MOSS masks, compositing and dual guidance."""

import importlib.util
import json
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
# Use the package's supported lightweight ComfyUI stubs for sampler unit tests.
_spec = importlib.util.spec_from_file_location("lanpaint_moss_test", ROOT / "__init__.py", submodule_search_locations=[str(ROOT)])
_package = importlib.util.module_from_spec(_spec)
import sys

sys.modules[_spec.name] = _package
_spec.loader.exec_module(_package)
from lanpaint_moss_test.src.LanPaint import nodes as lan
from lanpaint_moss_test.src.LanPaint.moss import (
    MossLanPaintPrepareImage,
    MossLanPaintPrepareMask,
    MossLanPaintComposite,
)


@pytest.fixture(autouse=True)
def linear_cfg(monkeypatch):
    def cfg(model, positive, negative, scale, x, timestep, **kwargs):
        return negative + (positive - negative) * scale

    monkeypatch.setattr(lan, "cfg_function", cfg, raising=False)


def test_all_thin_pixel_positions_survive():
    latent = {"samples": torch.zeros(256, 128, 1, 1)}
    mask = torch.eye(256).reshape(256, 16, 16)
    (result,) = MossLanPaintPrepareMask().prepare(latent, mask)
    assert result["noise_mask"].shape == (256, 1, 1, 1)
    assert result["noise_mask"].sum() == 256
    assert "noise_mask" not in latent


def test_dilation_crosses_latent_boundary():
    latent = {"samples": torch.zeros(1, 128, 1, 2)}
    mask = torch.zeros(1, 16, 32)
    mask[0, 8, 15] = 1
    (a,) = MossLanPaintPrepareMask().prepare(latent, mask, 0)
    (b,) = MossLanPaintPrepareMask().prepare(latent, mask, 1)
    assert a["noise_mask"].sum() == 1
    assert b["noise_mask"].sum() == 2


@pytest.mark.parametrize("fill", [0.0, 1.0])
def test_empty_full_and_neutralization(fill):
    image = torch.rand(1, 32, 48, 3)
    neutral, mask = MossLanPaintPrepareImage().prepare(image, torch.full((1, 32, 48), fill))
    assert torch.equal(mask, torch.full_like(mask, fill))
    assert torch.equal(neutral, image) if fill == 0 else torch.all(neutral == 128 / 255)


@pytest.mark.parametrize(
    "image,mask",
    [
        (torch.zeros(1, 31, 32, 3), torch.zeros(1, 31, 32)),
        (torch.zeros(1, 32, 32, 3), torch.zeros(1, 16, 32)),
        (torch.zeros(1, 32, 32, 3), torch.full((1, 32, 32), float("nan"))),
        (torch.full((1, 32, 32, 3), float("inf")), torch.zeros(1, 32, 32)),
    ],
)
def test_reject_invalid_image_mask(image, mask):
    with pytest.raises(ValueError):
        MossLanPaintPrepareImage().prepare(image, mask)


@pytest.mark.parametrize("shape", [(1, 16, 2, 2), (1, 128, 2, 2, 1)])
def test_reject_wrong_latent(shape):
    with pytest.raises(ValueError):
        MossLanPaintPrepareMask().prepare({"samples": torch.zeros(shape)}, torch.zeros(1, 32, 32))


@pytest.mark.parametrize("cfg,big,needed", [(1.0, -0.5, True), (1.0, 1.0, False), (3.0, 3.0, True), (1.0, 3.0, True)])
def test_dual_guidance_branch_optimization(monkeypatch, cfg, big, needed):
    seen = []

    def calc(model, conds, x, sigma, options):
        seen.append(conds[1])
        return [torch.ones_like(x) * 2, torch.ones_like(x)]

    monkeypatch.setattr(lan, "calc_cond_batch", calc, raising=False)
    x = torch.zeros(1)
    normal, bidirectional = lan.sampling_function_LanPaint(None, x, torch.tensor([0.5]), "negative", "positive", cfg, big)
    assert (seen[0] is not None) == needed
    assert torch.equal(normal, torch.ones_like(x) * (1 + cfg))
    assert torch.equal(bidirectional, torch.ones_like(x) * (1 + big))


def test_pre_cfg_hook_preserved(monkeypatch):
    monkeypatch.setattr(lan, "calc_cond_batch", lambda *a: [torch.tensor([2.0]), torch.tensor([1.0])], raising=False)
    out, big = lan.sampling_function_LanPaint(
        None,
        torch.zeros(1),
        torch.tensor([0.5]),
        "n",
        "p",
        3.0,
        -0.5,
        model_options={"sampler_pre_cfg_function": [lambda a: [v * 2 for v in a["conds_out"]]]},
    )
    assert out.item() == 8 and big.item() == 1


def test_workflow_is_acyclic_and_uses_neutral_source():
    graph = json.loads((ROOT / "examples/MOSS/moss_lanpaint_api.json").read_text())
    visited, active = set(), set()

    def visit(key):
        assert key not in active
        if key in visited:
            return
        active.add(key)
        for v in graph[key]["inputs"].values():
            if isinstance(v, list):
                visit(v[0])
        active.remove(key)
        visited.add(key)

    for k in graph:
        visit(k)
    assert graph["7"]["inputs"]["pixels"] == ["6", 0]
    assert graph["14"]["inputs"]["source"] == ["4", 0]
    assert graph["14"]["inputs"]["mask"] == ["6", 1]


def test_source_fill_is_explicit():
    image = torch.rand(1, 32, 32, 3)
    filled, edit = MossLanPaintPrepareImage().prepare(image, torch.ones(1, 32, 32), source_fill="source")
    assert torch.equal(filled, image) and edit.all()


@pytest.mark.parametrize("feather", [0, 8, 16])
def test_composite_preserves_outside_pixels(feather):
    source = torch.rand(1, 64, 64, 3)
    generated = torch.rand_like(source)
    mask = torch.zeros(1, 64, 64)
    mask[:, 8:56, 8:56] = 1
    (out,) = MossLanPaintComposite().composite(source, generated, mask, feather)
    assert torch.equal(out[mask == 0], source[mask == 0])
    assert torch.equal(out[:, 32, 32], generated[:, 32, 32])
    if feather:
        assert torch.equal(out[:, 8, 8], source[:, 8, 8])


def test_full_mask_feather_has_no_source_dependence():
    generated = torch.rand(1, 32, 32, 3)
    (out,) = MossLanPaintComposite().composite(torch.zeros_like(generated), generated, torch.ones(1, 32, 32), 16)
    assert torch.equal(out, generated)


def test_native_cond_batch_callback_is_used(monkeypatch):
    seen = []

    def callback(args):
        seen.append(args)
        return [torch.tensor([2.0]), torch.tensor([1.0])]

    def unexpected(*args):
        pytest.fail("native conditioning callback was bypassed")

    monkeypatch.setattr(lan, "calc_cond_batch", unexpected, raising=False)
    x, sigma = torch.zeros(1), torch.tensor([0.5])
    options = {"sampler_calc_cond_batch_function": callback}
    out, big = lan.sampling_function_LanPaint("model", x, sigma, "negative", "positive", 1.0, -0.5, options)
    assert out.item() == 2 and big.item() == 0.5
    assert seen[0]["conds"] == ["positive", "negative"]
    assert seen[0]["model_options"] is options
    assert seen[0]["input"] is x and seen[0]["sigma"] is sigma


def test_moss_nodes_registered():
    assert set(_package.NODE_CLASS_MAPPINGS) >= {
        "MossLanPaintPrepareImage",
        "MossLanPaintPrepareMask",
        "MossLanPaintSampler",
        "MossLanPaintComposite",
        "MossLanPaintStatistics",
    }


def test_sampler_uses_native_lanpaint_and_restores_rng(monkeypatch):
    from lanpaint_moss_test.src.LanPaint.moss import MossLanPaintSampler

    class Model:
        load_device = torch.device("cpu")
        model_options = {}

        def clone(self):
            return Model()

        def set_model_unet_function_wrapper(self, fn):
            self.wrapper = fn

        def set_model_sampler_pre_cfg_function(self, fn):
            self.pre_cfg = fn

    calls = []

    def sample(self, model, *args, **kwargs):
        calls.append((args, kwargs))
        torch.rand(3)
        model.pre_cfg({"sigma": torch.tensor([0.5]), "conds_out": []})
        result = model.wrapper(
            lambda x, sigma, **c: x,
            {
                "input": torch.ones(1, 128, 1, 1),
                "timestep": torch.tensor([0.5]),
                "cond_or_uncond": [0, 1],
                "c": {},
            },
        )
        return ({"samples": result},)

    monkeypatch.setattr(lan.LanPaint_KSamplerAdvanced, "sample", sample)
    state = torch.random.get_rng_state().clone()
    (out,) = MossLanPaintSampler().sample(
        Model(),
        "positive",
        "negative",
        {"samples": torch.zeros(1, 128, 1, 1)},
        42,
        50,
        3.0,
        3,
        0.1,
        5.0,
        "Image First",
    )
    assert torch.equal(state, torch.random.get_rng_state())
    assert calls[0][1]["LanPaint_NumSteps"] == 3
    assert out["moss_lanpaint"]["denoiser_evaluations"] == 1
    assert out["moss_lanpaint"]["transformer_forwards"] == 1
    assert out["moss_lanpaint"]["branch_evaluations"] == 2

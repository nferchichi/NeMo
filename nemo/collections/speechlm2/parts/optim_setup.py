# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import re
from typing import Generator, Iterable

import hydra
import torch
from lightning import LightningModule

from nemo.utils import logging


def configure_optimizers(model: LightningModule):
    """
    Re-usable optimizer configuration function for top-level PyTorch Lightning modules in this collection.
    It sets up parameter freezing, optimizer, and LR scheduler.

    The ``model`` object is expected to have a ``model.cfg`` attribute with OmegaConf configuration.
    The following fields are expected:

    * ``optimizer`` with hydra-style ``_target_`` pointing to optimizer class, and the remaining options
        passed directly to its ``__init__`` method.

    * (optional) ``freeze_params`` with a list of regex pattern for identifying frozen parameters.

    * (optional) ``prevent_freeze_params`` with a list of regex pattern for keeping specific parameters trainable
        (overrides ``freeze_params``).

    * (optional) ``optim_param_groups`` with a list of per-group LR overrides for the optimizer.
        See :func:`build_param_groups` for the expected schema. When unset/empty, the optimizer is
        constructed from a single flat parameter iterable -- byte-identical to the pre-patch path.

    * (optional) ``lr_scheduler`` with hydra-style ``_target_`` pointing to LR scheduler class,
        and the remaining options passed directly to its ``__init__`` method.

    Returns:
        PyTorch Lightning Trainer-compatible dict with structure::

            {
                "optimizer": <optimizer>,
                "lr_scheduler": {"scheduler": <lr_scheduler>, "interval": "step", "frequency": 1}
            }

    """
    assert hasattr(model, "cfg"), "Expected `model.cfg` attribute to exist."
    assert "optimizer" in model.cfg, "Expected `model.cfg` to contain 'optimizer' configuration."

    freeze_patterns = model.cfg.get("freeze_params", []) or []
    keep_patterns = model.cfg.get("prevent_freeze_params", []) or []
    pg_specs = model.cfg.get("optim_param_groups", None)

    if not pg_specs:
        # ORIGINAL PATH -- byte-identical to pre-patch behavior. Any recipe that
        # does not set `model.optim_param_groups` (i.e. expA through expK and any
        # currently running job) goes through this branch.
        parameters = freeze_and_subset(
            model.named_parameters(),
            exclude_patterns=freeze_patterns,
            keep_patterns=keep_patterns,
        )
        optimizer = hydra.utils.instantiate(model.cfg.optimizer, parameters, _convert_='all')
    else:
        # NEW PATH -- per-group LR overrides for differential fine-tuning.
        # Each spec yields a separate PyTorch param group with its own `lr`,
        # which the LR scheduler then anneals from independently (via base_lrs).
        base_lr = float(model.cfg.optimizer.get("lr"))
        param_groups = build_param_groups(
            model.named_parameters(),
            freeze_patterns=freeze_patterns,
            keep_patterns=keep_patterns,
            pg_specs=pg_specs,
            base_lr=base_lr,
        )
        optimizer = hydra.utils.instantiate(model.cfg.optimizer, param_groups, _convert_='all')

    # Defense-in-depth: verify no `requires_grad=False` parameter ended up inside
    # any optimizer param group (which would make it eligible for weight decay /
    # momentum-driven updates despite being "frozen"). This is a static structural
    # check independent of `freeze_and_subset`/`build_param_groups`'s own logic, so
    # it will catch a regression in either of those functions, a bad `keep_patterns`
    # override, or any future refactor that accidentally reintroduces frozen params
    # into the optimizer.
    assert_frozen_params_excluded_from_optimizer(model, optimizer)

    # NOTE: the frozen-parameter fingerprint snapshot used to be taken HERE, but
    # `configure_optimizers()` runs *before* the trainer's strategy finishes
    # wrapping the model (DDP's constructor does a one-time, value-preserving
    # broadcast of rank 0's parameters/buffers to every other rank, and other
    # one-time device/precision materialization can also happen after this
    # point). Snapshotting this early produced spurious "drift" reports for
    # hundreds of parameters at the very first checkpoint save -- a false
    # positive from comparing against a pre-setup baseline, not real training-
    # induced mutation (confirmed via a live instrumented repro: the same
    # parameters showed zero drift when snapshotted at `on_fit_start` instead).
    # The real snapshot now happens in `DuplexS2SSpeechDecoderModel2.on_fit_start`
    # (see that method), which runs after strategy/DDP setup completes.

    ans = {"optimizer": optimizer}
    if "lr_scheduler" in model.cfg:
        lr_scheduler = hydra.utils.instantiate(model.cfg.lr_scheduler, optimizer)
        ans["lr_scheduler"] = {"scheduler": lr_scheduler, "interval": "step", "frequency": 1}
    return ans


def build_param_groups(
    named_parameters: Iterable[tuple[str, torch.nn.Parameter]],
    freeze_patterns: list[str],
    keep_patterns: list[str],
    pg_specs,
    base_lr: float,
) -> list[dict]:
    """
    Build PyTorch optimizer ``param_groups`` with per-group LR overrides.

    Args:
        named_parameters: output of ``torch.nn.Module.named_parameters()``.
        freeze_patterns: regex patterns whose matches get ``requires_grad=False`` and
            are EXCLUDED from every group (same semantics as :func:`freeze_and_subset`).
        keep_patterns: regex patterns that override ``freeze_patterns`` (same semantics
            as :func:`freeze_and_subset`).
        pg_specs: iterable of dicts; each entry describes ONE param group:

            * ``patterns``: list[str] -- regex patterns matched against parameter
              names; first matching spec wins (so order matters).
            * ``lr_scale``: float -- per-group LR is ``base_lr * lr_scale``.
              Defaults to 1.0 if absent.
            * ``name``: str (optional) -- used only for logging.

        base_lr: the optimizer's top-level ``lr`` (group lr = ``base_lr * lr_scale``).

    Returns:
        A list of dicts suitable as the first positional arg to a PyTorch optimizer:
        ``[{"params": [...], "lr": ...}, ...]``. Parameters that do not match any
        spec's pattern are placed into a default group at ``lr=base_lr``. Empty
        groups are dropped (PyTorch raises on an empty parameter group).

    Notes:
        * The PyTorch ``_LRScheduler`` reads ``base_lrs`` from each group's
          ``initial_lr`` (= ``lr`` at construction time), so NeMo's
          ``InverseSquareRootAnnealing`` / ``WarmupPolicy`` etc. will anneal each
          group from its OWN base. ``min_lr`` is a single scalar though, applied
          identically to every group.
        * First-match wins across ``pg_specs``: if a parameter matches multiple
          spec patterns, it is placed in the EARLIEST spec it matches. Reorder
          ``pg_specs`` if a different priority is desired.
    """
    compiled_exclude = [re.compile(p) for p in freeze_patterns]
    compiled_keep = [re.compile(p) for p in (keep_patterns or [])]

    spec_entries = []
    for i, spec in enumerate(pg_specs):
        patterns = list(spec.get("patterns", []))
        name = spec.get("name", f"group_{i}")
        lr_scale = float(spec.get("lr_scale", 1.0))
        spec_entries.append(
            {
                "name": name,
                "patterns": patterns,
                "compiled": [re.compile(p) for p in patterns],
                "lr": base_lr * lr_scale,
                "lr_scale": lr_scale,
                "params": [],
                "match_count": 0,
                "trainable_elems": 0,
            }
        )

    default_entry = {
        "name": "default",
        "patterns": [],
        "compiled": [],
        "lr": base_lr,
        "lr_scale": 1.0,
        "params": [],
        "match_count": 0,
        "trainable_elems": 0,
    }

    frozen_elems = 0
    for name, param in named_parameters:
        is_excluded = any(p.match(name) is not None for p in compiled_exclude)
        is_kept = any(p.match(name) is not None for p in compiled_keep)
        if is_excluded and not is_kept:
            param.requires_grad = False
            frozen_elems += param.numel()
            continue

        placed = False
        for entry in spec_entries:
            if any(p.match(name) is not None for p in entry["compiled"]):
                entry["params"].append(param)
                entry["match_count"] += 1
                entry["trainable_elems"] += param.numel()
                placed = True
                break
        if not placed:
            default_entry["params"].append(param)
            default_entry["match_count"] += 1
            default_entry["trainable_elems"] += param.numel()

    # Assemble final param_groups in the order [default, spec_0, spec_1, ...] and
    # drop any empty group (AdamW errors out on `optimizer got an empty parameter list`).
    final_groups: list[dict] = []
    for entry in [default_entry, *spec_entries]:
        if not entry["params"]:
            logging.warning(
                f"[optim_param_groups] group '{entry['name']}' has 0 trainable params "
                f"(patterns={entry['patterns']}); dropping. Check the regex or the "
                f"freeze_params interaction."
            )
            continue
        final_groups.append({"params": entry["params"], "lr": entry["lr"]})
        logging.info(
            f"[optim_param_groups] group '{entry['name']}': "
            f"lr={entry['lr']:.3e} (scale={entry['lr_scale']:.3g}) | "
            f"params={entry['match_count']} | trainable_elems={entry['trainable_elems']}"
        )

    total_trainable = sum(e["trainable_elems"] for e in [default_entry, *spec_entries])
    total = total_trainable + frozen_elems
    pct = (total_trainable / total) if total else 0.0
    logging.info(
        f"Parameters | trainable={total_trainable} ({pct:.2%}) | total={total} "
        f"| param_groups={len(final_groups)} (LR-differentiated)"
    )

    return final_groups


def freeze_and_subset(
    named_parameters: Iterable[tuple[str, torch.nn.Parameter]],
    exclude_patterns: list[str],
    keep_patterns: list[str] = None,
) -> Generator[torch.nn.Parameter, None, None]:
    """
    Utility used to freeze select model parameters, and skip them for the purpose
    of initializing an optimizer's parameter group.

    Args:
        named_parameters: The output of `torch.nn.Module.named_parameters()`
        exclude_patterns: A list of regex patterns matching parameter names to be frozen
            and excluded from optimization.
        keep_patterns: A list of regex patterns matching parameter names to be trained.
            This list overrides all matches to `exclude_patterns`.

    Returns:
        A generator over parameters, equivalent to calling `torch.nn.Module.parameters()`,
            that will be passed to the optimizer and trained.

    Example:

        >>> model = MyModel()
        ... # freeze all LLM parameters in "model.llm"
        ... params = freeze_and_subset(model.named_parameters(), ['^llm\..+$'])
        ... optimizer = torch.optim.AdamW(params, lr=1e-3)

    """
    exclude_counter = {p: 0 for p in exclude_patterns}

    if not keep_patterns:
        keep_counter = {}

        def _must_keep(_) -> bool:
            return False

    else:
        keep_counter = {p: 0 for p in keep_patterns}
        compiled_keep_patterns = [re.compile(p) for p in keep_patterns]

        def _must_keep(name: str) -> bool:
            for p in compiled_keep_patterns:
                if p.match(name) is not None:
                    keep_counter[p.pattern] += 1
                    return True
            return False

    compiled_exclude_patterns = [re.compile(p) for p in exclude_patterns]

    def _exclude(name: str) -> bool:
        for p in compiled_exclude_patterns:
            if p.match(name) is not None:
                exclude_counter[p.pattern] += 1
                return True
        return False

    trainable, nontrainable = 0, 0
    for name, param in named_parameters:
        discard = False
        if _exclude(name) and not _must_keep(name):
            param.requires_grad = False
            discard = True
        if not discard:
            yield param
            trainable += param.numel()
        else:
            nontrainable += param.numel()
    total = trainable + nontrainable

    logging.info(f"Parameters | trainable={trainable} ({trainable / total:.2%}) | total={total}")

    if unused_excluded_patterns := [k for k, v in exclude_counter.items() if v == 0]:
        msg = "['" + "', '".join(unused_excluded_patterns) + "']"
        logging.warning(f"Parameter freezing patterns UNMATCHED against any parameter: {msg} (bad regexp?)")

    if unused_keep_patterns := [k for k, v in keep_counter.items() if v == 0]:
        msg = "['" + "', '".join(unused_keep_patterns) + "']"
        logging.warning(f"Parameter freeze-preventing patterns UNMATCHED against any parameter: {msg} (bad regexp?)")


def is_frozen(module: torch.nn.Module) -> bool:
    return all(not p.requires_grad for p in module.parameters())


# ---------------------------------------------------------------------------
# Frozen-parameter integrity checks.
#
# `freeze_and_subset`/`build_param_groups` already correctly exclude every
# `requires_grad=False` parameter from the optimizer's param groups, so those
# parameters should never be touched by `optimizer.step()` (no gradient descent,
# no weight decay, no momentum). The two helpers below add a second, independent
# line of defense: a structural check that no frozen param leaked into the
# optimizer despite that logic, and a value-level fingerprint check that proves
# frozen params truly stayed bit-for-bit unchanged across training -- regardless
# of *how* an unwanted update might occur (a future refactor of the freeze logic,
# an unrelated callback writing into `state_dict()`, a DDP/bucket-view aliasing
# issue, etc.). Catching drift here, at checkpoint-save time, is far cheaper than
# discovering it later via degraded downstream eval metrics.
# ---------------------------------------------------------------------------


def assert_frozen_params_excluded_from_optimizer(model: LightningModule, optimizer: torch.optim.Optimizer) -> None:
    """
    Raise if any `requires_grad=False` parameter is present in any of the
    optimizer's param groups. A frozen parameter has no gradient, so it should
    never have been added to a group in the first place.
    """
    optimizer_param_ids = {id(p) for group in optimizer.param_groups for p in group["params"]}
    leaked = [name for name, p in model.named_parameters() if not p.requires_grad and id(p) in optimizer_param_ids]
    if leaked:
        raise RuntimeError(
            f"{len(leaked)} frozen parameter(s) (requires_grad=False) were found inside the "
            f"optimizer's param groups -- they would be subject to weight decay / momentum "
            f"updates despite being marked frozen. First few: {leaked[:10]}. This indicates a "
            f"bug in freeze_and_subset/build_param_groups or their caller."
        )


def _tensor_fingerprint(t: torch.Tensor) -> int:
    """
    Cheap, exact (bit-level) fingerprint of a tensor's current values. Works
    uniformly across dtypes (bf16/fp16/fp32/...) by hashing the raw byte
    representation, so it can't be fooled by NaN/Inf-related equality quirks
    and needs no dtype-specific handling. Not cryptographically collision-proof,
    but astronomically unlikely to collide for real (non-adversarial) model
    weight drift, and O(1) memory per tensor (no full clone retained).
    """
    with torch.no_grad():
        byte_view = t.detach().reshape(-1).contiguous().cpu().view(torch.uint8).to(torch.int64)
        n = byte_view.numel()
        # Position-weighted sum so a byte-order permutation can't hide a change.
        weights = torch.arange(1, n + 1, dtype=torch.int64)
        return int((byte_view * weights).sum().item())


def snapshot_frozen_param_fingerprints(model: LightningModule) -> dict[str, int]:
    """Return {param_name: fingerprint} for every currently-frozen parameter."""
    return {name: _tensor_fingerprint(p) for name, p in model.named_parameters() if not p.requires_grad}


def verify_frozen_params_unchanged(model: LightningModule, snapshot: dict[str, int]) -> list[str]:
    """
    Recompute fingerprints for every parameter name in ``snapshot`` and return
    the list of names whose fingerprint no longer matches (i.e. drifted).
    Empty list means all frozen parameters are still bit-for-bit unchanged.
    Parameters that no longer exist or are no longer frozen are skipped with a
    warning rather than silently ignored, so an unexpected architecture change
    doesn't quietly disable the check.
    """
    current = dict(model.named_parameters())
    drifted: list[str] = []
    for name, old_fp in snapshot.items():
        p = current.get(name)
        if p is None:
            logging.warning(f"[frozen-param-check] '{name}' no longer exists on the model; skipping.")
            continue
        if p.requires_grad:
            logging.warning(f"[frozen-param-check] '{name}' is no longer frozen (requires_grad=True); skipping.")
            continue
        if _tensor_fingerprint(p) != old_fp:
            drifted.append(name)
    return drifted

"""Checkpoint-configured loader for the official scGPT TransformerModel."""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any

import numpy as np

from .scgpt_adapter import AdapterError


def load_vocabulary(path: Path) -> dict[str, int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not all(
        isinstance(key, str) and isinstance(value, int) for key, value in payload.items()
    ):
        raise AdapterError("Vocabulary must be a JSON object mapping tokens to integer IDs")
    return payload


def _rank_bin(values: np.ndarray[Any, Any], bins: int) -> np.ndarray[Any, Any]:
    if bins < 2:
        raise AdapterError("n_input_bins must be at least two")
    result = np.zeros(values.shape, dtype=np.float32)
    for index, row in enumerate(values):
        positive = np.flatnonzero(row > 0)
        if len(positive):
            order = np.argsort(row[positive], kind="stable")
            ranks = np.empty(len(positive), dtype=np.int64)
            ranks[order] = np.arange(len(positive))
            result[index, positive] = 1 + np.floor(ranks * (bins - 1) / max(len(positive), 1))
    return result


def _truncate_and_pad(
    gene_ids: np.ndarray[Any, Any],
    raw_values: np.ndarray[Any, Any],
    transformed_values: np.ndarray[Any, Any],
    *,
    token_length: int,
    cls_id: int,
    pad_id: int,
    pad_value: float,
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Keep each cell's highest-expression genes and pad to a fixed length.

    The pan-cancer checkpoint was trained with ``trunc_by_sample=true``. Global
    first-column truncation would depend on H5AD column order and could discard
    the expressed genes in a cell, so selection is deterministic per cell.
    """
    if token_length < 2:
        raise AdapterError("token_length must leave room for CLS and at least one gene")
    if gene_ids.shape != raw_values.shape or raw_values.shape != transformed_values.shape:
        raise AdapterError("Gene IDs and values must have matching two-dimensional shapes")
    rows = raw_values.shape[0]
    capacity = token_length - 1
    output_ids = np.full((rows, token_length), pad_id, dtype=np.int64)
    output_values = np.full((rows, token_length), pad_value, dtype=np.float32)
    output_ids[:, 0] = cls_id
    output_values[:, 0] = 0.0
    for row_index in range(rows):
        positive = np.flatnonzero(raw_values[row_index] > 0)
        if len(positive) > capacity:
            order = np.argsort(-raw_values[row_index, positive], kind="stable")
            positive = positive[order[:capacity]]
        count = len(positive)
        if count:
            output_ids[row_index, 1 : count + 1] = gene_ids[row_index, positive]
            output_values[row_index, 1 : count + 1] = transformed_values[row_index, positive]
    return output_ids, output_values


class OfficialScGPTRunner:
    """Numpy-facing callable around a loaded official scGPT model."""

    def __init__(
        self,
        model: Any,
        *,
        device: str,
        cls_id: int,
        pad_id: int,
        pad_value: float,
        token_length: int | None,
        n_input_bins: int,
        value_transform: str,
    ) -> None:
        self.model = model
        self.device = device
        self.cls_id = cls_id
        self.pad_id = pad_id
        self.pad_value = pad_value
        self.token_length = token_length
        self.n_input_bins = n_input_bins
        self.value_transform = value_transform

    def modules(self) -> Any:
        return self.model.modules()

    def __call__(self, **kwargs: Any) -> dict[str, np.ndarray[Any, Any]]:
        import torch

        gene_ids = np.asarray(kwargs["gene_ids"], dtype=np.int64)
        raw_values = np.asarray(kwargs["values"], dtype=np.float32)
        if self.value_transform == "rank_bin":
            values = _rank_bin(raw_values, self.n_input_bins)
        elif self.value_transform != "none":
            raise AdapterError(f"Unsupported scGPT value_transform: {self.value_transform}")
        if self.token_length is not None:
            gene_ids, values = _truncate_and_pad(
                gene_ids,
                raw_values,
                values,
                token_length=self.token_length,
                cls_id=self.cls_id,
                pad_id=self.pad_id,
                pad_value=self.pad_value,
            )
        else:
            cls = np.full((len(values), 1), self.cls_id, dtype=np.int64)
            cls_values = np.zeros((len(values), 1), dtype=np.float32)
            gene_ids = np.concatenate([cls, gene_ids], axis=1)
            values = np.concatenate([cls_values, values], axis=1)
        src = torch.as_tensor(gene_ids, device=self.device)
        expression = torch.as_tensor(values, device=self.device)
        padding = src.eq(self.pad_id)
        precision = str(kwargs.get("precision", "float32"))
        dtype = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }[precision]
        autocast_enabled = self.device.startswith("cuda") and precision != "float32"
        with (
            torch.inference_mode(),
            torch.autocast(device_type="cuda", dtype=dtype, enabled=autocast_enabled),
        ):
            result = self.model(
                src,
                expression,
                src_key_padding_mask=padding,
                CLS=False,
                CCE=False,
                MVC=False,
                ECS=False,
                do_sample=False,
            )
        embedding = result["cell_emb"] if isinstance(result, dict) else result
        return {"cell_emb": embedding.detach().float().cpu().numpy()}


def load_official_scgpt(
    checkpoint_path: Path,
    vocabulary_path: Path,
    device: str,
    config: dict[str, Any],
) -> OfficialScGPTRunner:
    """Build the official model from checkpoint-adjacent args and load weights."""
    try:
        import torch
        from scgpt.model import TransformerModel  # type: ignore[import-not-found]
        from scgpt.tokenizer.gene_tokenizer import (  # type: ignore[import-not-found]
            GeneVocab,
        )
    except ImportError as exc:
        raise AdapterError(f"Official scGPT runtime is unavailable: {exc}") from exc
    args_value = config.get("model_args_path")
    args_path = Path(str(args_value)) if args_value else checkpoint_path.parent / "args.json"
    if not args_path.is_file():
        raise AdapterError(f"Checkpoint model args are missing: {args_path}")
    model_args = json.loads(args_path.read_text(encoding="utf-8"))
    if not isinstance(model_args, dict):
        raise AdapterError("Checkpoint args.json must contain an object")
    vocabulary = load_vocabulary(vocabulary_path)
    vocab_object = GeneVocab.from_file(vocabulary_path)
    pad_token = str(config.get("pad_token", "<pad>"))
    cls_token = str(config.get("cls_token", "<cls>"))
    for token in (pad_token, cls_token):
        if token not in vocabulary:
            raise AdapterError(f"Required scGPT token is missing from vocabulary: {token}")

    def required_int(*keys: str, default: int) -> int:
        for key in keys:
            value = model_args.get(key)
            if value is not None:
                return int(value)
        return default

    embsize = required_int("embsize", "d_model", default=512)
    nhead = required_int("nheads", "nhead", default=8)
    d_hid = required_int("d_hid", "d_model", default=512)
    nlayers = required_int("nlayers", default=12)
    optional = {
        "vocab": vocab_object,
        "dropout": float(model_args.get("dropout", 0.2)),
        "pad_token": pad_token,
        "pad_value": float(config.get("pad_value", -2)),
        "do_mvc": bool(model_args.get("MVC", False)),
        "do_dab": bool(model_args.get("DAB", False)),
        "use_batch_labels": bool(model_args.get("use_batch_labels", False)),
        "n_input_bins": int(config.get("n_input_bins", 51)),
        "pre_norm": bool(model_args.get("pre_norm", False)),
        "use_fast_transformer": bool(config.get("use_fast_transformer", False)),
    }
    signature = inspect.signature(TransformerModel)
    optional = {key: value for key, value in optional.items() if key in signature.parameters}
    model = TransformerModel(len(vocabulary), embsize, nhead, d_hid, nlayers, **optional)
    raw = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state = raw.get("model_state_dict", raw) if isinstance(raw, dict) else raw
    if not isinstance(state, dict):
        raise AdapterError("scGPT checkpoint does not contain a state dictionary")
    # Pan-cancer checkpoints trained with flash-attn store the mathematically
    # equivalent fused Q/K/V projection as ``Wqkv``. Standard PyTorch MHA calls
    # the same tensors ``in_proj_*``. Translating the names avoids requiring
    # flash-attn at inference while preserving the learned values exactly.
    normalized = dict(state)
    for key in list(normalized):
        if ".self_attn.Wqkv.weight" in key:
            normalized[key.replace(".self_attn.Wqkv.weight", ".self_attn.in_proj_weight")] = (
                normalized.pop(key)
            )
        elif ".self_attn.Wqkv.bias" in key:
            normalized[key.replace(".self_attn.Wqkv.bias", ".self_attn.in_proj_bias")] = (
                normalized.pop(key)
            )
    expected = model.state_dict()
    loadable = {
        key: value
        for key, value in normalized.items()
        if key in expected and tuple(value.shape) == tuple(expected[key].shape)
    }
    incompatible = model.load_state_dict(loadable, strict=False)
    required_prefixes = ("encoder.", "value_encoder.", "transformer_encoder.")
    essential_missing = [
        key for key in incompatible.missing_keys if key.startswith(required_prefixes)
    ]
    if essential_missing:
        raise AdapterError(
            "Checkpoint is missing essential embedding/transformer weights: "
            + ", ".join(essential_missing[:10])
        )
    if bool(config.get("checkpoint_strict", False)) and incompatible.missing_keys:
        raise AdapterError(
            "Strict checkpoint loading found missing instantiated-model weights: "
            + ", ".join(incompatible.missing_keys[:10])
        )
    model.to(device)
    model.eval()
    pad_value_raw = config.get("pad_value")
    if pad_value_raw is None:
        pad_value_raw = model_args.get("pad_value", -2)
    return OfficialScGPTRunner(
        model,
        device=device,
        cls_id=vocabulary[cls_token],
        pad_id=vocabulary[pad_token],
        pad_value=float(pad_value_raw),
        token_length=(
            int(config["token_length"])
            if config.get("token_length")
            else int(model_args["max_seq_len"])
            if model_args.get("max_seq_len")
            else None
        ),
        n_input_bins=int(config.get("n_input_bins", 51)),
        value_transform=str(config.get("value_transform", "rank_bin")),
    )

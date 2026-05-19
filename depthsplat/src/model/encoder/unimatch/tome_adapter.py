from __future__ import annotations

import importlib.util
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class ToMePatchError(RuntimeError):
    pass


TomeFnBundle = tuple[Callable[..., Any], Callable[..., Any], Callable[..., Any]]


@lru_cache(maxsize=None)
def _load_tome_fns(tome_root: str) -> TomeFnBundle:
    root = Path(tome_root).expanduser().resolve()
    merge_path = root / "tome" / "merge.py"
    utils_path = root / "tome" / "utils.py"

    if not merge_path.exists() or not utils_path.exists():
        raise ToMePatchError(
            f"Local ToMe files not found under {root}. Expected {merge_path} and {utils_path}."
        )

    def load_module(module_name: str, module_path: Path):
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            raise ToMePatchError(f"Failed to load ToMe module from {module_path}.")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    module_suffix = str(abs(hash(str(root))))
    merge_module = load_module(f"depthsplat_tome_merge_{module_suffix}", merge_path)
    utils_module = load_module(f"depthsplat_tome_utils_{module_suffix}", utils_path)
    return merge_module.merge_source, merge_module.merge_wavg, utils_module.parse_r


@lru_cache(maxsize=None)
def _make_tome_transformer_class(transformer_class: type[nn.Module]):
    class ToMeDinoVisionTransformer(transformer_class):
        def _tome_reset_state(self) -> None:
            active_r = self.r
            if self.training and self._tome_info["inference_only"]:
                active_r = 0

            self._tome_info["r"] = self._tome_info["parse_r"](
                self._tome_info["num_blocks"], active_r
            )
            self._tome_info["size"] = None
            self._tome_info["source"] = None
            self._tome_info["unmerge"] = []

        def _tome_restore_tokens(self, x: Tensor) -> Tensor:
            for unmerge in reversed(self._tome_info["unmerge"]):
                x = unmerge(x)
            return x

        def _get_intermediate_layers_not_chunked(self, x, n=1):
            self._tome_reset_state()
            x = self.prepare_tokens_with_masks(x)
            output, total_block_len = [], len(self.blocks)
            blocks_to_take = (
                range(total_block_len - n, total_block_len) if isinstance(n, int) else n
            )
            for i, blk in enumerate(self.blocks):
                x = blk(x)
                if i in blocks_to_take:
                    output.append(self._tome_restore_tokens(x))
            assert len(output) == len(blocks_to_take), (
                f"only {len(output)} / {len(blocks_to_take)} blocks found"
            )
            return output

        def _get_intermediate_layers_chunked(self, x, n=1):
            self._tome_reset_state()
            x = self.prepare_tokens_with_masks(x)
            output, i, total_block_len = [], 0, len(self.blocks[-1])
            blocks_to_take = (
                range(total_block_len - n, total_block_len) if isinstance(n, int) else n
            )
            for block_chunk in self.blocks:
                for blk in block_chunk[i:]:
                    x = blk(x)
                    if i in blocks_to_take:
                        output.append(self._tome_restore_tokens(x))
                    i += 1
            assert len(output) == len(blocks_to_take), (
                f"only {len(output)} / {len(blocks_to_take)} blocks found"
            )
            return output

        def forward_features(self, x, masks=None):
            self._tome_reset_state()
            return super().forward_features(x, masks)

        def forward_features_list(self, x_list, masks_list):
            self._tome_reset_state()
            return super().forward_features_list(x_list, masks_list)

    ToMeDinoVisionTransformer.__name__ = (
        f"ToMe{transformer_class.__name__}"
    )
    return ToMeDinoVisionTransformer


@lru_cache(maxsize=None)
def _make_tome_block_class(block_class: type[nn.Module]):
    class ToMeDinoBlock(block_class):
        def forward(self, x_or_x_list):
            if isinstance(x_or_x_list, list):
                return super().forward(x_or_x_list)

            if self.training:
                if self._tome_info["inference_only"]:
                    return super().forward(x_or_x_list)
                raise ToMePatchError(
                    "DepthSplat ToMe integration currently supports eval/inference only. "
                    "Set model.encoder.tome.inference_only=true."
                )

            x = x_or_x_list
            attn_size = self._tome_info["size"] if self._tome_info["prop_attn"] else None
            x_attn, metric = self.attn(self.norm1(x), size=attn_size)
            x = x + self.ls1(x_attn)

            r_schedule = self._tome_info["r"]
            r = r_schedule.pop(0) if r_schedule else 0
            if r > 0:
                merge, unmerge = _protected_bipartite_soft_matching(
                    metric,
                    r,
                    self._tome_info["protected_prefix_tokens"],
                )
                if self._tome_info["trace_source"]:
                    self._tome_info["source"] = self._tome_info["merge_source"](
                        merge,
                        x,
                        self._tome_info["source"],
                    )
                x, self._tome_info["size"] = self._tome_info["merge_wavg"](
                    merge,
                    x,
                    self._tome_info["size"],
                )
                self._tome_info["unmerge"].append(unmerge)

            x = x + self.ls2(self.mlp(self.norm2(x)))
            return x

    ToMeDinoBlock.__name__ = f"ToMe{block_class.__name__}"
    return ToMeDinoBlock


@lru_cache(maxsize=None)
def _make_tome_attention_class(attn_class: type[nn.Module]):
    class ToMeDinoAttention(attn_class):
        def forward(self, x: Tensor, size: Tensor | None = None, attn_bias=None):
            if attn_bias is not None:
                raise ToMePatchError(
                    "DepthSplat ToMe integration does not support nested-tensor attn_bias inputs."
                )

            bsz, num_tokens, channels = x.shape
            qkv = self.qkv(x).reshape(
                bsz, num_tokens, 3, self.num_heads, channels // self.num_heads
            )
            q, k, v = torch.unbind(qkv, dim=2)
            q, k, v = [t.transpose(1, 2) for t in (q, k, v)]

            attn = (q @ k.transpose(-2, -1)) * self.scale
            if size is not None:
                attn = attn + size.log()[:, None, None, :, 0]

            attn = attn.softmax(dim=-1)
            if self.attn_drop > 0 and self.training:
                attn = F.dropout(attn, p=self.attn_drop)

            x = (attn @ v).transpose(1, 2).contiguous().view(bsz, num_tokens, channels)
            x = self.proj_drop(self.proj(x))
            return x, k.mean(dim=1)

    ToMeDinoAttention.__name__ = f"ToMe{attn_class.__name__}"
    return ToMeDinoAttention


def _do_nothing(x: Tensor, mode: str | None = None) -> Tensor:
    return x


def _protected_bipartite_soft_matching(
    metric: Tensor,
    r: int,
    protected_prefix_tokens: int,
):
    protected = min(max(int(protected_prefix_tokens), 0), metric.shape[1])
    if protected >= metric.shape[1]:
        return _do_nothing, _do_nothing

    metric_body = metric[:, protected:, :]
    body_tokens = metric_body.shape[1]
    r = min(r, body_tokens // 2)
    if r <= 0:
        return _do_nothing, _do_nothing

    with torch.no_grad():
        metric_body = metric_body / metric_body.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        a, b = metric_body[..., ::2, :], metric_body[..., 1::2, :]
        scores = a @ b.transpose(-1, -2)

        node_max, node_idx = scores.max(dim=-1)
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

        unm_idx = edge_idx[..., r:, :]
        src_idx = edge_idx[..., :r, :]
        dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx)

    def merge(x: Tensor, mode: str = "mean") -> Tensor:
        prefix, body = x[..., :protected, :], x[..., protected:, :]
        src, dst = body[..., ::2, :], body[..., 1::2, :]
        n, src_tokens, c = src.shape

        unm = src.gather(dim=-2, index=unm_idx.expand(n, src_tokens - r, c))
        src = src.gather(dim=-2, index=src_idx.expand(n, r, c))
        dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=mode)

        merged_body = torch.cat([unm, dst], dim=-2)
        return torch.cat([prefix, merged_body], dim=-2)

    def unmerge(x: Tensor) -> Tensor:
        prefix, body = x[..., :protected, :], x[..., protected:, :]
        unm_len = unm_idx.shape[1]
        unm, dst = body[..., :unm_len, :], body[..., unm_len:, :]
        n, _, c = unm.shape

        src = dst.gather(dim=-2, index=dst_idx.expand(n, r, c))
        out = body.new_zeros((n, metric_body.shape[1], c))

        out[..., 1::2, :] = dst
        out.scatter_(dim=-2, index=(2 * unm_idx).expand(n, unm_len, c), src=unm)
        out.scatter_(dim=-2, index=(2 * src_idx).expand(n, r, c), src=src)
        return torch.cat([prefix, out], dim=-2)

    return merge, unmerge


def apply_tome_to_dinov2(
    model: nn.Module,
    *,
    tome_root: str = "/root/ToMe-main",
    r: int = 0,
    prop_attn: bool = True,
    inference_only: bool = True,
    trace_source: bool = False,
) -> nn.Module:
    merge_source, merge_wavg, parse_r = _load_tome_fns(tome_root)

    transformer_class = _make_tome_transformer_class(model.__class__)
    model.__class__ = transformer_class
    model.r = r

    protected_prefix_tokens = 1 + int(getattr(model, "num_register_tokens", 0))

    tome_info = {
        "r": [],
        "size": None,
        "source": None,
        "unmerge": [],
        "trace_source": trace_source,
        "prop_attn": prop_attn,
        "inference_only": inference_only,
        "protected_prefix_tokens": protected_prefix_tokens,
        "merge_source": merge_source,
        "merge_wavg": merge_wavg,
        "parse_r": parse_r,
        "num_blocks": sum(
            1
            for module in model.modules()
            if hasattr(module, "attn") and hasattr(module, "norm1") and hasattr(module, "norm2")
        ),
    }
    model._tome_info = tome_info

    patched_blocks = 0
    for module in model.modules():
        if hasattr(module, "attn") and hasattr(module, "norm1") and hasattr(module, "norm2"):
            module.__class__ = _make_tome_block_class(module.__class__)
            module._tome_info = tome_info
            module.attn.__class__ = _make_tome_attention_class(module.attn.__class__)
            patched_blocks += 1

    if patched_blocks == 0:
        raise ToMePatchError("Failed to find any DINOv2 transformer blocks to patch with ToMe.")

    return model

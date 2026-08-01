"""Comfy 侧 Cache-DiT 集成：官方 MiniMaxH3DiTModel BlockAdapter（Pattern_3）。"""
from __future__ import annotations
import logging
from typing import Any
from .h3_settings import (
    CACHE_DIT_BLOCKS_ATTR, CACHE_DIT_FORWARD_PATTERN, CACHE_DIT_MARK, CACHE_DIT_PKG,
)
from .quality_profiles import CacheDitResolved
LOGGER = logging.getLogger("MiniMaxH3.cache_dit")

def _import_cache_dit():
    try:
        import cache_dit
        from cache_dit import BlockAdapter, DBCacheConfig, ForwardPattern, TaylorSeerCalibratorConfig
    except ImportError as exc:
        raise RuntimeError(f"未安装 {CACHE_DIT_PKG}，无法启用 Cache-DiT") from exc
    return cache_dit, BlockAdapter, DBCacheConfig, ForwardPattern, TaylorSeerCalibratorConfig

def _forward_pattern(ForwardPattern: Any):
    pat = getattr(ForwardPattern, CACHE_DIT_FORWARD_PATTERN, None)
    if pat is None: raise RuntimeError(f"cache-dit 缺少 ForwardPattern.{CACHE_DIT_FORWARD_PATTERN}")
    return pat

def _build_adapter(transformer: Any, BlockAdapter: Any, ForwardPattern: Any):
    name = transformer.__class__.__name__
    if name != "MiniMaxH3DiTModel":
        raise TypeError(f"Cache-DiT 仅支持 MiniMaxH3DiTModel，实际为 {name}")
    blocks = getattr(transformer, CACHE_DIT_BLOCKS_ATTR, None)
    if blocks is None: raise ValueError(f"MiniMaxH3DiTModel 缺少 {CACHE_DIT_BLOCKS_ATTR!r}")
    return BlockAdapter(
        transformer=transformer, blocks=blocks, forward_pattern=_forward_pattern(ForwardPattern),
        has_separate_cfg=False,  # H3 checkpoint 为 CFG-distilled，单正分支
    )

def _db_config(cfg: CacheDitResolved, DBCacheConfig: Any, *, num_denoise_steps: int, mask: list[int] | None):
    return DBCacheConfig(
        num_inference_steps=int(num_denoise_steps), Fn_compute_blocks=cfg.Fn_compute_blocks,
        Bn_compute_blocks=cfg.Bn_compute_blocks, max_warmup_steps=cfg.max_warmup_steps,
        residual_diff_threshold=cfg.residual_diff_threshold,
        max_continuous_cached_steps=cfg.max_continuous_cached_steps,
        steps_computation_mask=mask, steps_computation_policy=cfg.scm_policy,
    )

def _scm_mask(cache_dit: Any, cfg: CacheDitResolved, num_denoise_steps: int) -> list[int] | None:
    if cfg.steps_computation_mask is not None: return list(cfg.steps_computation_mask)
    if str(cfg.scm_preset or "none") == "none": return None
    return list(cache_dit.steps_mask(mask_policy=cfg.scm_preset, total_steps=int(num_denoise_steps)))

def prepare_transformer_cache_dit(transformer: Any, cfg: CacheDitResolved, *, num_denoise_steps: int) -> Any:
    """在 denoise 前 enable/refresh Cache-DiT；num_denoise_steps = len(sigmas)-1（官方 H3 合同）。"""
    if not cfg.enabled or int(num_denoise_steps) < 1: return transformer
    cache_dit, BlockAdapter, DBCacheConfig, ForwardPattern, TaylorSeerCalibratorConfig = _import_cache_dit()
    mask = _scm_mask(cache_dit, cfg, num_denoise_steps)
    if bool(getattr(transformer, CACHE_DIT_MARK, False)):
        scm = None if str(cfg.scm_preset or "none") == "none" else cfg.scm_preset
        if scm is not None: mask = list(cache_dit.steps_mask(mask_policy=scm, total_steps=int(num_denoise_steps)))
        cache_dit.refresh_context(
            transformer,
            cache_config=DBCacheConfig().reset(
                num_inference_steps=int(num_denoise_steps), steps_computation_mask=mask,
                steps_computation_policy=None if scm is None else cfg.scm_policy,
            ),
            verbose=False,
        )
        LOGGER.info("Cache-DiT refreshed steps=%d profile=%s", num_denoise_steps, cfg.profile_id)
        return transformer
    calibrator = (TaylorSeerCalibratorConfig(taylorseer_order=cfg.taylorseer_order)
                  if cfg.enable_taylorseer else None)
    adapter = _build_adapter(transformer, BlockAdapter, ForwardPattern)
    LOGGER.info(
        "Enabling Cache-DiT on MiniMaxH3DiTModel Fn=%d Bn=%d W=%d R=%.3f MC=%d steps=%d profile=%s",
        cfg.Fn_compute_blocks, cfg.Bn_compute_blocks, cfg.max_warmup_steps,
        cfg.residual_diff_threshold, cfg.max_continuous_cached_steps, num_denoise_steps, cfg.profile_id,
    )
    cache_dit.enable_cache(
        adapter, cache_config=_db_config(cfg, DBCacheConfig, num_denoise_steps=num_denoise_steps, mask=mask),
        calibrator_config=calibrator, parallelism_config=None,
    )
    setattr(transformer, CACHE_DIT_MARK, True)
    return transformer

__all__ = ["prepare_transformer_cache_dit"]

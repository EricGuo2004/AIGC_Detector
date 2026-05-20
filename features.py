from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
from scipy.fft import fft2, fftshift
from scipy.fftpack import dctn
from scipy.stats import kurtosis, skew


@dataclass
class FeatureConfig:
    image_size: int = 256
    radial_bins: int = 64
    angular_bins: int = 18
    patch_grid: int = 4


def load_grayscale(path: str, size: int) -> np.ndarray:
    img = Image.open(path).convert("L")
    img = img.resize((size, size), Image.Resampling.BICUBIC)
    return np.asarray(img, dtype=np.float32)


def apply_jpeg_compression(image: np.ndarray, quality: int) -> np.ndarray:
    pil_img = Image.fromarray(np.clip(image, 0, 255).astype(np.uint8))
    from io import BytesIO

    buffer = BytesIO()
    pil_img.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return np.asarray(Image.open(buffer).convert("L"), dtype=np.float32)


def apply_resize_attack(image: np.ndarray, scale: float) -> np.ndarray:
    h, w = image.shape
    small_w = max(8, int(round(w * scale)))
    small_h = max(8, int(round(h * scale)))
    pil_img = Image.fromarray(np.clip(image, 0, 255).astype(np.uint8))
    down = pil_img.resize((small_w, small_h), Image.Resampling.BICUBIC)
    up = down.resize((w, h), Image.Resampling.BICUBIC)
    return np.asarray(up, dtype=np.float32)


def apply_gaussian_noise(image: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    noisy = image + rng.normal(0.0, sigma, size=image.shape)
    return np.clip(noisy, 0.0, 255.0).astype(np.float32)


def _radial_profile(mag: np.ndarray, bins: int) -> np.ndarray:
    h, w = mag.shape
    cy, cx = h // 2, w // 2
    y, x = np.indices((h, w))
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    r_norm = r / (r.max() + 1e-8)
    edges = np.linspace(0.0, 1.0, bins + 1)
    out = np.zeros(bins, dtype=np.float32)
    for i in range(bins):
        mask = (r_norm >= edges[i]) & (r_norm < edges[i + 1])
        out[i] = float(mag[mask].mean()) if np.any(mask) else 0.0
    return out


def _angular_profile(mag: np.ndarray, bins: int) -> np.ndarray:
    h, w = mag.shape
    cy, cx = h // 2, w // 2
    y, x = np.indices((h, w))
    angles = np.arctan2(y - cy, x - cx)
    edges = np.linspace(-np.pi, np.pi, bins + 1)
    out = np.zeros(bins, dtype=np.float32)
    for i in range(bins):
        mask = (angles >= edges[i]) & (angles < edges[i + 1])
        out[i] = float(mag[mask].mean()) if np.any(mask) else 0.0
    return out


def _band_energy(radial_psd: np.ndarray) -> np.ndarray:
    n = len(radial_psd)
    thirds = [0, n // 3, 2 * n // 3, n]
    vals = []
    total = radial_psd.sum() + 1e-8
    for i in range(3):
        vals.append(float(radial_psd[thirds[i] : thirds[i + 1]].sum() / total))
    return np.asarray(vals, dtype=np.float32)


def _spectral_slope(radial_psd: np.ndarray) -> np.ndarray:
    x = np.arange(1, len(radial_psd) + 1, dtype=np.float32)
    y = np.maximum(radial_psd, 1e-8)
    lx, ly = np.log(x), np.log(y)
    slope, intercept = np.polyfit(lx, ly, 1)
    return np.asarray([slope, intercept], dtype=np.float32)


def _high_freq_residual_stats(mag: np.ndarray, threshold: float = 0.6) -> np.ndarray:
    h, w = mag.shape
    cy, cx = h // 2, w // 2
    y, x = np.indices((h, w))
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    r_norm = r / (r.max() + 1e-8)
    hi = mag[r_norm >= threshold]
    if hi.size == 0:
        return np.zeros(4, dtype=np.float32)
    return np.asarray(
        [
            float(np.mean(hi)),
            float(np.std(hi)),
            float(skew(hi, bias=False)),
            float(kurtosis(hi, fisher=True, bias=False)),
        ],
        dtype=np.float32,
    )


def _patch_high_freq_variation(image: np.ndarray, patch_grid: int = 4) -> np.ndarray:
    h, w = image.shape
    ph, pw = h // patch_grid, w // patch_grid
    values: List[float] = []
    for gy in range(patch_grid):
        for gx in range(patch_grid):
            patch = image[gy * ph : (gy + 1) * ph, gx * pw : (gx + 1) * pw]
            patch = patch - np.mean(patch)
            f = fftshift(fft2(patch))
            mag = np.log1p(np.abs(f) ** 2)
            rh = _high_freq_residual_stats(mag, threshold=0.7)[0]
            values.append(float(rh))
    arr = np.asarray(values, dtype=np.float32)
    return np.asarray([arr.mean(), arr.std(), arr.max(), arr.min()], dtype=np.float32)


def _dct_radial_profile(image: np.ndarray, bins: int) -> np.ndarray:
    dct = np.abs(dctn(image, norm="ortho"))
    dct = np.log1p(dct)
    return _radial_profile(dct, bins=bins)


def extract_feature_vector(
    image: np.ndarray,
    cfg: FeatureConfig,
) -> np.ndarray:
    image = image.astype(np.float32)
    image = image - np.mean(image)
    image = image * np.outer(np.hanning(image.shape[0]), np.hanning(image.shape[1]))

    spectrum = fftshift(fft2(image))
    power = np.log1p(np.abs(spectrum) ** 2)

    radial = _radial_profile(power, cfg.radial_bins)
    angular = _angular_profile(power, cfg.angular_bins)
    bands = _band_energy(radial)
    slope = _spectral_slope(radial)
    high_stats = _high_freq_residual_stats(power)
    patch_stats = _patch_high_freq_variation(image, cfg.patch_grid)
    dct_radial = _dct_radial_profile(image, bins=max(16, cfg.radial_bins // 2))

    feats = np.concatenate([radial, angular, bands, slope, high_stats, patch_stats, dct_radial]).astype(
        np.float32
    )
    return feats


def make_feature_names(cfg: FeatureConfig) -> List[str]:
    names: List[str] = []
    names += [f"fft_radial_{i}" for i in range(cfg.radial_bins)]
    names += [f"fft_angular_{i}" for i in range(cfg.angular_bins)]
    names += ["band_low", "band_mid", "band_high"]
    names += ["spectral_slope", "spectral_intercept"]
    names += ["hf_mean", "hf_std", "hf_skew", "hf_kurtosis"]
    names += ["patch_hf_mean", "patch_hf_std", "patch_hf_max", "patch_hf_min"]
    names += [f"dct_radial_{i}" for i in range(max(16, cfg.radial_bins // 2))]
    return names

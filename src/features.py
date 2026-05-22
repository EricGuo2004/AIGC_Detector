from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import Dict, List

import numpy as np
from PIL import Image
from scipy.fft import fft2, fftshift
from scipy.fftpack import dctn
from scipy.ndimage import laplace, median_filter, uniform_filter
from scipy.stats import kurtosis, skew


@dataclass
class FeatureConfig:
    image_size: int = 256
    radial_bins: int = 64
    angular_bins: int = 18
    patch_grid: int = 4
    feature_profile: str = "baseline"


FEATURE_PROFILE_CHOICES = (
    "baseline",
    "enhanced",
    "color_freq",
    "multiscale_freq",
    "block_dct",
    "residual_freq",
    "fusion_freq",
)

TRAIN_AUGMENTATION_CHOICES = ("none", "mild_freq", "robust_freq")

FEATURE_SET_CHOICES = (
    "all",
    "fft_radial",
    "fft_angular",
    "band_slope",
    "high_freq_stats",
    "patch_high_freq",
    "high_freq_all",
    "dct_radial",
    "no_dct",
)


def load_grayscale(path: str, size: int) -> np.ndarray:
    img = Image.open(path).convert("L")
    img = img.resize((size, size), Image.Resampling.BICUBIC)
    return np.asarray(img, dtype=np.float32)


def load_rgb(path: str, size: int) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    img = img.resize((size, size), Image.Resampling.BICUBIC)
    return np.asarray(img, dtype=np.float32)


def load_feature_image(path: str, cfg: FeatureConfig) -> np.ndarray:
    if cfg.feature_profile in {"color_freq", "fusion_freq"}:
        return load_rgb(path, cfg.image_size)
    return load_grayscale(path, cfg.image_size)


def apply_jpeg_compression(image: np.ndarray, quality: int) -> np.ndarray:
    pil_img = Image.fromarray(np.clip(image, 0, 255).astype(np.uint8))
    buffer = BytesIO()
    pil_img.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return np.asarray(Image.open(buffer).convert("L"), dtype=np.float32)


def apply_resize_attack(image: np.ndarray, scale: float) -> np.ndarray:
    h, w = image.shape[:2]
    small_w = max(8, int(round(w * scale)))
    small_h = max(8, int(round(h * scale)))
    pil_img = Image.fromarray(np.clip(image, 0, 255).astype(np.uint8))
    down = pil_img.resize((small_w, small_h), Image.Resampling.BICUBIC)
    up = down.resize((w, h), Image.Resampling.BICUBIC)
    return np.asarray(up, dtype=np.float32)


def apply_gaussian_noise(image: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    noisy = image + rng.normal(0.0, sigma, size=image.shape)
    return np.clip(noisy, 0.0, 255.0).astype(np.float32)


def _as_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image.astype(np.float32)
    rgb = image[..., :3].astype(np.float32)
    return (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]).astype(np.float32)


def _as_rgb(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3 and image.shape[2] >= 3:
        return image[..., :3].astype(np.float32)
    gray = image.astype(np.float32)
    return np.stack([gray, gray, gray], axis=-1)


def _resize_array(image: np.ndarray, size: int) -> np.ndarray:
    mode = "RGB" if image.ndim == 3 else "L"
    pil_img = Image.fromarray(np.clip(image, 0, 255).astype(np.uint8), mode=mode)
    pil_img = pil_img.resize((size, size), Image.Resampling.BICUBIC)
    return np.asarray(pil_img, dtype=np.float32)


def _load_pil(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def _pil_to_array(pil_img: Image.Image, size: int, color: bool) -> np.ndarray:
    mode = "RGB" if color else "L"
    return np.asarray(pil_img.convert(mode).resize((size, size), Image.Resampling.BICUBIC), dtype=np.float32)


def _jpeg_pil(pil_img: Image.Image, quality: int) -> Image.Image:
    buffer = BytesIO()
    pil_img.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def _resize_attack_pil(pil_img: Image.Image, scale: float) -> Image.Image:
    w, h = pil_img.size
    small = (max(8, int(round(w * scale))), max(8, int(round(h * scale))))
    down = pil_img.resize(small, Image.Resampling.BICUBIC)
    return down.resize((w, h), Image.Resampling.BICUBIC)


def _noise_pil(pil_img: Image.Image, sigma: float, rng: np.random.Generator) -> Image.Image:
    arr = np.asarray(pil_img.convert("RGB"), dtype=np.float32)
    noisy = arr + rng.normal(0.0, sigma, size=arr.shape)
    return Image.fromarray(np.clip(noisy, 0, 255).astype(np.uint8), mode="RGB")


def augmented_pil_variants(
    pil_img: Image.Image,
    train_augmentation: str,
    seed: int = 42,
) -> List[Image.Image]:
    if train_augmentation not in TRAIN_AUGMENTATION_CHOICES:
        raise ValueError(
            f"Unknown train augmentation '{train_augmentation}'. "
            f"Expected one of: {', '.join(TRAIN_AUGMENTATION_CHOICES)}"
        )
    if train_augmentation == "none":
        return [pil_img]

    rng = np.random.default_rng(seed)
    if train_augmentation == "mild_freq":
        return [
            pil_img,
            _jpeg_pil(pil_img, 90),
            _resize_attack_pil(pil_img, 0.75),
            _noise_pil(pil_img, 2.0, rng),
        ]

    return [
        pil_img,
        _jpeg_pil(pil_img, 90),
        _resize_attack_pil(pil_img, 0.75),
        _noise_pil(pil_img, 2.0, rng),
        _jpeg_pil(pil_img, 70),
        _resize_attack_pil(pil_img, 0.50),
        _noise_pil(pil_img, 5.0, rng),
    ]


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


def _multi_band_energy(radial_psd: np.ndarray, bands: int = 5) -> np.ndarray:
    edges = np.linspace(0, len(radial_psd), bands + 1, dtype=int)
    total = radial_psd.sum() + 1e-8
    vals = []
    for i in range(bands):
        vals.append(float(radial_psd[edges[i] : edges[i + 1]].sum() / total))
    return np.asarray(vals, dtype=np.float32)


def _spectral_shape_stats(radial_psd: np.ndarray) -> np.ndarray:
    weights = np.maximum(radial_psd.astype(np.float64), 0.0)
    probs = weights / (weights.sum() + 1e-12)
    pos = np.linspace(0.0, 1.0, len(probs), dtype=np.float64)
    centroid = float(np.sum(pos * probs))
    spread = float(np.sqrt(np.sum(((pos - centroid) ** 2) * probs)))
    entropy = float(-np.sum(probs * np.log(probs + 1e-12)) / np.log(len(probs)))
    return np.asarray([centroid, spread, entropy], dtype=np.float32)


def _radial_diff_stats(radial_psd: np.ndarray) -> np.ndarray:
    diff = np.diff(radial_psd.astype(np.float64))
    if diff.size == 0:
        return np.zeros(4, dtype=np.float32)
    return np.asarray([diff.mean(), diff.std(), diff.min(), diff.max()], dtype=np.float32)


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


def _high_freq_annulus_quantiles(mag: np.ndarray, threshold: float = 0.6) -> np.ndarray:
    h, w = mag.shape
    cy, cx = h // 2, w // 2
    y, x = np.indices((h, w))
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    r_norm = r / (r.max() + 1e-8)
    hi = mag[r_norm >= threshold]
    if hi.size == 0:
        return np.zeros(5, dtype=np.float32)
    return np.asarray(np.quantile(hi, [0.10, 0.25, 0.50, 0.75, 0.90]), dtype=np.float32)


def _tail_energy_ratios(radial_psd: np.ndarray) -> np.ndarray:
    n = len(radial_psd)
    low = float(radial_psd[: max(1, n // 3)].sum()) + 1e-8
    mid = float(radial_psd[n // 3 : max(n // 3 + 1, 2 * n // 3)].sum()) + 1e-8
    high = float(radial_psd[2 * n // 3 :].sum()) + 1e-8
    tail = float(radial_psd[max(0, int(0.8 * n)) :].sum()) + 1e-8
    return np.asarray([tail / low, tail / mid, high / (low + mid)], dtype=np.float32)


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


def _windowed_power(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    image = image.astype(np.float32)
    image = image - np.mean(image)
    image = image * np.outer(np.hanning(image.shape[0]), np.hanning(image.shape[1]))
    spectrum = fftshift(fft2(image))
    power = np.log1p(np.abs(spectrum) ** 2)
    return image, power


def _base_frequency_features(gray: np.ndarray, cfg: FeatureConfig) -> np.ndarray:
    image, power = _windowed_power(gray)
    radial = _radial_profile(power, cfg.radial_bins)
    angular = _angular_profile(power, cfg.angular_bins)
    bands = _band_energy(radial)
    slope = _spectral_slope(radial)
    high_stats = _high_freq_residual_stats(power)
    patch_stats = _patch_high_freq_variation(image, cfg.patch_grid)
    dct_radial = _dct_radial_profile(image, bins=max(16, cfg.radial_bins // 2))
    return np.concatenate([radial, angular, bands, slope, high_stats, patch_stats, dct_radial])


def _compact_frequency_features(gray: np.ndarray, cfg: FeatureConfig) -> np.ndarray:
    image, power = _windowed_power(gray)
    radial = _radial_profile(power, cfg.radial_bins)
    bands = _band_energy(radial)
    slope = _spectral_slope(radial)
    high_stats = _high_freq_residual_stats(power)
    dct_radial = _dct_radial_profile(image, bins=max(16, cfg.radial_bins // 2))
    return np.concatenate([radial, bands, slope, high_stats, dct_radial])


def _residual_compact_frequency_features(gray: np.ndarray, cfg: FeatureConfig) -> np.ndarray:
    image, power = _windowed_power(gray)
    radial = _radial_profile(power, cfg.radial_bins)
    bands = _band_energy(radial)
    slope = _spectral_slope(radial)
    high_stats = _high_freq_residual_stats(power)
    return np.concatenate([radial, bands, slope, high_stats])


def _enhanced_frequency_features(radial: np.ndarray, power: np.ndarray) -> np.ndarray:
    feats = np.concatenate(
        [
            _multi_band_energy(radial, bands=5),
            _spectral_shape_stats(radial),
            _radial_diff_stats(radial),
            _high_freq_annulus_quantiles(power),
            _tail_energy_ratios(radial),
        ]
    )
    return np.nan_to_num(feats.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def _enhanced_from_gray(gray: np.ndarray, cfg: FeatureConfig) -> np.ndarray:
    _, power = _windowed_power(gray)
    radial = _radial_profile(power, cfg.radial_bins)
    return _enhanced_frequency_features(radial, power)


def _rgb_ycbcr_channels(image: np.ndarray) -> Dict[str, np.ndarray]:
    rgb = _as_rgb(image)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = 128.0 - 0.168736 * r - 0.331264 * g + 0.5 * b
    cr = 128.0 + 0.5 * r - 0.418688 * g - 0.081312 * b
    return {
        "rgb_r": r,
        "rgb_g": g,
        "rgb_b": b,
        "ycbcr_y": y,
        "ycbcr_cb": cb,
        "ycbcr_cr": cr,
    }


def _color_frequency_features(image: np.ndarray, cfg: FeatureConfig) -> np.ndarray:
    parts = []
    for channel in _rgb_ycbcr_channels(image).values():
        parts.append(_compact_frequency_features(channel.astype(np.float32), cfg))
    return np.concatenate(parts)


def _multiscale_sizes(cfg: FeatureConfig) -> tuple[int, int, int]:
    return (max(64, cfg.image_size // 2), cfg.image_size, max(cfg.image_size + 1, int(round(cfg.image_size * 1.5))))


def _multiscale_frequency_features(image: np.ndarray, cfg: FeatureConfig) -> np.ndarray:
    gray = _as_gray(image)
    parts = []
    for size in _multiscale_sizes(cfg):
        resized = _resize_array(gray, size)
        parts.append(_compact_frequency_features(resized, cfg))
    return np.concatenate(parts)


def _zigzag_indices(n: int = 8) -> List[tuple[int, int]]:
    out: List[tuple[int, int]] = []
    for s in range(2 * n - 1):
        if s % 2 == 0:
            xs = range(min(s, n - 1), max(-1, s - n), -1)
        else:
            xs = range(max(0, s - n + 1), min(s, n - 1) + 1)
        for i in xs:
            j = s - i
            if 0 <= i < n and 0 <= j < n:
                out.append((i, j))
    return out


def _block_dct_features(image: np.ndarray, cfg: FeatureConfig) -> np.ndarray:
    gray = _as_gray(_resize_array(_as_gray(image), cfg.image_size))
    h, w = gray.shape
    block = 8
    h = h - (h % block)
    w = w - (w % block)
    gray = gray[:h, :w]
    coeffs = []
    low_mid_high = []
    dc_vals = []
    ac_energy = []
    zigzag = [pos for pos in _zigzag_indices(block) if pos != (0, 0)][:12]
    for y in range(0, h, block):
        for x in range(0, w, block):
            patch = gray[y : y + block, x : x + block]
            dct = np.abs(dctn(patch - patch.mean(), norm="ortho"))
            dc_vals.append(float(dct[0, 0]))
            ac = dct.copy()
            ac[0, 0] = 0.0
            ac_energy.append(float(np.mean(ac**2)))
            coeffs.append([float(dct[i, j]) for i, j in zigzag])
            low = float(dct[:3, :3].sum() - dct[0, 0])
            mid = float(dct[:5, :5].sum() - dct[:3, :3].sum())
            high = float(dct.sum() - dct[:5, :5].sum())
            total = low + mid + high + 1e-8
            low_mid_high.append([low / total, mid / total, high / total])

    coeff_arr = np.asarray(coeffs, dtype=np.float32)
    bands = np.asarray(low_mid_high, dtype=np.float32)
    dc = np.asarray(dc_vals, dtype=np.float32)
    ac = np.asarray(ac_energy, dtype=np.float32)
    v_boundary = np.abs(np.diff(gray[:, 7::8], axis=1)).ravel() if gray.shape[1] >= 16 else np.asarray([0.0])
    h_boundary = np.abs(np.diff(gray[7::8, :], axis=0)).ravel() if gray.shape[0] >= 16 else np.asarray([0.0])
    v_all = np.abs(np.diff(gray, axis=1)).ravel()
    h_all = np.abs(np.diff(gray, axis=0)).ravel()
    aggregate = np.asarray(
        [
            float(dc.mean()),
            float(dc.std()),
            float(ac.mean()),
            float(ac.std()),
            float(skew(ac, bias=False)) if ac.size > 2 else 0.0,
            float(kurtosis(ac, fisher=True, bias=False)) if ac.size > 3 else 0.0,
            float(v_boundary.mean()),
            float(v_boundary.std()),
            float(h_boundary.mean()),
            float(h_boundary.std()),
            float(v_boundary.mean() / (v_all.mean() + 1e-8)),
            float(h_boundary.mean() / (h_all.mean() + 1e-8)),
        ],
        dtype=np.float32,
    )
    return np.concatenate([coeff_arr.mean(axis=0), coeff_arr.std(axis=0), bands.mean(axis=0), bands.std(axis=0), aggregate])


def _residual_images(image: np.ndarray, cfg: FeatureConfig) -> Dict[str, np.ndarray]:
    gray = _as_gray(_resize_array(_as_gray(image), cfg.image_size))
    median_resid = gray - median_filter(gray, size=3)
    smooth_resid = gray - uniform_filter(gray, size=5)
    laplace_resid = laplace(gray)
    return {
        "median": median_resid.astype(np.float32),
        "smooth": smooth_resid.astype(np.float32),
        "laplace": laplace_resid.astype(np.float32),
    }


def _residual_frequency_features(image: np.ndarray, cfg: FeatureConfig) -> np.ndarray:
    return np.concatenate([_residual_compact_frequency_features(resid, cfg) for resid in _residual_images(image, cfg).values()])


def _base_feature_names(cfg: FeatureConfig) -> List[str]:
    names: List[str] = []
    names += [f"fft_radial_{i}" for i in range(cfg.radial_bins)]
    names += [f"fft_angular_{i}" for i in range(cfg.angular_bins)]
    names += ["band_low", "band_mid", "band_high"]
    names += ["spectral_slope", "spectral_intercept"]
    names += ["hf_mean", "hf_std", "hf_skew", "hf_kurtosis"]
    names += ["patch_hf_mean", "patch_hf_std", "patch_hf_max", "patch_hf_min"]
    names += [f"dct_radial_{i}" for i in range(max(16, cfg.radial_bins // 2))]
    return names


def _compact_feature_names(prefix: str, cfg: FeatureConfig) -> List[str]:
    names: List[str] = []
    names += [f"{prefix}_fft_radial_{i}" for i in range(cfg.radial_bins)]
    names += [f"{prefix}_band_low", f"{prefix}_band_mid", f"{prefix}_band_high"]
    names += [f"{prefix}_spectral_slope", f"{prefix}_spectral_intercept"]
    names += [f"{prefix}_hf_mean", f"{prefix}_hf_std", f"{prefix}_hf_skew", f"{prefix}_hf_kurtosis"]
    names += [f"{prefix}_dct_radial_{i}" for i in range(max(16, cfg.radial_bins // 2))]
    return names


def _residual_compact_feature_names(prefix: str, cfg: FeatureConfig) -> List[str]:
    names: List[str] = []
    names += [f"{prefix}_fft_radial_{i}" for i in range(cfg.radial_bins)]
    names += [f"{prefix}_band_low", f"{prefix}_band_mid", f"{prefix}_band_high"]
    names += [f"{prefix}_spectral_slope", f"{prefix}_spectral_intercept"]
    names += [f"{prefix}_hf_mean", f"{prefix}_hf_std", f"{prefix}_hf_skew", f"{prefix}_hf_kurtosis"]
    return names


def _enhanced_feature_names() -> List[str]:
    names = [f"five_band_energy_{i}" for i in range(5)]
    names += ["spectral_centroid", "spectral_spread", "spectral_entropy"]
    names += ["radial_diff_mean", "radial_diff_std", "radial_diff_min", "radial_diff_max"]
    names += [f"hf_annulus_q{q}" for q in ("10", "25", "50", "75", "90")]
    names += ["tail_to_low_energy", "tail_to_mid_energy", "high_to_low_mid_energy"]
    return names


def _color_feature_names(cfg: FeatureConfig) -> List[str]:
    names: List[str] = []
    for channel in ("rgb_r", "rgb_g", "rgb_b", "ycbcr_y", "ycbcr_cb", "ycbcr_cr"):
        names += _compact_feature_names(f"color_{channel}", cfg)
    return names


def _multiscale_feature_names(cfg: FeatureConfig) -> List[str]:
    names: List[str] = []
    for size in _multiscale_sizes(cfg):
        names += _compact_feature_names(f"scale_{size}", cfg)
    return names


def _block_dct_feature_names() -> List[str]:
    names = [f"block_dct_zigzag_mean_{i}" for i in range(12)]
    names += [f"block_dct_zigzag_std_{i}" for i in range(12)]
    names += ["block_dct_band_low_mean", "block_dct_band_mid_mean", "block_dct_band_high_mean"]
    names += ["block_dct_band_low_std", "block_dct_band_mid_std", "block_dct_band_high_std"]
    names += [
        "block_dct_dc_mean",
        "block_dct_dc_std",
        "block_dct_ac_energy_mean",
        "block_dct_ac_energy_std",
        "block_dct_ac_energy_skew",
        "block_dct_ac_energy_kurtosis",
        "block_boundary_vertical_mean",
        "block_boundary_vertical_std",
        "block_boundary_horizontal_mean",
        "block_boundary_horizontal_std",
        "block_boundary_vertical_ratio",
        "block_boundary_horizontal_ratio",
    ]
    return names


def _residual_feature_names(cfg: FeatureConfig) -> List[str]:
    names: List[str] = []
    for resid in ("median", "smooth", "laplace"):
        names += _residual_compact_feature_names(f"residual_{resid}", cfg)
    return names


def extract_feature_vector(image: np.ndarray, cfg: FeatureConfig) -> np.ndarray:
    if cfg.feature_profile not in FEATURE_PROFILE_CHOICES:
        raise ValueError(f"Unknown feature profile '{cfg.feature_profile}'. Expected one of: {', '.join(FEATURE_PROFILE_CHOICES)}")

    gray = _as_gray(image)
    parts = [_base_frequency_features(gray, cfg)]
    if cfg.feature_profile == "enhanced":
        parts.append(_enhanced_from_gray(gray, cfg))
    elif cfg.feature_profile == "color_freq":
        parts.append(_color_frequency_features(image, cfg))
    elif cfg.feature_profile == "multiscale_freq":
        parts.append(_multiscale_frequency_features(image, cfg))
    elif cfg.feature_profile == "block_dct":
        parts.append(_block_dct_features(image, cfg))
    elif cfg.feature_profile == "residual_freq":
        parts.append(_residual_frequency_features(image, cfg))
    elif cfg.feature_profile == "fusion_freq":
        parts.extend(
            [
                _color_frequency_features(image, cfg),
                _multiscale_frequency_features(image, cfg),
                _block_dct_features(image, cfg),
                _residual_frequency_features(image, cfg),
            ]
        )

    feats = np.concatenate(parts).astype(np.float32)
    return np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)


def extract_feature_vector_from_path(path: str, cfg: FeatureConfig) -> np.ndarray:
    color = cfg.feature_profile in {"color_freq", "fusion_freq"}
    image = _pil_to_array(_load_pil(path), cfg.image_size, color=color)
    return extract_feature_vector(image, cfg)


def extract_feature_variants_from_path(
    path: str,
    cfg: FeatureConfig,
    train_augmentation: str = "none",
    seed: int = 42,
) -> List[np.ndarray]:
    color = cfg.feature_profile in {"color_freq", "fusion_freq"}
    pil_img = _load_pil(path)
    features = []
    for variant in augmented_pil_variants(pil_img, train_augmentation, seed=seed):
        image = _pil_to_array(variant, cfg.image_size, color=color)
        features.append(extract_feature_vector(image, cfg))
    return features


def make_feature_names(cfg: FeatureConfig) -> List[str]:
    if cfg.feature_profile not in FEATURE_PROFILE_CHOICES:
        raise ValueError(f"Unknown feature profile '{cfg.feature_profile}'. Expected one of: {', '.join(FEATURE_PROFILE_CHOICES)}")

    names = _base_feature_names(cfg)
    if cfg.feature_profile == "enhanced":
        names += _enhanced_feature_names()
    elif cfg.feature_profile == "color_freq":
        names += _color_feature_names(cfg)
    elif cfg.feature_profile == "multiscale_freq":
        names += _multiscale_feature_names(cfg)
    elif cfg.feature_profile == "block_dct":
        names += _block_dct_feature_names()
    elif cfg.feature_profile == "residual_freq":
        names += _residual_feature_names(cfg)
    elif cfg.feature_profile == "fusion_freq":
        names += _color_feature_names(cfg)
        names += _multiscale_feature_names(cfg)
        names += _block_dct_feature_names()
        names += _residual_feature_names(cfg)
    return names


def feature_group_indices(cfg: FeatureConfig) -> Dict[str, np.ndarray]:
    dct_bins = max(16, cfg.radial_bins // 2)
    start = 0
    groups: Dict[str, np.ndarray] = {}

    groups["fft_radial"] = np.arange(start, start + cfg.radial_bins)
    start += cfg.radial_bins

    groups["fft_angular"] = np.arange(start, start + cfg.angular_bins)
    start += cfg.angular_bins

    band = np.arange(start, start + 3)
    start += 3
    slope = np.arange(start, start + 2)
    start += 2
    groups["band_slope"] = np.concatenate([band, slope])

    groups["high_freq_stats"] = np.arange(start, start + 4)
    start += 4

    groups["patch_high_freq"] = np.arange(start, start + 4)
    start += 4

    groups["dct_radial"] = np.arange(start, start + dct_bins)
    groups["high_freq_all"] = np.concatenate([groups["high_freq_stats"], groups["patch_high_freq"]])

    all_indices = np.arange(len(make_feature_names(cfg)))
    groups["all"] = all_indices
    groups["no_dct"] = np.setdiff1d(all_indices, groups["dct_radial"], assume_unique=True)
    return groups


def select_feature_indices(cfg: FeatureConfig, feature_set: str) -> np.ndarray:
    groups = feature_group_indices(cfg)
    if feature_set not in groups:
        raise ValueError(f"Unknown feature set '{feature_set}'. Expected one of: {', '.join(FEATURE_SET_CHOICES)}")
    return groups[feature_set]


def select_feature_array(X: np.ndarray, cfg: FeatureConfig, feature_set: str) -> np.ndarray:
    idx = select_feature_indices(cfg, feature_set)
    return X[:, idx]


def select_feature_columns(
    X: np.ndarray,
    feature_names: List[str],
    cfg: FeatureConfig,
    feature_set: str,
) -> tuple[np.ndarray, List[str]]:
    idx = select_feature_indices(cfg, feature_set)
    return X[:, idx], [feature_names[i] for i in idx]

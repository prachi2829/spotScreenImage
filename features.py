
import cv2
import numpy as np


# LBP settings
LBP_RADIUS = 1
LBP_POINTS = 8 * LBP_RADIUS



FEATURE_NAMES = [
    "fft_high_freq_ratio",
    "fft_peakiness",
    "moire_energy",

    "laplacian_variance",
    "laplacian_variance_normalized",

    "entropy",

    "saturation_mean",
    "saturation_std",

    "highlight_ratio",
    "largest_highlight_blob",

    "channel_correlation",

    "edge_density",

    "noise_std",
]

# Add names for the LBP histogram bins
for i in range(LBP_POINTS + 2):
    FEATURE_NAMES.append(f"lbp_bin_{i}")


def load_image(image_path, max_side=640):

    img = cv2.imread(image_path)

    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")

    h, w = img.shape[:2]

    scale = min(max_side / max(h, w), 1.0)

    if scale < 1:

        img = cv2.resize(
            img,
            (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_AREA,
        )

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return img, gray


def fft_features(gray):
  
    # Convert to float
    img = gray.astype(np.float32) / 255.0

    # Apply Hann window to reduce FFT edge artifacts
    window = np.outer(
        np.hanning(img.shape[0]),
        np.hanning(img.shape[1])
    )

    img = img * window

    # FFT
    fft = np.fft.fft2(img)
    fft = np.fft.fftshift(fft)

    magnitude = np.log1p(np.abs(fft))

    h, w = magnitude.shape
    cy, cx = h // 2, w // 2

    Y, X = np.ogrid[:h, :w]
    radius = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)

    max_radius = radius.max()

    # Frequency bands
    low = radius < 0.08 * max_radius
    mid = (radius >= 0.08 * max_radius) & (radius < 0.35 * max_radius)
    high = radius >= 0.35 * max_radius

    low_energy = magnitude[low].mean()
    mid_energy = magnitude[mid].mean()
    high_energy = magnitude[high].mean()

    # High-frequency ratio
    high_freq_ratio = (mid_energy + high_energy) / (low_energy + 1e-6)

    # Peakiness (strong periodic peaks)
    ring = magnitude[mid]

    if len(ring) == 0:
        peakiness = 1.0
    else:

        top = np.sort(ring)[-max(1, int(0.001 * len(ring))):]

        peakiness = top.mean() / (ring.mean() + 1e-6)

    return high_freq_ratio, peakiness


def moire_energy(gray):
    img = gray.astype(np.float32) / 255.0

    window = np.outer(
        np.hanning(img.shape[0]),
        np.hanning(img.shape[1])
    )

    fft = np.fft.fft2(img * window)
    fft = np.fft.fftshift(fft)

    magnitude = np.abs(fft)

    h, w = magnitude.shape

    cy, cx = h // 2, w // 2

    Y, X = np.ogrid[:h, :w]

    radius = np.sqrt(
        (X - cx) ** 2 +
        (Y - cy) ** 2
    )

    radius = radius / max(h, w)

    band = (radius > 0.12) & (radius < 0.28)

    total_energy = magnitude.sum() + 1e-6

    return magnitude[band].sum() / total_energy



def laplacian_features(gray):

    lap = cv2.Laplacian(gray, cv2.CV_64F)

    variance = lap.var()

    # Normalize by image contrast
    contrast = gray.std() + 1e-6

    normalized = variance / (contrast ** 2)

    return variance, normalized


def entropy_feature(gray):

    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()

    total = hist.sum()

    if total <= 0:
        return 0.0

    p = hist / total

    p = p[p > 0]

    return float(-np.sum(p * np.log2(p)))


def saturation_features(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    saturation = hsv[:, :, 1].astype(np.float32) / 255.0

    return (
        saturation.mean(),
        saturation.std()
    )


def highlight_features(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Very bright pixels
    mask = (gray >= 240).astype(np.uint8)

    highlight_ratio = mask.mean()

    largest_blob = 0.0

    if mask.sum() > 0:

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)

        if num_labels > 1:

            largest_blob = (
                stats[1:, cv2.CC_STAT_AREA].max()
                / mask.size
            )

    return highlight_ratio, largest_blob


def channel_correlation(img):
    small = cv2.resize(img, (256, 256),interpolation=cv2.INTER_AREA)
    

    small = small.astype(np.float32)

    b = small[:, :, 0]
    g = small[:, :, 1]
    r = small[:, :, 2]

    def highpass(channel):

        blur = cv2.GaussianBlur(channel, (0, 0), 2)

        return channel - blur

    b = highpass(b)
    g = highpass(g)
    r = highpass(r)

    bg = np.corrcoef(b.flatten(), g.flatten())[0, 1]
    gr = np.corrcoef(g.flatten(), r.flatten())[0, 1]
    br = np.corrcoef(b.flatten(), r.flatten())[0, 1]

    vals = [v for v in (bg, gr, br) if not np.isnan(v)]

    return np.mean(vals)


def edge_density(gray):

    edges = cv2.Canny(gray, 60, 120)

    density = np.sum(edges > 0) / edges.size

    return density


def noise_feature(gray):

    blur = cv2.GaussianBlur(gray, (3, 3), 0)

    noise = gray.astype(np.float32) - blur.astype(np.float32)

    return noise.std()


def lbp_features(gray):

    g = gray.astype(np.int16)

    # 8 neighbors in circular (clockwise) order around the center,
    # each shifted so it aligns with the center pixel's coordinates.
    top_left     = g[0:-2, 0:-2]
    top          = g[0:-2, 1:-1]
    top_right    = g[0:-2, 2:]
    right        = g[1:-1, 2:]
    bottom_right = g[2:,   2:]
    bottom       = g[2:,   1:-1]
    bottom_left  = g[2:,   0:-2]
    left         = g[1:-1, 0:-2]

    center = g[1:-1, 1:-1]

    neighbors = [
        top_left, top, top_right, right,
        bottom_right, bottom, bottom_left, left,
    ]

    bits = [(n >= center).astype(np.uint8) for n in neighbors]
    stacked = np.stack(bits, axis=-1)  # (H-2, W-2, 8)

    # circular transitions: compare each bit to the next one, wrapping around
    next_stacked = np.roll(stacked, shift=-1, axis=-1)
    transitions = np.abs(stacked.astype(np.int16) - next_stacked.astype(np.int16)).sum(axis=-1)

    ones_count = stacked.sum(axis=-1).astype(np.int16)

    is_uniform = transitions <= 2

    codes = np.where(is_uniform, ones_count, LBP_POINTS + 1)

    hist = np.bincount(codes.ravel(), minlength=LBP_POINTS + 2).astype(np.float64)
    hist = hist / (hist.sum() + 1e-9)

    return hist.tolist()

def extract_features(image_path):

    img, gray = load_image(image_path)

    features = []

    # FFT
    fft_ratio, fft_peak = fft_features(gray)
    features.extend([fft_ratio, fft_peak])

    # Moire
    features.append(moire_energy(gray))

    # Laplacian
    lap_var, lap_norm = laplacian_features(gray)
    features.extend([lap_var, lap_norm])

    # Entropy
    features.append(entropy_feature(gray))

    # Saturation
    sat_mean, sat_std = saturation_features(img)
    features.extend([sat_mean, sat_std])

    # Highlights
    highlight_ratio, largest_blob = highlight_features(img)
    features.extend([highlight_ratio, largest_blob])

    # Channel correlation
    features.append(channel_correlation(img))

    # Edge density
    features.append(edge_density(gray))

    # Noise
    features.append(noise_feature(gray))

    # LBP histogram
    features.extend(lbp_features(gray))

    features = np.array(features, dtype=np.float32)

    assert len(features) == len(FEATURE_NAMES)

    return features


if __name__ == "__main__":

    image = "dataset/real/real_001.jpg"

    features = extract_features(image)

    print("Feature vector length:", len(features))

    print()

    for name, value in zip(FEATURE_NAMES, features):
        print(f"{name:35s}: {value:.6f}")

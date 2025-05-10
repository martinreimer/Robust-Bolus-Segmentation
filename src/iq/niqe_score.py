import numpy as np
import scipy.io
import scipy.ndimage
import scipy.special
import scipy.linalg
import math
import os
import sys
from PIL import Image

# === PRE-COMPUTATION ===
gamma_range = np.arange(0.2, 10, 0.001)
a = scipy.special.gamma(2.0 / gamma_range)
a *= a
b = scipy.special.gamma(1.0 / gamma_range)
c = scipy.special.gamma(3.0 / gamma_range)
prec_gammas = a / (b * c)

# === CORE FUNCTIONS ===
def aggd_features(imdata):
    imdata = imdata.flatten()
    imdata2 = imdata * imdata
    left_data = imdata2[imdata < 0]
    right_data = imdata2[imdata >= 0]
    left_mean_sqrt = np.sqrt(np.mean(left_data)) if len(left_data) > 0 else 0
    right_mean_sqrt = np.sqrt(np.mean(right_data)) if len(right_data) > 0 else 0

    gamma_hat = left_mean_sqrt / right_mean_sqrt if right_mean_sqrt != 0 else np.inf
    r_hat = (np.mean(np.abs(imdata)) ** 2) / (np.mean(imdata2)) if np.mean(imdata2) != 0 else np.inf
    rhat_norm = r_hat * (((gamma_hat**3 + 1) * (gamma_hat + 1)) / ((gamma_hat**2 + 1) ** 2))

    pos = np.argmin((prec_gammas - rhat_norm) ** 2)
    alpha = gamma_range[pos]

    gam1 = scipy.special.gamma(1.0 / alpha)
    gam2 = scipy.special.gamma(2.0 / alpha)
    gam3 = scipy.special.gamma(3.0 / alpha)

    aggdratio = np.sqrt(gam1) / np.sqrt(gam3)
    bl = aggdratio * left_mean_sqrt
    br = aggdratio * right_mean_sqrt
    N = (br - bl) * (gam2 / gam1)

    return (alpha, N, bl, br, left_mean_sqrt, right_mean_sqrt)

def paired_product(im):
    return tuple(np.roll(im, shift, axis) * im for shift, axis in [(1, 1), (1, 0), (1, 0), (1, 0)])

def compute_image_mscn_transform(image, C=1):
    avg_window = np.array([np.exp(-0.5 * (i**2) / ((7.0/6.0)**2)) for i in range(-3, 4)])
    avg_window /= np.sum(avg_window)

    mu = scipy.ndimage.correlate1d(image, avg_window, 0, mode='constant')
    mu = scipy.ndimage.correlate1d(mu, avg_window, 1, mode='constant')
    var = scipy.ndimage.correlate1d(image**2, avg_window, 0, mode='constant')
    var = scipy.ndimage.correlate1d(var, avg_window, 1, mode='constant')
    var = np.sqrt(np.abs(var - mu**2))

    return (image - mu) / (var + C)

def _niqe_extract_subband_feats(mscn):
    alpha_m, N, bl, br, *_ = aggd_features(mscn)
    pps = paired_product(mscn)
    feats = [alpha_m, (bl + br) / 2.0]
    for p in pps:
        alpha, N, bl, br, *_ = aggd_features(p)
        feats.extend([alpha, N, bl, br])
    return np.array(feats)

def extract_on_patches(img, patch_size):
    h, w = img.shape
    patches = [img[i:i+patch_size, j:j+patch_size]
               for i in range(0, h - patch_size + 1, patch_size)
               for j in range(0, w - patch_size + 1, patch_size)]
    return np.array([_niqe_extract_subband_feats(p) for p in patches])

def get_patches_test_features(img, patch_size=96):
    h, w = img.shape
    img = img[:h - h % patch_size, :w - w % patch_size]
    img_small = scipy.ndimage.zoom(img, 0.5, order=3)
    mscn1 = compute_image_mscn_transform(img)
    mscn2 = compute_image_mscn_transform(img_small)
    feats1 = extract_on_patches(mscn1, patch_size)
    feats2 = extract_on_patches(mscn2, patch_size // 2)
    return np.hstack((feats1, feats2))

def niqe(gray_img_np, model_mat_path='data/niqe_image_params.mat'):
    if gray_img_np.ndim != 2:
        raise ValueError("Input image must be grayscale.")
    if gray_img_np.shape[0] < 192 or gray_img_np.shape[1] < 192:
        raise ValueError("Image must be at least 192x192 in size for NIQE computation.")

    features = get_patches_test_features(gray_img_np)
    model = scipy.io.loadmat(model_mat_path)
    pop_mu = model['pop_mu'].flatten()
    pop_cov = model['pop_cov']
    sample_mu = np.mean(features, axis=0)
    sample_cov = np.cov(features.T)
    X = sample_mu - pop_mu
    cov = (sample_cov + pop_cov) / 2.0
    pinv = scipy.linalg.pinv(cov)
    return np.sqrt(np.dot(np.dot(X, pinv), X))


# === MAIN SCRIPT ===
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python niqe_score.py path/to/image.png")
        sys.exit(1)

    img_path = sys.argv[1]
    if not os.path.exists(img_path):
        print(f"Error: {img_path} not found.")
        sys.exit(1)

    img = Image.open(img_path).convert("L")  # Convert to grayscale
    img_np = np.array(img).astype(np.float32)

    score = niqe(img_np)
    print(f"NIQE score for {img_path}: {score:.3f}")

    '''
    python niqe_score.py "D:\Martin\thesis\data\processed\dataset_labelbox_export_test_2504_test_final_roi_crop\test\imgs\1888.png"
    '''
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def plot_img_and_mask(img, mask):
    """
    Original function: display the image and each mask class in separate subplots.
    """
    classes = mask.max() + 1
    fig, ax = plt.subplots(1, classes + 1, figsize=(6 + 4*classes, 4))
    ax[0].set_title('Input image')
    ax[0].imshow(img)

    for i in range(classes):
        ax[i + 1].set_title(f'Mask (class {i})')
        ax[i + 1].imshow(mask == i)

    plt.xticks([]), plt.yticks([])
    plt.show()

def overlay_mask_on_image(original_image, mask, color=(255, 0, 255), alpha=0.4):
    """
    Overlays `mask` on top of `original_image` with a given RGBA color and alpha transparency.
    - original_image: PIL Image (RGB or grayscale)
    - mask: np.ndarray shape (H, W) or (H, W, 1). Non-zero pixels get overlaid.
    - color: tuple(r,g,b)
    - alpha: transparency, 0=fully transparent, 1=fully opaque

    Returns a new PIL Image with mask overlayed.
    """

    # Convert original to RGBA for alpha compositing
    orig_mode = original_image.mode
    if orig_mode != 'RGBA':
        base_img = original_image.convert('RGBA')
    else:
        base_img = original_image.copy()

    # Ensure mask is 2D
    if mask.ndim == 3 and mask.shape[-1] == 1:
        mask = mask.squeeze(-1)
    mask = mask.astype(bool)  # any non-zero is True

    # Convert base_img to NumPy for easy overlay
    overlay = np.array(base_img, dtype=np.uint8)

    # The color we paint
    overlay_color = np.array(color, dtype=np.uint8)

    # Where mask == True, blend in the color
    # overlay[y, x] = alpha * overlay_color + (1 - alpha) * overlay[y, x]
    # We'll do it channel by channel, ignoring the alpha channel for the base
    # Then we set final alpha to 255 for those pixels.

    # Indices of all mask pixels
    indices = np.where(mask)

    for c in range(3):  # RGB channels
        overlay[indices + (c,)] = (alpha * overlay_color[c] +
                                   (1 - alpha) * overlay[indices + (c,)]).astype(np.uint8)

    # For these masked pixels, set alpha channel to 255
    overlay[indices + (3,)] = 255

    # Convert back to PIL
    out_pil = Image.fromarray(overlay, 'RGBA')
    return out_pil

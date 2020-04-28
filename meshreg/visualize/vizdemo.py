import cv2
import numpy as np


def draw_contours(img, mask, color=(0, 255, 255)):
    """
    Get outline of mask and draw it on the image

    Args:
        img: image on which to draw
        mask: mask for which contours should be found
        color: color of contours to draw
    """
    contours = cv2.findContours((mask > 0).astype("uint8"), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    res_img = cv2.drawContours(img.copy(), contours[0], -1, color, 3)
    return res_img


def get_crop(mask_image, scale_ratio=1.2):
    """
    Get bounding box of square encompassing all positive
    values of the mask, with a margin according to scale_ratio

    Args:
        scale_ratio: final crop is obtained by taking the square
            centered around the center of the tight bounding box
            around the masked values, with a square size of
            scale_ratio * size of largest bounding box dimension
    """
    mask_image = mask_image.numpy()
    xs, ys = (mask_image[:, :, ::3].sum(2) > 0).nonzero()
    x_min = xs.min()
    x_max = xs.max()
    y_min = ys.min()
    y_max = ys.max()
    radius = max((x_max - x_min), (y_max - y_min)) // 2 * scale_ratio
    x_c = (x_max + x_min) / 2
    y_c = (y_max + y_min) / 2
    x_min = max(int((x_c - radius).item()), 0)
    y_min = max(int((y_c - radius).item()), 0)
    x_max = int((x_c + radius).item())
    y_max = int((y_c + radius).item())
    return x_min, y_min, x_max, y_max


def get_common_crop(crops):
    """
    Given bounding box values, get the the encompassing bounding
    box.
    """
    crops = np.array(crops)
    common_crop = np.concatenate([crops.min(0)[:2], crops.max(0)[2:]])
    return common_crop


def get_axis(axes, row_nb, col_nb, row_idx, col_idx):
    """
    Extract matplotlib axis among array of axis returned
    by subplots, handling the fact that the number of dimensions
    is affected by the number of columns and rows.
    """
    if row_nb == 1 and col_nb == 1:
        ax = axes
    elif col_nb > 1 and row_nb == 1:
        ax = axes[col_idx]
    elif col_nb == 1 and row_nb > 1:
        ax = axes[row_idx]
    else:
        ax = axes[row_idx, col_idx]
    ax.axis("off")
    return ax

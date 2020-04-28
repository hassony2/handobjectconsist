from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch


def get_verts_colors(verts, color=None):
    """
    Args:
        color: either [r, g, b] with each color in [0, 1] or None
            if None, maps xyz as color
    """
    if color is None:
        batch_size = verts.shape[0]
        colors = verts - verts.mean(1).unsqueeze(1)
        colors = colors / colors.norm(2, 2).max(1)[0].view(batch_size, 1, 1)
        colors = colors / 2 + 0.5
    else:
        colors = torch.ones_like(verts)
        colors[:, :, 0] = color[0] * colors[:, :, 0]
        colors[:, :, 1] = color[1] * colors[:, :, 1]
        colors[:, :, 2] = color[2] * colors[:, :, 2]
    return colors


def squashfig(fig=None):
    # TomNorway - https://stackoverflow.com/a/53516034
    if not fig:
        fig = plt.gcf()

    plt.subplots_adjust(0, 0, 1, 1, 0, 0)
    for ax in fig.axes:
        if isinstance(ax, Axes3D):
            ax.margins(0, 0, 0)
            ax.xaxis.set_major_locator(plt.NullLocator())
            ax.yaxis.set_major_locator(plt.NullLocator())
            ax.zaxis.set_major_locator(plt.NullLocator())
        else:
            ax.axis("off")
            ax.margins(0, 0)
            ax.xaxis.set_major_locator(plt.NullLocator())
            ax.yaxis.set_major_locator(plt.NullLocator())

import matplotlib
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter

matplotlib.font_manager._rebuild()
matplotlib.rc("font", **{"family": "Times New Roman", "style": "normal", "weight": "normal"})
matplotlib.rcParams["xtick.labelsize"] = 12
matplotlib.rcParams["ytick.labelsize"] = 12
title_fs = 15
ax_fs = 14
leg_fs = 12

plt.switch_backend("agg")

def plot_ho3d(
    ax, fractions, ref_errs, warp_errs, full, pref="Hand", ylim=0.05, unit="m", metric="EPE", show_legend=True
):
    ax.plot(fractions, ref_errs, "-", c="r", label="baseline")
    ax.plot(fractions, warp_errs, "-.", color="purple", label="+ warping loss")
    ax.plot(fractions, full, "--", label="full supervision", alpha=0.5, color="black")
    ax.set_title(f"{pref} pose error", fontsize=title_fs)
    all_vals = ref_errs + warp_errs
    y_min = 0.5 * min(all_vals)
    y_max = 1.2 * max(all_vals)
    ax.set_ylim(y_min, y_max)
    ax.set_xscale("log")
    ax.set_xlabel("% of fully supervised data", fontsize=ax_fs)
    ax.set_ylabel(f"Mean {metric} ({unit})", fontsize=ax_fs)
    ax.grid(alpha=0.2)
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.1d"))
    if unit == "m":
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    if show_legend:
        ax.legend()


fig = plt.figure(figsize=(8, 6))
axes = fig.subplots(2, 2)
fractions = [0.625, 2.5, 100]
ref_obj2d_errs = [61.3, 45.9, 30.9]
warp_obj2d_errs = [55.8, 44.3, 30.9]
full_obj2d = [30.9] * len(fractions)

ref_obj3d_errs = [0.0605, 0.0468, 0.0278]
warp_obj3d_errs = [0.0610, 0.0470, 0.0278]
full_obj3d = [0.278] * len(fractions)

ref_hand2d_errs = [46.6, 30.9, 25.5]
warp_hand2d_errs = [42.1, 30.7, 25.5]
full_hand2d = [25.5] * len(fractions)

ref_hand3d_errs = [0.0496, 0.0335, 0.0258]
warp_hand3d_errs = [0.0450, 0.0336, 0.0258]
full_hand3d = [0.0258] * len(fractions)

plot_ho3d(
    axes[0, 0],
    fractions,
    ref_hand3d_errs,
    warp_hand3d_errs,
    full_hand3d,
    pref="Hand",
    unit="m",
    metric="EPE",
    show_legend=False,
)
plot_ho3d(
    axes[1, 0],
    fractions,
    ref_hand2d_errs,
    warp_hand2d_errs,
    full_hand2d,
    pref="Hand",
    unit="pixels",
    metric="EPE",
    show_legend=False,
)
plot_ho3d(
    axes[0, 1],
    fractions,
    ref_obj3d_errs,
    warp_obj3d_errs,
    full_obj3d,
    pref="Object",
    unit="m",
    metric="vertex error",
    show_legend=True,
)
plot_ho3d(
    axes[1, 1],
    fractions,
    ref_obj2d_errs,
    warp_obj2d_errs,
    full_obj2d,
    pref="Object",
    unit="pixels",
    metric="vertex error",
    show_legend=True,
)
fig.tight_layout()
fig.savefig("fig7.png")
fig.savefig("fig7.pdf")
print("printed")

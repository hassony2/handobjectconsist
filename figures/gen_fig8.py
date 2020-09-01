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

def plot_fhb(
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
fractions = [0.625, 1.25, 2.5, 5, 10, 20, 100]
ref_obj2d_errs = [41.3, 30.8, 23.1, 19.2, 16.7, 14.3, 13.9]
warp_obj2d_errs = [33.9, 25.0, 18.4, 16.9, 15.3, 14.3, 13.9]
full_obj2d = [13.9] * len(fractions)

ref_obj3d_errs = [0.0618, 0.0455, 0.0349, 0.0295, 0.0256, 0.0230, 0.0223]
warp_obj3d_errs = [0.0544, 0.0394, 0.0297, 0.0269, 0.0242, 0.0227, 0.0222]
full_obj3d = [0.022] * len(fractions)

ref_hand2d_errs = [27.8, 21.1, 16.3, 14.0, 12.0, 10.7, 10.4]
warp_hand2d_errs = [23.5, 18.8, 15.2, 13.1, 12.3, 11.0, 10.4]
full_hand2d = [10.4] * len(fractions)

ref_hand3d_errs = [0.0443, 0.0344, 0.0272, 0.0230, 0.0205, 0.0187, 0.0180]
warp_hand3d_errs = [0.0402, 0.0325, 0.0266, 0.0225, 0.0201, 0.0188, 0.0180]
full_hand3d = [0.018] * len(fractions)

plot_fhb(
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
plot_fhb(
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
plot_fhb(
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
plot_fhb(
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
fig.savefig("fig8.png")
fig.savefig("fig8.pdf")
print("printed")

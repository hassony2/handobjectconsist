def eval_vis(eval_res, save_img_path, fig=None):
    fig.clf()
    fig_nb = len(eval_res)
    axes = fig.subplots(len(eval_res))
    for eval_idx, (eval_name, eval_res) in enumerate(eval_res.items()):
        if fig_nb > 1:
            ax = axes[eval_idx]
        else:
            ax = axes
        ax.plot(eval_res["thresholds"], eval_res["pck_curve"], "ro-", markersize=1, label="Ours")
        auc = eval_res["auc"]
        epe_mean = eval_res["epe_mean"]
        epe_med = eval_res["epe_median"]
        ax.set_title(f"{eval_name} epe_mean: {epe_mean:.3f}, auc: {auc:.3f}, epe_med: {epe_med:.3f}")
    fig.savefig(save_img_path)

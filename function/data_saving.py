import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.gridspec import GridSpec

def show_result(save_path, result_data):
    (
        measured_intensity,
        gth_phase,
        retrieved_phase,
        gth_aberration,
        retrieved_aberration,
        psnr_pha,
        ssim_pha,
        rmse_pha,
        pcc_pha,
    ) = result_data

    plt.rcParams["font.size"] = 10  # 全局字号
    fig = plt.figure(figsize=(12, 7), constrained_layout=True)
    gs = GridSpec(2, 3, figure=fig, height_ratios=[1, 0.8])

    # ---------------------------
    # 1) Intensity
    # ---------------------------
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(measured_intensity, cmap="gray")
    ax1.set_title("Diffraction (GT)", pad=6)
    ax1.axis("off")
    _add_colorbar(fig, ax1, im1)

    # ---------------------------
    # 2) ground-truth phase
    # ---------------------------
    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(gth_phase, cmap="hot")
    ax2.set_title("Phase (GT)", pad=6)
    ax2.axis("off")
    _add_colorbar(fig, ax2, im2)

    # ---------------------------
    # 3) Retrieved phase + metrics
    # ---------------------------
    ax3 = fig.add_subplot(gs[0, 2])
    title_text = (
        "Phase (Pred)\n"
        f"PSNR: {psnr_pha:.2f}  "
        f"SSIM: {ssim_pha:.2f}\n"
        f"RMSE: {rmse_pha:.4f}  "
        f"PCC: {pcc_pha:.2f}"
    )
    im3 = ax3.imshow(retrieved_phase, cmap="hot")
    ax3.set_title(title_text, pad=6)
    ax3.axis("off")
    _add_colorbar(fig, ax3, im3)

    # ---------------------------
    # 4) Zernike coefficients
    # ---------------------------
    ax4 = fig.add_subplot(gs[1, :])
    gth_aberration = np.round(gth_aberration, 4).flatten()
    retrieved_aberration = np.round(retrieved_aberration, 4).flatten()

    x = np.arange(len(gth_aberration))
    bar_w = 0.35

    ax4.bar(x - bar_w / 2, gth_aberration, width=bar_w, label="Ground truth")
    ax4.bar(x + bar_w / 2, retrieved_aberration, width=bar_w, label="PICNet")

    ax4.set_xticks(x)
    ax4.set_xticklabels([str(i) for i in range(3, 3 + len(x))])

    ax4.set_xlabel("Zernike polynomials")
    ax4.set_ylabel("Coefficients")
    ax4.legend(loc="upper right", frameon=False)
    ax4.grid(axis="y", linestyle="--", alpha=0.4)

    # ---------------------------
    # Save
    # ---------------------------
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _add_colorbar(fig, ax, im, width="3%", pad=0.03):
    """给单张图加一个窄色条，保持整体统一"""
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size=width, pad=pad)
    fig.colorbar(im, cax=cax)
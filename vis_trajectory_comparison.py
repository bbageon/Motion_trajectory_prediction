"""
3-Model Trajectory Comparison (논문용 정적 이미지)
GT (green) / Delta (blue) / Absolute (red)

- 스켈레톤: baseline(★회색) + 마지막 프레임만 표시
- 중간 프레임: 각 관절의 궤적선(점+연결선)만 표시
- 손목: 화살표 + 프레임 번호 레이블

출력: vis_trajectory_comparison.png
"""

import json
import math
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np

matplotlib.rcParams["font.family"] = "DejaVu Sans"
matplotlib.rcParams["axes.unicode_minus"] = False

# ── 경로 ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
OUTPUT_PNG = ROOT / "vis_trajectory_comparison.png"

SAMPLES = [
    {
        "label":       "Qualitative comparison of predicted arm motion trajectories for the 90° arm-raise action",
        "delta_json":  ROOT / "delta/predicted_motion_delta_90_0.json",
        "abs_json":    ROOT / "absolute/visAbsolute_x100_90_0.json",
        "source_json": ROOT / "dataset/new_data/90/1-ok.json",
        "baseline_idx": 34,
    },
]

PRED_FRAMES = 8

JOINTS = [
    "LEFT_SHOULDER", "RIGHT_SHOULDER",
    "LEFT_ELBOW",    "RIGHT_ELBOW",
    "LEFT_WRIST",    "RIGHT_WRIST",
]
CONNECTIONS = [
    ("LEFT_SHOULDER",  "LEFT_ELBOW"),
    ("LEFT_ELBOW",     "LEFT_WRIST"),
    ("RIGHT_SHOULDER", "RIGHT_ELBOW"),
    ("RIGHT_ELBOW",    "RIGHT_WRIST"),
    ("LEFT_SHOULDER",  "RIGHT_SHOULDER"),
]
WRIST_JOINTS = ["LEFT_WRIST", "RIGHT_WRIST"]

GT_COLOR       = "#2E7D32"   # 초록 (스켈레톤/궤적)
DELTA_COLOR    = "#1565C0"   # 파랑 (스켈레톤/궤적)
ABS_COLOR      = "#C62828"   # 빨강 (스켈레톤/궤적)
BASE_COLOR     = "#999999"   # 회색 (baseline)

GT_ARROW    = "#69F0AE"   # 밝은 민트 초록 (GT 화살표, 진초록 스켈레톤과 구분)
DELTA_ARROW = "#40C4FF"   # 밝은 하늘색 (Delta 화살표 강조)
ABS_ARROW   = "#FF6D00"   # 주황 (Absolute 화살표 강조)


# ── 데이터 로딩 ───────────────────────────────────────────────────────────────
def _load_vis_json(path: Path):
    with open(path, encoding="utf-8") as f:
        seq = json.load(f)["pose_sequence"]
    baseline = {j: tuple(seq[0][j]) for j in JOINTS if j in seq[0]}
    frames   = [{j: tuple(fr[j]) for j in JOINTS if j in fr}
                for fr in seq[1: PRED_FRAMES + 1]]
    return baseline, frames


def load_gt(source_path: Path, baseline_idx: int, n_frames: int):
    with open(source_path, encoding="utf-8") as f:
        seq = json.load(f)["pose_sequence"]
    baseline = {j: tuple(seq[baseline_idx][j])
                for j in JOINTS if j in seq[baseline_idx]}
    frames = []
    for t in range(1, n_frames + 1):
        fi = baseline_idx + t
        frames.append(
            {j: tuple(seq[fi][j]) for j in JOINTS if j in seq[fi]}
            if fi < len(seq) else {}
        )
    return baseline, frames


# ── 스켈레톤 (baseline / 마지막 프레임용) ────────────────────────────────────
def draw_skeleton(ax, frame: dict, color: str, alpha: float,
                  lw: float, ms: float, zorder: int, marker: str = "o"):
    for (ja, jb) in CONNECTIONS:
        if ja in frame and jb in frame:
            ax.plot(
                [frame[ja][0], frame[jb][0]],
                [frame[ja][1], frame[jb][1]],
                color=color, lw=lw, alpha=alpha, zorder=zorder,
            )
    for j in JOINTS:
        if j in frame:
            ax.scatter(
                frame[j][0], frame[j][1],
                s=ms, color=color,
                edgecolors="white", linewidths=0.8,
                alpha=alpha,
                zorder=zorder + 1, marker=marker,
            )


# ── 프레임별 스켈레톤 + 관절 궤적선 ─────────────────────────────────────────
def draw_joint_trajectories(ax, baseline: dict, frames: list,
                             color: str, lw: float, ms: float,
                             zorder: int):
    """
    - 각 프레임의 스켈레톤(뼈대 연결선 + 관절 점)을 alpha 그라디언트로 표시
      frame1=0.08(거의 투명) → frame8=1.0(완전 불투명)
    - 각 관절의 이동 궤적선도 동일 alpha로 표시
    """
    n = len(frames)
    alphas = np.linspace(0.08, 1.0, n) if n > 1 else [1.0]

    # 1) 프레임별 스켈레톤 (뼈대 연결선 + 관절 점)
    for fi, frame in enumerate(frames):
        alpha = float(alphas[fi])
        # 뼈대 연결선
        for (ja, jb) in CONNECTIONS:
            if ja in frame and jb in frame:
                ax.plot([frame[ja][0], frame[jb][0]],
                        [frame[ja][1], frame[jb][1]],
                        color=color, lw=lw, alpha=alpha, zorder=zorder)
        # 관절 점
        for j in JOINTS:
            if j in frame:
                ax.scatter(frame[j][0], frame[j][1],
                           s=ms, color=color,
                           edgecolors="white", linewidths=0.5,
                           alpha=alpha, zorder=zorder + 1, marker="o")

    # 2) 관절별 이동 궤적선 (baseline→frame1→...→frameN)
    for joint in JOINTS:
        coords = []
        if joint in baseline:
            coords.append(baseline[joint])
        for fr in frames:
            if joint in fr:
                coords.append(fr[joint])
        if len(coords) < 2:
            continue
        for i in range(len(coords) - 1):
            alpha = float(alphas[min(i, n - 1)])
            x1, y1 = coords[i]
            x2, y2 = coords[i + 1]
            ax.plot([x1, x2], [y1, y2],
                    color=color, lw=max(lw - 0.6, 0.8), alpha=alpha,
                    zorder=zorder - 1, linestyle="--")


# ── 손목 화살표 + 프레임 번호 ─────────────────────────────────────────────────
def draw_wrist_arrows(ax, frames: list,
                      color: str, lw: float, zorder: int):
    n = len(frames)
    alphas = np.linspace(0.15, 1.0, n) if n > 1 else [1.0]
    for joint in WRIST_JOINTS:
        coords = []
        for fr in frames:
            if joint in fr:
                coords.append(fr[joint])

        # 모든 구간에 화살표 표시 (alpha 그라디언트)
        for i in range(len(coords) - 1):
            x1, y1 = coords[i]
            x2, y2 = coords[i + 1]
            if math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) >= 1e-5:
                alpha = float(alphas[min(i + 1, n - 1)])
                ax.annotate(
                    "",
                    xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(
                        arrowstyle="-|>",
                        color=color,
                        lw=lw,
                        mutation_scale=10,
                        alpha=alpha,
                    ),
                    zorder=zorder,
                )



# ── 메인 ──────────────────────────────────────────────────────────────────────
def main():
    fig, axes = plt.subplots(1, 1, figsize=(9, 10))
    axes = [axes]
    # fig.suptitle(
    #     "Fig. 1.  Predicted arm motion trajectories for the 90° Turn action.\n"
    #     "Skeleton shown at baseline (★) and final frame only. "
    #     "Lines = per-joint trajectory.  Arrows = wrist direction.",
    #     fontsize=10.5, y=1.02, ha="center",
    # )

    for ax, sample in zip(axes, SAMPLES):
        label      = sample["label"]
        delta_path = sample["delta_json"]
        abs_path   = sample["abs_json"]
        src_path   = sample["source_json"]
        bidx       = sample["baseline_idx"]

        if not delta_path.exists():
            ax.text(0.5, 0.5, f"Delta file missing\n{delta_path.name}",
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_title(label); continue

        d_base, d_seq = _load_vis_json(delta_path)
        _, gt_seq     = load_gt(src_path, bidx, PRED_FRAMES)
        gt_base       = d_base

        a_base, a_seq = {}, []
        if abs_path.exists():
            a_base, a_seq = _load_vis_json(abs_path)

        # --- 축 범위 ---
        all_x, all_y = [], []
        for pool in ([d_base] + d_seq + gt_seq +
                     ([a_base] if a_base else []) + a_seq):
            for j in JOINTS:
                if j in pool:
                    all_x.append(pool[j][0])
                    all_y.append(pool[j][1])

        pad  = 0.04
        xmin = min(all_x) - pad;  xmax = max(all_x) + pad
        ymin = min(all_y) - pad;  ymax = max(all_y) + pad
        xr, yr = xmax - xmin, ymax - ymin
        if xr > yr:
            mid = (ymin + ymax) / 2
            ymin, ymax = mid - xr / 2, mid + xr / 2
        else:
            mid = (xmin + xmax) / 2
            xmin, xmax = mid - yr / 2, mid + yr / 2

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymax, ymin)
        ax.set_aspect("equal")
        ax.xaxis.set_major_locator(plt.MultipleLocator(0.05))
        ax.yaxis.set_major_locator(plt.MultipleLocator(0.05))
        ax.grid(True, linestyle="-", alpha=0.3, linewidth=0.6, color="gray")
        ax.tick_params(labelsize=9)
        ax.set_xlabel("X Coordinate", fontsize=11, fontweight="bold")
        ax.set_ylabel("Y Coordinate", fontsize=11, fontweight="bold")
        ax.set_title(label, fontsize=12, fontweight="bold", pad=10)

        # ── 그리기 순서: GT → Absolute → Delta ──────────────────────────────

        # 1) GT (초록)
        draw_joint_trajectories(ax, gt_base, gt_seq, GT_COLOR,
                                 lw=1.8, ms=30, zorder=3)
        draw_wrist_arrows(ax, gt_seq, GT_ARROW, lw=1.8, zorder=4)

        # 3) Absolute (빨강)
        if a_seq:
            draw_joint_trajectories(ax, a_base, a_seq, ABS_COLOR,
                                     lw=1.8, ms=30, zorder=5)
            draw_wrist_arrows(ax, a_seq, ABS_ARROW, lw=1.8, zorder=6)

        # 4) Delta (파랑, 최상단)
        draw_joint_trajectories(ax, d_base, d_seq, DELTA_COLOR,
                                 lw=2.2, ms=35, zorder=7)
        draw_wrist_arrows(ax, d_seq, DELTA_ARROW, lw=2.0, zorder=8)

    # --- 범례 ---
    ax = axes[0]
    legend_hdl = [
        mlines.Line2D([], [], color=GT_COLOR,    lw=2.0, marker="o", ms=6,
                      label="GT"),
        mlines.Line2D([], [], color=DELTA_COLOR,  lw=2.2, marker="o", ms=6,
                      label="Delta (Ours)"),
        mlines.Line2D([], [], color=ABS_COLOR,    lw=2.0, marker="o", ms=6,
                      label="Absolute"),
    ]
    ax.legend(
        handles=legend_hdl,
        loc="upper right",
        fontsize=10,
        framealpha=0.93,
        edgecolor="#aaaaaa",
        title="Legend",
        title_fontsize=10,
    )

    fig.tight_layout()
    fig.savefig(str(OUTPUT_PNG), dpi=220, bbox_inches="tight")
    print(f"[SAVED] {OUTPUT_PNG}")
    plt.show()


if __name__ == "__main__":
    main()

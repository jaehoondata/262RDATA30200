"""Generate calibration figures and undistortion diagnostics."""
import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.path import Path as MplPath  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402
from mpl_toolkits.axes_grid1 import make_axes_locatable  # noqa: E402

from .camera import as_dist5, distort_normalized  # noqa: E402

SURFACE = "#ffffff"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#333333"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

CONDITION_COLORS = {
    "Z0": "#0072b2", "Z_gamma": "#d55e00", "CV5": "#009e73", "CV0": "#eda100",
    "LM0": "#e87ba4", "LM2": "#008300", "CV2": "#4a3aa7",
}
# Report condition labels.
CONDITION_LABELS = {
    "Z0": r"$Z_0$", "Z_gamma": r"$Z_\gamma$",
    "LM0": r"$\mathrm{LM}_0$", "LM2": r"$\mathrm{LM}_2$",
    "CV0": r"$\mathrm{CV}_0$", "CV2": r"$\mathrm{CV}_2$", "CV5": r"$\mathrm{CV}_5$",
}
CONDITION_MARKERS = {"Z0": "+", "Z_gamma": "x", "CV5": "s", "CV0": "2", "LM0": "3", "LM2": "4", "CV2": "+"}
BLUE_RAMP = LinearSegmentedColormap.from_list(
    "blue_seq", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])
MAX_OVERLAY_CONDITIONS = 3

plt.rcParams.update({
    "font.family": ["Helvetica Neue", "Arial", "DejaVu Sans"],
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK_2, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlecolor": INK,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 8.5, "legend.frameon": False,
})
_HALO = [pe.withStroke(linewidth=2.5, foreground="white")]


def _rgb(image):
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _nice_step(x):
    """Choose a readable 1-2-5 step."""
    if x <= 1:
        return 1.0
    base = 10.0 ** np.floor(np.log10(x))
    return float(max(m * base for m in (1, 2, 5) if m * base <= x))


def _save(fig, path, tight=True):
    kwargs = {"pil_kwargs": {"quality": 95}} if str(path).lower().endswith((".jpg", ".jpeg")) else {}
    fig.savefig(path, dpi=300, bbox_inches="tight" if tight else None, pad_inches=0.04, **kwargs)
    plt.close(fig)


def _board_view(ax, image, points, padding=0.12):
    """Crop the plot around the board."""
    h, w = image.shape[:2]
    points = np.asarray(points)
    lo, hi = points.min(axis=0), points.max(axis=0)
    pad = padding * max(hi - lo)
    ax.imshow(_rgb(image))
    ax.set_xlim(max(0, lo[0] - pad), min(w, hi[0] + pad))
    ax.set_ylim(min(h, hi[1] + pad), max(0, lo[1] - pad))
    ax.set_aspect("equal")
    ax.set_axis_off()


def plot_corner_example(image, corners, path, title):
    """Plot detected corners and the board origin."""
    points = np.asarray(corners)
    fig, ax = plt.subplots(figsize=(2.7, 2.25), layout="constrained")
    _board_view(ax, image, points)
    ax.scatter(*points.T, s=8, facecolors="#00bcd4", edgecolors="black",
               linewidths=0.4, path_effects=[pe.withStroke(linewidth=1.4, foreground="white")])
    origin = points[0]
    ax.scatter(*origin, s=70, facecolors="none", edgecolors="#c62828", linewidths=1.5)
    ax.annotate("", xy=points[1], xytext=origin,
                arrowprops={"arrowstyle": "-|>", "color": "#c62828", "lw": 1.6},
                annotation_clip=False)
    ax.annotate("0", xy=origin, xytext=(-8, 9), textcoords="offset points",
                color="#c62828", fontsize=10, fontweight="bold",
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 1})
    ax.set_anchor("N")
    fig.suptitle(title, x=0.02, ha="left", fontsize=9)
    _save(fig, path, tight=False)


def plot_reprojection(image, observed, projected, path, title):
    """Plot residual vectors and a zoomed reprojection comparison."""
    if len(projected) > MAX_OVERLAY_CONDITIONS:
        raise ValueError(f"At most {MAX_OVERLAY_CONDITIONS} conditions per overlay, got {list(projected)}")
    observed = np.asarray(observed)
    h, w = image.shape[:2]
    residuals = {c: np.asarray(p) - observed for c, p in projected.items()}
    max_res = max(float(np.linalg.norm(r, axis=1).max()) for r in residuals.values())
    magnify = _nice_step(0.04 * max(w, h) / max(max_res, 1e-9))
    worst = int(np.argmax(np.max([np.linalg.norm(r, axis=1) for r in residuals.values()], axis=0)))
    cx, cy = observed[worst]
    half = max(6.0, 1.8 * max_res)

    fig, (ax, axz) = plt.subplots(1, 2, figsize=(5.5, 2.8), layout="constrained",
                                  gridspec_kw={"width_ratios": [1.45, 1]})
    _board_view(ax, image, observed)
    # Avoid overlapping Z_gamma and Z0 overview arrows.
    overview = {c: r for c, r in residuals.items() if c != "Z_gamma" or "Z0" not in residuals}
    for cond, r in overview.items():
        arrows = ax.quiver(observed[:, 0], observed[:, 1], r[:, 0] * magnify, r[:, 1] * magnify,
                           angles="xy", scale_units="xy", scale=1, color=CONDITION_COLORS[cond],
                           width=0.004, headwidth=3.5)
        arrows.set_path_effects([pe.withStroke(linewidth=0.6, foreground="white")])
    ax.scatter(*observed.T, s=5, facecolors="none", edgecolors="black", linewidths=0.35,
               path_effects=[pe.withStroke(linewidth=1.0, foreground="white")])
    box = Rectangle((cx - half, cy - half), 2 * half, 2 * half, fill=False,
                    edgecolor="black", linewidth=1.1, linestyle="--", path_effects=_HALO)
    ax.add_patch(box)
    ax.annotate(str(worst), xy=(cx, cy), xytext=(-15, -16), textcoords="offset points",
                ha="right", va="top", fontsize=8,
                bbox={"facecolor": "white", "edgecolor": AXIS, "pad": 1},
                arrowprops={"arrowstyle": "->", "color": INK, "lw": 0.8,
                            "path_effects": _HALO})
    names = ", ".join(CONDITION_LABELS.get(c, c) for c in overview)
    ax.set_title(f"{title}\n{names}: residual vectors ×{magnify:.0f}", loc="left")

    axz.imshow(_rgb(image))
    axz.scatter(*observed.T, s=45, facecolors="none", edgecolors=INK, linewidths=1.2,
                label="Observed", path_effects=_HALO)
    for cond, p in projected.items():
        axz.scatter(*np.asarray(p).T, s=48, marker=CONDITION_MARKERS[cond],
                    color=CONDITION_COLORS[cond], linewidths=1.4,
                    label=CONDITION_LABELS.get(cond, cond), path_effects=_HALO)
    axz.set_xlim(cx - half, cx + half)
    axz.set_ylim(cy + half, cy - half)
    tick = _nice_step(0.7 * half)
    offsets = np.array([-tick, 0, tick])
    axz.set_xticks(cx + offsets, [f"{x:g}" for x in offsets])
    axz.set_yticks(cy + offsets, [f"{x:g}" for x in offsets])
    axz.set_xlabel("Horizontal offset (px)")
    axz.set_ylabel("Vertical offset (px)")
    axz.set_title(f"Corner {worst}\nResiduals ×1", loc="left")
    handles, labels = axz.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0),
               ncol=len(labels), handletextpad=0.4, columnspacing=1.1)
    _save(fig, path)


def plot_rmse_bars(image_ids, per_image, overall, path):
    """Plot per-image RMSE by condition."""
    conds = list(per_image)
    n = len(image_ids)
    width = 0.8 / len(conds)
    x = np.arange(n)
    fig, ax = plt.subplots(figsize=(5.5, 2.8), layout="constrained")
    for k, cond in enumerate(conds):
        name = CONDITION_LABELS.get(cond, cond)
        ax.bar(x + (k - (len(conds) - 1) / 2) * width, per_image[cond], width=width * 0.94,
               color=CONDITION_COLORS[cond], label=f"{name}: {overall[cond]:.3f} px",
               edgecolor="white", linewidth=0.25)
    labels = [str(i).removesuffix(".jpg").removeprefix("IMG_") for i in image_ids]
    ax.set_xticks(x, labels, rotation=60, ha="right")
    ax.set_xlabel("Image ID (IMG_ prefix omitted)")
    ax.set_ylabel("RMSE (px)")
    ax.set_ylim(bottom=0)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=4))
    ax.yaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.01), ncol=min(3, len(conds)),
              borderaxespad=0, handlelength=1.2, columnspacing=1.2, handletextpad=0.4)
    _save(fig, path)


# Allow slow-converging inverse distortion points.
UNDISTORT_CRITERIA = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 1000, 1e-10)
ROUND_TRIP_TOL_PX = 0.01


def undistortion_shift(points, K, dist, criteria=UNDISTORT_CRITERIA, tol=ROUND_TRIP_TOL_PX):
    """Return undistorted points, shifts (px), and round-trip validity."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    K = np.asarray(K, dtype=np.float64)
    d = as_dist5(dist)
    und = np.full_like(pts, np.nan)
    finite = np.isfinite(pts).all(axis=1)
    if finite.any():
        und[finite] = cv2.undistortPointsIter(pts[finite].reshape(-1, 1, 2), K, d, None, K,
                                              criteria).reshape(-1, 2)
    with np.errstate(invalid="ignore"):
        xy = (np.linalg.inv(K) @ np.column_stack([und, np.ones(len(und))]).T).T[:, :2]
        back = (K @ np.column_stack([distort_normalized(xy, d), np.ones(len(xy))]).T).T[:, :2]
        err = np.linalg.norm(back - pts, axis=1)
        valid = finite & np.isfinite(und).all(axis=1) & np.isfinite(err) & (err <= tol)
        shift = np.linalg.norm(und - pts, axis=1)
    return und, shift, valid


def undistortion_field(K, dist, image_size, coverage_points=None, step=None):
    """Measure correction magnitude inside and outside observed coverage."""
    w, h = image_size
    step = step or max(4, max(w, h) // 200)
    gx, gy = np.meshgrid(np.arange(0, w, step), np.arange(0, h, step))
    grid = np.column_stack([gx.ravel(), gy.ravel()]).astype(np.float64)
    corners = np.array([[0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1]], dtype=np.float64)
    _, shift, valid = undistortion_shift(np.vstack([grid, corners]), K, dist)
    hull = None
    inside = np.ones(len(shift), dtype=bool)
    if coverage_points is not None:
        hull = cv2.convexHull(np.asarray(coverage_points, np.float32).reshape(-1, 1, 2)).reshape(-1, 2)
        inside = MplPath(hull).contains_points(np.vstack([grid, corners]))
    n = len(grid)

    def region_max(mask):
        vals = shift[mask & valid]
        return float(vals.max()) if len(vals) else None

    return {
        "gx": gx, "gy": gy, "hull": hull,
        "shift": shift[:n].reshape(gx.shape), "valid": valid[:n].reshape(gx.shape),
        "inside": inside[:n].reshape(gx.shape),
        "inside_max_px": region_max(inside),
        "outside_max_px": region_max(~inside),
        "n_failed": int((~valid).sum()),
        "n_points": int(len(shift)),
        "criteria": list(UNDISTORT_CRITERIA), "round_trip_tol_px": ROUND_TRIP_TOL_PX,
    }


def _fmt_max(v, digits):
    return "N/A" if v is None else f"{v:.{digits}f} px"


def plot_undistortion(image, K, dist, path, title, coverage_points=None):
    """Plot original and undistorted images with a correction map."""
    h, w = image.shape[:2]
    und = cv2.undistort(image, K, np.asarray(dist, dtype=np.float64))
    f = undistortion_field(K, dist, (w, h), coverage_points)
    gx, gy, valid, inside = f["gx"], f["gy"], f["valid"], f["inside"]

    fig = plt.figure(figsize=(5.5, 5.4), layout="constrained")
    fig.suptitle(title, x=0.02, ha="left", fontsize=9)
    grid = fig.add_gridspec(2, 1, height_ratios=[1, 1.65])
    photos = grid[0].subgridspec(1, 2)
    ax0, ax1 = fig.add_subplot(photos[0]), fig.add_subplot(photos[1])
    axm = fig.add_subplot(grid[1])
    ax0.imshow(_rgb(image))
    ax1.imshow(_rgb(und))
    ax0.set_title("(a) Original", loc="left")
    ax1.set_title("(b) Undistorted (same K)", loc="left")
    for ax in (ax0, ax1):
        ax.set_axis_off()

    shown = np.ma.masked_where(~(inside & valid), f["shift"])
    cmap = BLUE_RAMP.with_extremes(bad="#dedede")
    im = axm.imshow(shown, cmap=cmap, extent=(0, w, h, 0), vmin=0)
    if shown.count() > 0 and float(shown.max()) > 0:
        levels = MaxNLocator(nbins=4).tick_values(0, float(shown.max()))
        levels = levels[(levels > 0) & (levels < float(shown.max()))]
        if len(levels):
            cs = axm.contour(gx, gy, shown, levels=levels, colors=INK_2, linewidths=0.7)
            texts = axm.clabel(cs, fmt="%g", fontsize=8.5)
            for text in texts:
                text.set_path_effects([pe.withStroke(linewidth=2, foreground="white")])
    handles = [Patch(facecolor="#dedede", label="Outside coverage")]
    if f["hull"] is not None:
        loop = np.vstack([f["hull"], f["hull"][:1]])
        axm.plot(loop[:, 0], loop[:, 1], color="black", linewidth=1.1, linestyle="--")
        handles.insert(0, Line2D([], [], color="black", linestyle="--", label="Corner coverage"))
    if (~valid).any():
        axm.scatter(gx[~valid], gy[~valid], s=8, marker="x", color=INK, linewidths=0.6)
        handles.append(Line2D([], [], color=INK, marker="x", linestyle="none", label="Failed check"))
    axm.set_title("(c) Correction magnitude", loc="left")
    axm.set_axis_off()
    cax = make_axes_locatable(axm).append_axes("right", size="4%", pad=0.08)
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("Displacement (px)")
    cbar.ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    axm.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.015),
               ncol=len(handles), handlelength=1.6, columnspacing=1.1)
    axm.text(0.5, -0.13,
             f"Max inside: {_fmt_max(f['inside_max_px'], 1)}  |  "
             f"Max outside: {_fmt_max(f['outside_max_px'], 0)} (extrapolated)",
             transform=axm.transAxes, ha="center", va="top", fontsize=8.5)
    _save(fig, path)
    return {k: f[k] for k in ("inside_max_px", "outside_max_px", "n_failed", "n_points",
                              "criteria", "round_trip_tol_px")}

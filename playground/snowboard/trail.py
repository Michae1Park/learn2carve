"""The track the board leaves on the snow: drawn live in the viewport and saved as a top-down picture.

Each policy step adds one flat strip under the board. Its width is the snow the board sweeps sideways:

    width = L · |sin β| + w · |cos β|        L = board length, w = waist width, β = slip angle

A carve (β ≈ 0) leaves a line as wide as the board; a skid smears a band up to the board's length (side slip).

Positions come in the slope frame (x downhill along the snow, y across); ``q_slope`` places the strips in the world.
"""

from __future__ import annotations

import math

import torch


class SnowTrail:
    def __init__(self, board_length: float, board_width: float, q_slope: torch.Tensor, max_marks: int = 4000):
        self.L, self.w = board_length, board_width
        self.max_marks = max_marks
        self.q_slope = q_slope.detach().cpu()  # slope frame → world, (x, y, z, w)
        self.marks: list[tuple[float, float, float, float, float]] = []  # x, y, heading, step length, width
        import isaaclab.sim as sim_utils
        from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

        # headless, visualize() is a no-op, so this costs nothing without a viewer
        self.markers = VisualizationMarkers(
            VisualizationMarkersCfg(
                prim_path="/Visuals/SnowTrail",
                markers={
                    "track": sim_utils.CuboidCfg(
                        size=(1.0, 1.0, 0.002),
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.25, 0.45, 0.85)),
                    )
                },
            )
        )
        self.last_xy: tuple[float, float] | None = None

    def reset(self):
        """Start a new trail segment (after a fall the rider jumps back to the start)."""
        self.last_xy = None

    def add(self, board_pos: torch.Tensor, slip_angle: float):
        """board_pos: (3,) slope-frame position of one env's board."""
        x, y = board_pos[0].item(), board_pos[1].item()
        if self.last_xy is None:
            self.last_xy = (x, y)
            return
        px, py = self.last_xy
        step = math.hypot(x - px, y - py)
        if step < 1e-3:
            return
        heading = math.atan2(y - py, x - px)
        width = self.L * abs(math.sin(slip_angle)) + self.w * abs(math.cos(slip_angle))
        self.marks.append(((x + px) / 2, (y + py) / 2, heading, step, width))
        self.marks = self.marks[-self.max_marks:]
        self.last_xy = (x, y)
        self._draw()

    def _draw(self):
        if not self.marks:
            return
        from isaaclab.utils.math import quat_apply, quat_mul

        m = torch.tensor(self.marks)
        pos = torch.stack([m[:, 0], m[:, 1], torch.full_like(m[:, 0], 0.003)], dim=-1)
        quat = torch.zeros(len(m), 4)  # (x, y, z, w), yaw on the snow
        quat[:, 2], quat[:, 3] = torch.sin(m[:, 2] / 2), torch.cos(m[:, 2] / 2)
        q = self.q_slope.expand(len(m), 4)
        pos, quat = quat_apply(q, pos), quat_mul(q, quat)
        scale = torch.stack([m[:, 3] * 1.05, m[:, 4], torch.ones_like(m[:, 0])], dim=-1)
        self.markers.visualize(translations=pos, orientations=quat, scales=scale)

    def save_png(self, path, title: str = "", line=None, fence: float | None = None, x_end: float | None = None):
        """Top-down view, downhill pointing down the page. line(x) -> y_ref draws the target (dashed), fence the
        fences at ±fence (orange)."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        from matplotlib.transforms import Affine2D

        if not self.marks:
            return
        xs = [m[0] for m in self.marks]
        ys = [m[1] for m in self.marks]
        fig, ax = plt.subplots(figsize=(5, 9))
        ax.set_facecolor("#f4f7fb")
        for x, y, heading, step, width in self.marks:
            # page coordinates: right = +y (across the slope), down = +x (downhill)
            rect = Rectangle((-step / 2, -width / 2), step * 1.05, width, color="#3b6fd8", alpha=0.6, lw=0)
            angle = math.degrees(math.atan2(-math.cos(heading), math.sin(heading)))
            rect.set_transform(Affine2D().rotate_deg(angle).translate(y, -x) + ax.transData)
            ax.add_patch(rect)
        x_hi = max(max(xs), x_end or 0.0)
        if line is not None:
            grid = [x_hi * i / 400 for i in range(401)]
            ax.plot([line(x) for x in grid], [-x for x in grid], "--", color="#555555", lw=1, label="target line")
        if fence is not None:
            for side in (-1, 1):
                ax.plot([side * fence] * 2, [0, -x_hi], color="#ff6a2e", lw=2, label="fence" if side > 0 else None)
            ax.legend(loc="lower right", fontsize=8)
        pad = 2.0
        y_lim = fence + pad if fence is not None else max(abs(min(ys)), abs(max(ys))) + pad
        ax.set_xlim(-y_lim, y_lim)
        ax.set_ylim(-x_hi - pad, -min(xs) + pad)
        ax.set_aspect("equal")
        ax.set_xlabel("across the slope [m]")
        ax.set_ylabel("down the fall line [m] (downhill = down)")
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)

"""The board's track on the snow, drawn live in the viewport and saved as a top-down picture.

Each step adds one strip as wide as the snow the board sweeps: L · |sin β| + w · |cos β| (β = slip angle).
Positions are in the slope frame; ``q_slope`` places the strips in the world.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import torch

TRACK_COLOR = (0.25, 0.45, 0.85)
STRIP_OVERLAP = 1.05
STRIP_LIFT = 0.003  # m above the snow


class Mark(NamedTuple):
    x: float
    y: float
    heading: float
    length: float
    width: float


class SnowTrail:
    def __init__(self, board_length: float, board_width: float, q_slope: torch.Tensor, max_marks: int = 4000):
        import isaaclab.sim as sim_utils
        from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

        self.board_length, self.board_width = board_length, board_width
        self.max_marks = max_marks
        self.q_slope = q_slope.detach().cpu()
        self.marks: list[Mark] = []
        self.last_xy: tuple[float, float] | None = None
        self.markers = VisualizationMarkers(  # a no-op headless
            VisualizationMarkersCfg(
                prim_path="/Visuals/SnowTrail",
                markers={
                    "track": sim_utils.CuboidCfg(
                        size=(1.0, 1.0, 0.002), visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=TRACK_COLOR)
                    )
                },
            )
        )

    def reset(self):
        self.last_xy = None

    def add(self, board_pos: torch.Tensor, slip_angle: float):
        x, y = board_pos[0].item(), board_pos[1].item()
        if self.last_xy is None:
            self.last_xy = (x, y)
            return
        px, py = self.last_xy
        step = math.hypot(x - px, y - py)
        if step < 1e-3:
            return
        width = self.board_length * abs(math.sin(slip_angle)) + self.board_width * abs(math.cos(slip_angle))
        self.marks.append(Mark((x + px) / 2, (y + py) / 2, math.atan2(y - py, x - px), step, width))
        self.marks = self.marks[-self.max_marks:]
        self.last_xy = (x, y)
        self._draw()

    def _draw(self):
        from isaaclab.utils.math import quat_apply, quat_mul

        m = torch.tensor(self.marks)
        x, y, heading, length, width = m.unbind(-1)
        pos = torch.stack([x, y, torch.full_like(x, STRIP_LIFT)], dim=-1)
        yaw = torch.zeros(len(m), 4)
        yaw[:, 2], yaw[:, 3] = torch.sin(heading / 2), torch.cos(heading / 2)
        q = self.q_slope.expand(len(m), 4)
        scale = torch.stack([length * STRIP_OVERLAP, width, torch.ones_like(x)], dim=-1)
        self.markers.visualize(translations=quat_apply(q, pos), orientations=quat_mul(q, yaw), scales=scale)

    def save_png(self, path, title: str = "", line=None, fence: float | None = None, x_end: float | None = None):
        """Downhill points down the page. line(x) → y draws a dashed target; fence draws the fences at ±fence."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        from matplotlib.transforms import Affine2D

        if not self.marks:
            return
        xs = [m.x for m in self.marks]
        ys = [m.y for m in self.marks]
        fig, ax = plt.subplots(figsize=(5, 9))
        ax.set_facecolor("#f4f7fb")
        for m in self.marks:
            # page: right = +y (across), down = +x (downhill)
            rect = Rectangle((-m.length / 2, -m.width / 2), m.length * STRIP_OVERLAP, m.width, color="#3b6fd8",
                             alpha=0.6, lw=0)
            angle = math.degrees(math.atan2(-math.cos(m.heading), math.sin(m.heading)))
            rect.set_transform(Affine2D().rotate_deg(angle).translate(m.y, -m.x) + ax.transData)
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

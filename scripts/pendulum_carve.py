"""Inverted pendulum on a snowboard (roll plane): can a rider balanced only by its edge link turns every ~2 s?

    python3 scripts/pendulum_carve.py
    python3 scripts/pendulum_carve.py --period 3 --grip 1.2

No Isaac Sim. The rider is one rigid inverted pendulum (COM 0.9 m above the edge, lean φ from the snow normal,
+ = toward the toe side) on a free pivot: nothing at the board can hold it up. The only balancing force is the turn,
like a bicycle: on edge θ the sidecut steers the board onto R = 7.6 m · cos θ (envs/snow.py), so the COM feels
v² / R toward the outside, capped by grip (--grip, in g). Flat (|θ| < 3°) there is no turn and no side force.

    φ̈ = (g sin φ + a cos φ) / l,   a = -sign(θ) v² / (R_sc cos θ) on edge, 0 flat

(A double pendulum balanced by hip torque alone, on the same free pivot, diverged at every gain: it needs huge torso
swings. Balance comes from steering, as on a bicycle.)

Note the gap: on edge the side force is at least v² / R_sc, so while carving the rider is never balanced at less than
atan(v² / (g R_sc)) of lean (8 m/s: 41°). An edge change has to cross that gap through the flat board.

Controller (the edge is the only input): PD on the lean toward a target lean that switches sides every half period
gives the side force wanted, then the edge that makes it (flat if too small to carve). Over a grid of gains it reports
the best: time up, edge changes, and the largest edge used.
"""

import argparse
import math

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--period", type=float, default=4.0, help="full S [s] (a turn every period / 2)")
p.add_argument("--lean", type=float, default=None, help="target lean at the apex [deg]; default: a 45° edge's balance")
p.add_argument("--grip", type=float, default=1.5, help="max side force [g] before the edge skids")
p.add_argument("--seconds", type=float, default=12.0)
args = p.parse_args()

G, R_SC, L = 9.81, 7.6, 0.9
MIN_EDGE, MAX_EDGE = math.radians(3.0), math.radians(80.0)


def side_force(theta: float, v: float) -> float:
    if abs(theta) < MIN_EDGE:
        return 0.0
    a = v * v / (R_SC * math.cos(theta))
    return -math.copysign(min(a, args.grip * G), theta)


def edge_for(a_want: float, v: float) -> float:
    """Edge that makes side force a_want; flat if a_want is well below the smallest carving force."""
    a_min = v * v / R_SC
    if abs(a_want) < 0.5 * a_min:
        return 0.0
    c = min(1.0, a_min / abs(a_want))
    return -math.copysign(min(MAX_EDGE, max(MIN_EDGE * 1.01, math.acos(c))), a_want)


def ride(v: float, lean: float, kp: float, kd: float):
    dt = 1 / 1000
    phi, w = 0.0, 0.0
    flips, side, max_edge = 0, 0, 0.0
    for k in range(int(args.seconds / dt)):
        t = k * dt
        target = lean * max(-1.0, min(1.0, 1.8 * math.sin(2 * math.pi * t / args.period)))
        a_want = -G * math.tan(target) - kp * (phi - target) - kd * w  # lean + needs a side force -
        theta = edge_for(a_want, v)
        a = side_force(theta, v)
        w += dt * (G * math.sin(phi) + a * math.cos(phi)) / L
        phi += dt * w
        if abs(phi) > math.radians(85):
            return t, flips, max_edge, True
        held = (theta > math.radians(15)) - (theta < -math.radians(15))
        if held and side and held != side:
            flips += 1
        if held:
            side = held
        max_edge = max(max_edge, abs(theta))
    return args.seconds, flips, max_edge, False


print(f"[pend] S period {args.period} s (a turn every {args.period / 2:.1f} s), grip {args.grip} g, {args.seconds:.0f} s")
print("[pend] speed  min carve lean  target lean   up_s  flips  max edge°  fell   (kp, kd)")
for v in (4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 13.0):
    lean = math.radians(args.lean) if args.lean else math.atan2(v * v / (R_SC * math.cos(math.radians(45))), G)
    best = None
    for kp in (5.0, 10.0, 20.0, 40.0, 80.0):
        for kd in (1.0, 3.0, 6.0, 12.0, 24.0):
            r = ride(v, lean, kp, kd)
            key = (not r[3], r[0], r[1])
            if best is None or key > best[0]:
                best = (key, r, (kp, kd))
    up, flips, max_edge, fell = best[1]
    print(f"[pend] {v:5.1f}  {math.degrees(math.atan2(v * v / R_SC, G)):10.1f}°  {math.degrees(lean):10.1f}°  "
          f"{up:5.1f}  {flips:5d}  {math.degrees(max_edge):9.1f}  {'yes' if fell else 'no ':3s}   {best[2]}")

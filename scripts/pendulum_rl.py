"""Can PPO learn to carve with a double pendulum instead of the humanoid? Pure torch, no Isaac Sim (runs on the CPU).

    .venv-sim/bin/python scripts/pendulum_rl.py                 # train, then evaluate the mean policy
    .venv-sim/bin/python scripts/pendulum_rl.py --speed 5 9 --iters 300

The rider is a double pendulum in the board's roll plane on a free pivot at the edge (nothing at the board holds it up):
legs (0.9 m, 60 % of the mass, angle φ1 from the snow normal) and torso (0.7 m, 40 %, φ2) joined at the hip.
Actions (2, in [-1, 1]):
    edge   θ = φ1 - α: the ankle α is a servo, so the policy sets the edge angle directly, θ = 80° · a0
    hip    torque τ = 400 N·m · a1 between legs and torso (angulation: legs in, torso out)
The turn is the snow model's: on edge the sidecut steers onto R = 7.6 m · cos θ, every mass feels v² / R toward the
outside (capped by grip); flat (|θ| < 3°) there is no side force. Speed is fixed per episode.

Reward, as in the humanoid's edge-command run: track a timed edge command (heel <-> base <-> toe, a turn every
period / 2, 30 · exp(-((θ - θ*) / 12°)²), 0 while the edge is under 10°) and -50 for a fall (|φ1| or |φ2| > 80°).
Observation: φ1, φ2, ω1, ω2, θ, the command now / +0.3 s / +0.6 s, speed.
"""

import argparse
import math

import torch
import torch.nn as nn

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--speed", type=float, nargs=2, default=(5.0, 8.0), help="m/s, sampled per episode")
p.add_argument("--amp", type=float, default=45.0, help="edge command amplitude [deg]")
p.add_argument("--period", type=float, default=4.0, help="full S [s]")
p.add_argument("--grip", type=float, default=1.5, help="max side force [g]")
p.add_argument("--envs", type=int, default=1024)
p.add_argument("--iters", type=int, default=250)
p.add_argument("--seconds", type=float, default=12.0, help="episode length")
p.add_argument("--seed", type=int, default=0)
args = p.parse_args()
torch.manual_seed(args.seed)

G, R_SC = 9.81, 7.6
L1, L2, MASS = 0.9, 0.7, 65.0
M1, M2 = 0.6 * MASS, 0.4 * MASS
C1, C2 = L1 / 2, L2 / 2
I1, I2 = M1 * L1 ** 2 / 12, M2 * L2 ** 2 / 12
PHYS_DT, DECIM = 1 / 240, 8  # 30 Hz policy
EDGE_MAX, HIP_MAX = math.radians(80.0), 400.0
MIN_EDGE = math.radians(3.0)
FALL = math.radians(80.0)


class Env:
    def __init__(self, n):
        self.n = n
        self.s = torch.zeros(n, 4)
        self.t = torch.zeros(n)
        self.v = torch.zeros(n)
        self.phase = torch.zeros(n)
        self.theta = torch.zeros(n)
        self.reset(torch.ones(n, dtype=torch.bool))

    def command(self, ahead=0.0):
        arg = 2 * math.pi * (self.t + ahead) / args.period + self.phase
        return math.radians(args.amp) * (1.8 * torch.sin(arg)).clamp(-1, 1)

    def reset(self, m):
        k = int(m.sum())
        self.s[m] = torch.randn(k, 4) * 0.02
        self.t[m] = 0.0
        self.v[m] = args.speed[0] + (args.speed[1] - args.speed[0]) * torch.rand(k)
        self.phase[m] = math.pi * (torch.rand(k) < 0.5).float()
        self.theta[m] = 0.0

    def obs(self):
        p1, p2, w1, w2 = self.s.unbind(-1)
        return torch.stack([p1, p2, w1 / 5, w2 / 5, self.theta, self.command(), self.command(0.3), self.command(0.6),
                            self.v / 8], -1)

    def side(self, theta):
        a = self.v ** 2 / (R_SC * torch.cos(theta).clamp_min(0.17))
        a = a.clamp(max=args.grip * G)
        return torch.where(theta.abs() < MIN_EDGE, torch.zeros_like(a), -torch.sign(theta) * a)

    def deriv(self, s, tau, a):
        p1, p2, w1, w2 = s.unbind(-1)
        c = torch.cos(p1 - p2)
        m11 = I1 + M1 * C1 ** 2 + M2 * L1 ** 2
        m12 = M2 * L1 * C2 * c
        m22 = torch.full_like(p1, I2 + M2 * C2 ** 2)
        h1 = -M2 * L1 * C2 * torch.sin(p1 - p2) * w2 ** 2
        h2 = M2 * L1 * C2 * torch.sin(p1 - p2) * w1 ** 2
        g1 = (M1 * C1 + M2 * L1) * (G * torch.sin(p1) + a * torch.cos(p1))
        g2 = M2 * C2 * (G * torch.sin(p2) + a * torch.cos(p2))
        f1, f2 = g1 + h1 - tau, g2 + h2 + tau
        det = m11 * m22 - m12 ** 2
        acc1 = (m22 * f1 - m12 * f2) / det
        acc2 = (m11 * f2 - m12 * f1) / det
        return torch.stack([w1, w2, acc1, acc2], -1)

    def step(self, act):
        act = act.clamp(-1, 1)
        self.theta = EDGE_MAX * act[:, 0]
        tau = HIP_MAX * act[:, 1]
        a = self.side(self.theta)
        for _ in range(DECIM):  # RK2
            k1 = self.deriv(self.s, tau, a)
            self.s = self.s + PHYS_DT * self.deriv(self.s + 0.5 * PHYS_DT * k1, tau, a)
        self.t += PHYS_DT * DECIM
        cmd = self.command()
        edged = ((self.theta.abs() - math.radians(10)) / math.radians(10)).clamp(0, 1)
        r = 30.0 * torch.exp(-((self.theta - cmd) / math.radians(12)) ** 2) * edged * PHYS_DT * DECIM
        fell = (self.s[:, 0].abs() > FALL) | (self.s[:, 1].abs() > FALL) | ~torch.isfinite(self.s).all(-1)
        timeout = self.t >= args.seconds
        r = r - 50.0 * fell.float()
        done = fell | timeout
        info = (fell, timeout, (self.theta - cmd).abs(), self.theta.clone(), cmd.clone())
        self.s = torch.nan_to_num(self.s)
        self.reset(done)
        return self.obs(), r, done, timeout, info


class AC(nn.Module):
    def __init__(self, n_obs, n_act):
        super().__init__()
        mlp = lambda out: nn.Sequential(nn.Linear(n_obs, 128), nn.ELU(), nn.Linear(128, 128), nn.ELU(), nn.Linear(128, out))
        self.pi, self.vf = mlp(n_act), mlp(1)
        self.log_std = nn.Parameter(torch.full((n_act,), -0.5))

    def dist(self, o):
        return torch.distributions.Normal(self.pi(o), self.log_std.clamp(-3, 0).exp())


def train():
    env, ac = Env(args.envs), AC(9, 2)
    opt = torch.optim.Adam(ac.parameters(), lr=3e-4)
    T, gamma, lam = 96, 0.995, 0.98  # 96 steps = 3.2 s per rollout
    o = env.obs()
    for it in range(args.iters):
        buf = {k: [] for k in ("o", "a", "lp", "r", "d", "tout", "v")}
        falls = dones = 0
        err = 0.0
        for _ in range(T):
            with torch.no_grad():
                d = ac.dist(o)
                a = d.sample()
                buf["lp"].append(d.log_prob(a).sum(-1))
                buf["v"].append(ac.vf(o).squeeze(-1))
            buf["o"].append(o)
            buf["a"].append(a)
            o, r, done, tout, (fell, _, e, _, _) = env.step(a)
            buf["r"].append(r)
            buf["d"].append(done.float())
            buf["tout"].append(tout.float())
            falls += int(fell.sum())
            dones += int(done.sum())
            err += float(e.mean())
        with torch.no_grad():
            last_v = ac.vf(o).squeeze(-1)
        adv = torch.zeros(T, env.n)
        gae = torch.zeros(env.n)
        for t in reversed(range(T)):
            nv = last_v if t == T - 1 else buf["v"][t + 1]
            nonterm = 1.0 - buf["d"][t] + buf["tout"][t]  # time-outs bootstrap
            delta = buf["r"][t] + gamma * nv * nonterm - buf["v"][t]
            gae = delta + gamma * lam * (1.0 - buf["d"][t]) * gae
            adv[t] = gae
        ret = adv + torch.stack(buf["v"])
        O, A = torch.cat(buf["o"]), torch.cat(buf["a"])
        LP, ADV, RET = torch.cat(buf["lp"]), adv.flatten(), ret.flatten()
        ADV = (ADV - ADV.mean()) / (ADV.std() + 1e-8)
        for _ in range(5):
            for idx in torch.randperm(len(O)).split(len(O) // 8):
                d = ac.dist(O[idx])
                ratio = (d.log_prob(A[idx]).sum(-1) - LP[idx]).exp()
                pl = -torch.min(ratio * ADV[idx], ratio.clamp(0.8, 1.2) * ADV[idx]).mean()
                vl = (ac.vf(O[idx]).squeeze(-1) - RET[idx]).pow(2).mean()
                loss = pl + 0.5 * vl - 0.003 * d.entropy().sum(-1).mean()
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(ac.parameters(), 1.0)
                opt.step()
        if it % 10 == 0 or it == args.iters - 1:
            rate = falls / max(dones, 1)
            print(f"[prl] iter {it:4d}  reward/step {torch.stack(buf['r']).mean():6.3f}  falls/done {rate:4.2f}  "
                  f"|edge err| {math.degrees(err / T):5.1f}°  std {ac.log_std.exp().mean():.2f}", flush=True)
    return ac


def evaluate(ac):
    """Mean policy, one episode per env: time up, edge changes made, tracking error, edge range."""
    env = Env(256)
    up = torch.zeros(env.n)
    alive = torch.ones(env.n, dtype=torch.bool)
    flips = torch.zeros(env.n)
    side = torch.zeros(env.n)
    err = torch.zeros(env.n)
    hip_counter = torch.zeros(env.n)  # torso leaning against the legs while edged (angulation)
    trace = []
    o = env.obs()
    for _ in range(int(args.seconds * 30)):
        with torch.no_grad():
            a = ac.pi(o)
        p1, p2 = env.s[:, 0].clone(), env.s[:, 1].clone()
        o, r, done, tout, (fell, _, e, theta, cmd) = env.step(a)
        af = alive.float()
        up += af / 30
        err += af * e
        held = torch.sign(theta) * (theta.abs() > math.radians(15))
        flips += af * ((held != 0) & (side != 0) & (held != side)).float()
        side = torch.where(alive & (held != 0), held, side)
        hip_counter += af * ((theta.abs() > math.radians(20)) & (torch.sign(p2 - p1) == -torch.sign(p1))).float()
        trace.append((math.degrees(cmd[0]), math.degrees(theta[0]), math.degrees(p1[0]), math.degrees(p2[0])))
        alive = alive & ~fell & ~(tout & alive)
    steps = (up * 30).clamp_min(1)
    print(f"\n[prl] eval (mean policy, 256 envs, speed {args.speed[0]}-{args.speed[1]} m/s, edge ±{args.amp}°, "
          f"turn every {args.period / 2} s):")
    print(f"[prl]   up {up.mean():.1f} / {args.seconds:.0f} s, fell {(up < args.seconds - 0.1).float().mean():.0%}, "
          f"edge changes {flips.mean():.1f} (command: {2 * args.seconds / args.period - 1:.0f}), "
          f"|edge err| {math.degrees((err / steps).mean()):.1f}°, angulated {(hip_counter / steps).mean():.0%} of steps")
    print("[prl]   env 0, every 0.5 s:  t  cmd°  edge°  legs φ1°  torso φ2°")
    for k in range(0, len(trace), 15):
        c, th, a1, a2 = trace[k]
        print(f"[prl]   {k / 30:4.1f}  {c:5.0f}  {th:5.0f}  {a1:8.1f}  {a2:9.1f}")


if __name__ == "__main__":
    evaluate(train())

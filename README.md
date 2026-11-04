# learn2carve

### Demo: snowboard playground (plain RL, no demo clip yet)

<table>
  <tr><th></th><th>Front</th><th>Top</th></tr>
  <tr>
    <th>Carve</th>
    <td><img src="docs/demo/carve_front.gif" width="400" alt="Carve, front view: the rider tips the board on edge through S-turns"></td>
    <td><img src="docs/demo/carve_top.gif" width="400" alt="Carve, top view: a thin S line"></td>
  </tr>
  <tr>
    <th>Skid</th>
    <td><img src="docs/demo/skid_front.gif" width="400" alt="Skid, front view: the board pivots and slides through each turn"></td>
    <td><img src="docs/demo/skid_top.gif" width="400" alt="Skid, top view: a wide S"></td>
  </tr>
  <tr>
    <th>Side slip</th>
    <td><img src="docs/demo/sideslip_front.gif" width="400" alt="Side slip, front view: board across the fall line on the heel edge"></td>
    <td><img src="docs/demo/sideslip_top.gif" width="400" alt="Side slip, top view: a straight band as wide as the board is long"></td>
  </tr>
</table>

<em>Follow cameras on one evaluation run each, real time. Blue = the snow the board sweeps (wider = more sideways
slip). Recorded with <code>playground/snowboard/play.py &lt;run&gt; --record docs/demo/&lt;task&gt;</code>; see
<a href="playground/snowboard/README.md">playground/snowboard</a>.</em>

| Doc | What |
|---|---|
| [Overview](docs/OVERVIEW.md) | Pipeline, stages, layout, stack |
| [Humanoid model](docs/HUMANOID_MODEL.md) | Bodies, joints, drives, binding loop, stance |
| [Board model](docs/BOARD_MODEL.md) | Geometry, sidecut, bindings |
| [Snow model](docs/SNOW_MODEL.md) | Glide, sidecut steering, edge grip, validation |
| [RL training](docs/RL_TRAINING.md) | MDP, reward, PPO, learning curves |

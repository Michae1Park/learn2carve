# learn2carve

### Demo: snowboard playground (plain RL, no demo clip yet)

<table>
  <tr><th></th><th>Carve</th><th>Skid</th><th>Side slip</th></tr>
  <tr>
    <th>Front</th>
    <td><img src="docs/demo/carve_front.gif" width="280" alt="Carve, front view: the rider tips the board on edge through S-turns"></td>
    <td><img src="docs/demo/skid_front.gif" width="280" alt="Skid, front view: the board pivots and slides through each turn"></td>
    <td><img src="docs/demo/sideslip_front.gif" width="280" alt="Side slip, front view: board across the fall line on the heel edge"></td>
  </tr>
  <tr>
    <th>Top</th>
    <td><img src="docs/demo/carve_top.gif" width="280" alt="Carve, top view: a thin S line"></td>
    <td><img src="docs/demo/skid_top.gif" width="280" alt="Skid, top view: a wide S"></td>
    <td><img src="docs/demo/sideslip_top.gif" width="280" alt="Side slip, top view: a straight band as wide as the board is long"></td>
  </tr>
</table>

<em>Follow cameras on one evaluation run each, real time. Blue = the snow the board sweeps (wider = more sideways
slip). Recorded with <code>playground/snowboard/play.py &lt;run&gt; --record docs/demo/&lt;task&gt;</code>; see
<a href="playground/snowboard/README.md">playground/snowboard</a>.</em>

Pipeline, stages, layout and stack: [docs/OVERVIEW.md](docs/OVERVIEW.md).

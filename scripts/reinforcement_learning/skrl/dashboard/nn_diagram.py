"""Generate inline SVG for the VLP16 3-branch neural network architecture."""

from __future__ import annotations


def generate_nn_svg() -> str:
    """Return an SVG string for the VLP16 architecture diagram."""
    return """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1100 620" style="font-family: Inter, sans-serif;">
  <defs>
    <marker id="arrow" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
      <polygon points="0 0, 8 3, 0 6" fill="#7aa2f7"/>
    </marker>
    <filter id="glow">
      <feGaussianBlur stdDeviation="2" result="blur"/>
      <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
    </filter>
  </defs>

  <!-- Background -->
  <rect width="1100" height="620" rx="12" fill="#1a1b26"/>

  <!-- ═══ INPUT BLOCK (left) ═══ -->
  <text x="60" y="30" fill="#565f89" font-size="11" font-weight="600">INPUT: 139D Observation</text>

  <!-- Ego (4D) -->
  <rect x="20" y="45" width="120" height="50" rx="6" fill="#7aa2f7" opacity="0.15" stroke="#7aa2f7" stroke-width="1.5"/>
  <text x="80" y="65" fill="#7aa2f7" font-size="12" font-weight="600" text-anchor="middle">Ego</text>
  <text x="80" y="82" fill="#a9b1d6" font-size="10" text-anchor="middle">4D</text>

  <!-- Goal (2D) -->
  <rect x="20" y="105" width="120" height="50" rx="6" fill="#9ece6a" opacity="0.15" stroke="#9ece6a" stroke-width="1.5"/>
  <text x="80" y="125" fill="#9ece6a" font-size="12" font-weight="600" text-anchor="middle">Goal</text>
  <text x="80" y="142" fill="#a9b1d6" font-size="10" text-anchor="middle">2D</text>

  <!-- LiDAR (72D) -->
  <rect x="20" y="170" width="120" height="80" rx="6" fill="#ff9e64" opacity="0.15" stroke="#ff9e64" stroke-width="1.5"/>
  <text x="80" y="200" fill="#ff9e64" font-size="12" font-weight="600" text-anchor="middle">LiDAR</text>
  <text x="80" y="220" fill="#a9b1d6" font-size="10" text-anchor="middle">72D (5°/bin)</text>
  <text x="80" y="237" fill="#565f89" font-size="9" text-anchor="middle">VLP-16 min-pooled</text>

  <!-- Obstacles (60D) -->
  <rect x="20" y="265" width="120" height="80" rx="6" fill="#f7768e" opacity="0.15" stroke="#f7768e" stroke-width="1.5"/>
  <text x="80" y="295" fill="#f7768e" font-size="12" font-weight="600" text-anchor="middle">Obstacles</text>
  <text x="80" y="315" fill="#a9b1d6" font-size="10" text-anchor="middle">60D (10×6)</text>
  <text x="80" y="332" fill="#565f89" font-size="9" text-anchor="middle">body-frame + mask</text>

  <!-- Time (1D) -->
  <rect x="20" y="360" width="120" height="50" rx="6" fill="#565f89" opacity="0.25" stroke="#565f89" stroke-width="1.5"/>
  <text x="80" y="380" fill="#a9b1d6" font-size="12" font-weight="600" text-anchor="middle">Time</text>
  <text x="80" y="397" fill="#565f89" font-size="10" text-anchor="middle">1D</text>

  <!-- ═══ ROUTING ARROWS (input → branches) ═══ -->
  <!-- Ego+Goal+Time → State MLP -->
  <line x1="140" y1="70" x2="200" y2="480" stroke="#7aa2f7" stroke-width="1" stroke-dasharray="4,3" marker-end="url(#arrow)"/>
  <line x1="140" y1="130" x2="200" y2="485" stroke="#9ece6a" stroke-width="1" stroke-dasharray="4,3" marker-end="url(#arrow)"/>
  <line x1="140" y1="385" x2="200" y2="495" stroke="#565f89" stroke-width="1" stroke-dasharray="4,3" marker-end="url(#arrow)"/>

  <!-- LiDAR → LiDAR Conv1d -->
  <line x1="140" y1="210" x2="200" y2="95" stroke="#ff9e64" stroke-width="1.5" marker-end="url(#arrow)"/>

  <!-- Obstacles → Obstacle MLP -->
  <line x1="140" y1="305" x2="200" y2="290" stroke="#f7768e" stroke-width="1.5" marker-end="url(#arrow)"/>

  <!-- ═══ BRANCH 1: LiDAR Conv1d (top) ═══ -->
  <rect x="210" y="45" width="320" height="130" rx="8" fill="#292e42" stroke="#ff9e64" stroke-width="1.5"/>
  <text x="370" y="65" fill="#ff9e64" font-size="13" font-weight="700" text-anchor="middle">LiDAR Conv1d Branch</text>
  <text x="220" y="87" fill="#a9b1d6" font-size="10">Conv1d(1→32, k5) → ReLU</text>
  <text x="220" y="103" fill="#a9b1d6" font-size="10">Conv1d(32→64, k5, s2) → ReLU</text>
  <text x="220" y="119" fill="#a9b1d6" font-size="10">Conv1d(64→64, k3, s2) → ReLU</text>
  <text x="220" y="135" fill="#a9b1d6" font-size="10">AdaptiveMaxPool1d(1)</text>
  <text x="220" y="151" fill="#a9b1d6" font-size="10">Linear(64→64) + LayerNorm</text>
  <text x="220" y="167" fill="#ff9e64" font-size="11" font-weight="600">→ 64D</text>

  <!-- ═══ BRANCH 2: Obstacle MLP (middle) ═══ -->
  <rect x="210" y="195" width="320" height="120" rx="8" fill="#292e42" stroke="#f7768e" stroke-width="1.5"/>
  <text x="370" y="215" fill="#f7768e" font-size="13" font-weight="700" text-anchor="middle">Obstacle MLP Branch</text>
  <text x="220" y="237" fill="#a9b1d6" font-size="10">Per-object: Linear(6→32) → ReLU</text>
  <text x="220" y="253" fill="#a9b1d6" font-size="10">Per-object: Linear(32→32) → ReLU</text>
  <text x="220" y="269" fill="#a9b1d6" font-size="10">MaxPool over 10 objects</text>
  <text x="220" y="285" fill="#a9b1d6" font-size="10">LayerNorm(32)</text>
  <text x="220" y="303" fill="#f7768e" font-size="11" font-weight="600">→ 32D (permutation-invariant)</text>

  <!-- ═══ BRANCH 3: State MLP (bottom) ═══ -->
  <rect x="210" y="435" width="320" height="110" rx="8" fill="#292e42" stroke="#7aa2f7" stroke-width="1.5"/>
  <text x="370" y="455" fill="#7aa2f7" font-size="13" font-weight="700" text-anchor="middle">State MLP Branch</text>
  <text x="220" y="477" fill="#a9b1d6" font-size="10">Input: ego(4) + goal(2) + time(1) = 7D</text>
  <text x="220" y="493" fill="#a9b1d6" font-size="10">Linear(7→32) → ReLU</text>
  <text x="220" y="509" fill="#a9b1d6" font-size="10">Linear(32→32) → ReLU + LayerNorm</text>
  <text x="220" y="530" fill="#7aa2f7" font-size="11" font-weight="600">→ 32D</text>

  <!-- ═══ FUSION ═══ -->
  <line x1="530" y1="120" x2="580" y2="290" stroke="#ff9e64" stroke-width="1.5" marker-end="url(#arrow)"/>
  <line x1="530" y1="260" x2="580" y2="290" stroke="#f7768e" stroke-width="1.5" marker-end="url(#arrow)"/>
  <line x1="530" y1="490" x2="580" y2="300" stroke="#7aa2f7" stroke-width="1.5" marker-end="url(#arrow)"/>

  <rect x="590" y="260" width="140" height="70" rx="8" fill="#292e42" stroke="#bb9af7" stroke-width="2" filter="url(#glow)"/>
  <text x="660" y="285" fill="#bb9af7" font-size="13" font-weight="700" text-anchor="middle">Concatenate</text>
  <text x="660" y="305" fill="#a9b1d6" font-size="11" text-anchor="middle">[64 + 32 + 32]</text>
  <text x="660" y="322" fill="#bb9af7" font-size="12" font-weight="700" text-anchor="middle">= 128D</text>

  <!-- ═══ POLICY HEAD (top-right) ═══ -->
  <line x1="730" y1="280" x2="780" y2="160" stroke="#bb9af7" stroke-width="1.5" marker-end="url(#arrow)"/>

  <rect x="790" y="95" width="280" height="130" rx="8" fill="#292e42" stroke="#9ece6a" stroke-width="1.5"/>
  <text x="930" y="118" fill="#9ece6a" font-size="13" font-weight="700" text-anchor="middle">Policy Head (Actor)</text>
  <text x="800" y="140" fill="#a9b1d6" font-size="10">Linear(128→128) → ReLU</text>
  <text x="800" y="158" fill="#a9b1d6" font-size="10">Linear(128→38)</text>
  <text x="800" y="178" fill="#9ece6a" font-size="11" font-weight="600">→ 38 logits</text>
  <text x="800" y="196" fill="#565f89" font-size="10">MultiDiscrete([19, 19])</text>
  <text x="800" y="214" fill="#565f89" font-size="9">linear_accel(19) + angular_vel(19)</text>

  <!-- ═══ VALUE HEAD (bottom-right) ═══ -->
  <line x1="730" y1="310" x2="780" y2="400" stroke="#bb9af7" stroke-width="1.5" marker-end="url(#arrow)"/>

  <rect x="790" y="350" width="280" height="140" rx="8" fill="#292e42" stroke="#7dcfff" stroke-width="1.5"/>
  <text x="930" y="373" fill="#7dcfff" font-size="13" font-weight="700" text-anchor="middle">Value Head (Critic)</text>
  <text x="800" y="395" fill="#a9b1d6" font-size="10">Linear(128→64) → ReLU</text>
  <text x="800" y="413" fill="#a9b1d6" font-size="10">Linear(64→32) → ReLU</text>
  <text x="800" y="431" fill="#a9b1d6" font-size="10">Linear(32→1)</text>
  <text x="800" y="451" fill="#7dcfff" font-size="11" font-weight="600">→ Scalar V(s)</text>
  <text x="800" y="471" fill="#565f89" font-size="9">Init: orthogonal(0.01), bias=-8.0</text>
  <text x="800" y="483" fill="#565f89" font-size="9">No privileged info (v2 symmetric)</text>

  <!-- ═══ LEGEND ═══ -->
  <rect x="20" y="560" width="1060" height="50" rx="6" fill="#24283b"/>
  <text x="40" y="580" fill="#565f89" font-size="10" font-weight="600">LEGEND:</text>
  <rect x="110" y="570" width="14" height="14" rx="2" fill="#7aa2f7" opacity="0.4"/>
  <text x="130" y="582" fill="#a9b1d6" font-size="10">Ego/State</text>
  <rect x="210" y="570" width="14" height="14" rx="2" fill="#9ece6a" opacity="0.4"/>
  <text x="230" y="582" fill="#a9b1d6" font-size="10">Goal/Policy</text>
  <rect x="330" y="570" width="14" height="14" rx="2" fill="#ff9e64" opacity="0.4"/>
  <text x="350" y="582" fill="#a9b1d6" font-size="10">LiDAR</text>
  <rect x="420" y="570" width="14" height="14" rx="2" fill="#f7768e" opacity="0.4"/>
  <text x="440" y="582" fill="#a9b1d6" font-size="10">Obstacles</text>
  <rect x="530" y="570" width="14" height="14" rx="2" fill="#bb9af7" opacity="0.4"/>
  <text x="550" y="582" fill="#a9b1d6" font-size="10">Fusion</text>
  <rect x="620" y="570" width="14" height="14" rx="2" fill="#7dcfff" opacity="0.4"/>
  <text x="640" y="582" fill="#a9b1d6" font-size="10">Value</text>
  <text x="730" y="582" fill="#565f89" font-size="10">Total params: ~35,600</text>
  <text x="930" y="582" fill="#565f89" font-size="10">Activation: ReLU | Norm: LayerNorm</text>
</svg>"""

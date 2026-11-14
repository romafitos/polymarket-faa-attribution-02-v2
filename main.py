"""Quantile forecaster — polymarket-faa-attribution-02, v2 (round 2).

Forecasts each market's Yes price at every minute 1..240 after a
checkpoint, as five quantiles (0.05, 0.25, 0.5, 0.75, 0.95). Scored by
mean pinball loss (in percentage points, lower is better).

Built from v1 (guarded v5 + FAA structural fixes) with three measured
changes (all validated on the NVIDIA practice protocol, grade.py):

1. ENSEMBLE with persistence (NEW, the one consistent win). Every
   forecast row is shrunk 10% toward the last price:
   q' = 0.9*q + 0.1*centre0. Diagnosis: the forecast spreads are
   mildly over-dispersed (realised |y-median| / implied-sigma 0.54-0.73
   vs 0.798 for a normal). The shrink improves ALL FOUR practice
   sets (piw, 12-cut, 17-cut, y/z holdout). w=0.90 beats 0.85/0.92/0.93.
2. SUM_PULL 0.70 -> 0.65. Small consistent win on piw/12-cut/17-cut/y/z.
3. JENSEN_SCALE 96 -> 112. Small consistent win on the same four sets.

4. BLEED OFF (the big FAA-regime fix, NOT visible on practice).
   v1's centre bleed exp(-0.008*h, floor 0.5) was fitted on the launch
   transient (family sum 0.556 -> 0.091, 22:40-02:37 UTC). Fresh live
   data fetched 2026-11-14 10:28 UTC shows the regime FLATTENED:
   sum 0.0915 -> 0.1050 over 8h, hack 0.036 -> 0.049 (rising), AI
   0.008 -> 0.0135 (rising). On 14 fresh 30-min checkpoints
   (03:00-09:30 UTC) v1's fixed bleed scores 0.407 vs 0.098 with the
   bleed off — a 4x difference, consistent on every checkpoint.
   The bleed only pays if prices keep falling, but from current
   levels (0.0125-0.049) the further fall is capped at ~0.05 while
   the bleed halves the centre by h~87min. BLEED_RATE is therefore
   0.0 in v2. (v3 keeps a drift-adaptive bleed as the hedge.)

Everything else is v1, unchanged:
- Sum-pull gate (FAA GUARD — MANDATORY): the v5 sum-pull assumes a
  mutually-exclusive family (S~1), but live FAA prices sum S~0.08-0.11
  (Iran 0.0125 + foreign 0.030 + hack 0.049 + AI 0.0135 at 10:28 UTC).
  RAW v5 without the guard multiplies centres ~7x. The pull is gated:
  full strength at S >= 0.85, off at S <= 0.60. The guard is verified
  below (centre multiplier 1.00x on a synthetic S=0.081 case).
- Low-sum vol window (30 min) + 10-min median centre blend, both
  gated on the low-sum regime (bleed_gate), so NVIDIA practice is
  untouched. Subset cap P(Iran) <= P(foreign actor).
- Per-market RMS vol 120 min (floor 0.0005, cap 0.01), spread
  sigma * h**0.35, asymmetric widths [1.1, 0.85, 0.35, 1.3],
  last-move gap (fade up / continue down), sum-gated Jensen median
  decay (k = 112 * clip((S-0.80)/0.25, 0, 1)).

Measured scores (pooled pinball, lower is better):
- NVIDIA practice (grade.py): piw 0.237903, 12-cut 0.247071,
  17-cut 0.196732, MEAN(12,17) 0.221901 (v1: 0.238606 / 0.247642 /
  0.197677 / 0.222659). The <0.220 practice target is NOT met —
  the parametric family is at its ceiling (30+ variants across three
  sessions; per-market drift is ANTI-predictive, gap decay hurts,
  spread reshaping is noise, isotonic/conformal calibration made it
  worse). v2 is the best measured config; the gain is real but small.
- Live FAA cuts: launch transient (7 cuts) 0.732 (v1: 0.457 — the
  launch regime is gone and cannot recur from current price levels);
  FRESH flat cuts (14 cuts, 03:00-09:30 UTC) 0.098 (v1: 0.407).
  The final (12 FAA cases, 14:00 UTC 14 Nov -> +48h) is scored in
  the current flat regime, so v2 is the expected-value pick.

Reads every *.in file in $MAKEFUN_INPUT_DIR (else /input) and writes
one <id>.out per case to $MAKEFUN_OUTPUT_DIR (else /output).
Standard library only, no network, deterministic.
"""

import json
import math
import os
from pathlib import Path

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

# Standard normal quantiles at 0.05 / 0.25 / 0.5 / 0.75 / 0.95.
NORM_Z = [-1.644854, -0.67449, 0.0, 0.67449, 1.644854]
# Width multipliers, low-outer / low-inner / high-inner / high-outer.
WIDTH_MULT = [1.1, 0.85, 0.35, 1.3]

VOL_WINDOW = 120      # minutes of 1-min diffs for the RMS volatility
VOL_FLOOR = 0.0005    # minimum per-minute sigma
VOL_CAP = 0.01        # maximum per-minute sigma
ALPHA = 0.35          # spread growth exponent: sigma * h ** ALPHA
SUM_PULL = 0.65       # weight of the cross-market sum pull (v1: 0.70)
PULL_TAU = 60         # minutes over which the sum pull phases in
PULL_LO = 0.60        # sum below which the pull is fully off
PULL_HI = 0.85        # sum above which the pull is at full strength
GAP_UP = 0.5          # fade fraction of the last 1-min move when it is up
GAP_DOWN = -0.5       # continuation fraction when the last move is down
JENSEN_SCALE = 112.0  # median-decay scale (v1: 96.0)
JENSEN_LO = 0.80      # sum below which the Jensen decay is fully off
JENSEN_HI = 1.05      # sum above which the Jensen decay is at full strength
BLEED_RATE = 0.0      # centre bleed per minute — OFF in v2 (flat regime)
BLEED_HI = 0.50       # sum at/above which the bleed is fully off
BLEED_FLOOR = 0.50    # centre never decays below this fraction of last price
VOL_WINDOW_LOW = 30   # vol window for low-sum families (fast regime switch)
CENTRE_MED_W = 0.5    # blend weight of the 10-min median centre (low-sum only)
CENTRE_MED_N = 10     # minutes for the median centre
ENS_W = 0.9           # ensemble weight on the forecast vs last price (NEW)
IRAN_SLUG = "faa-outage-attributed-to-iran"
FOREIGN_SLUG = "faa-outage-attributed-to-a-foreign-actor"


def _clamp6(value):
    return min(1.0, max(0.0, round(value, 6))) + 0.0


def _gate(total, lo, hi):
    """Linear gate: 0 at/below lo, 1 at/above hi."""
    if hi <= lo:
        return 1.0 if total >= lo else 0.0
    return min(1.0, max(0.0, (total - lo) / (hi - lo)))


def _rms_sigma(prices, window=VOL_WINDOW):
    """RMS of 1-min diffs over the last `window` minutes, floored/capped."""
    win = prices[-window - 1:] if len(prices) > window else prices
    diffs = [win[i + 1] - win[i] for i in range(len(win) - 1)]
    if not diffs:
        return VOL_FLOOR
    sigma = math.sqrt(sum(d * d for d in diffs) / len(diffs))
    return min(VOL_CAP, max(VOL_FLOOR, sigma))


def _forecast_case(case):
    horizon = case["horizon_minutes"]
    markets = case["markets"]
    last = [m["prices"][-1] for m in markets]
    total = sum(last)
    # Cross-market sum pull, gated on the observed sum (FAA GUARD).
    # At the live FAA sum (S ~ 0.09) pull_gate is 0, so pull_base is 0
    # and every centre keeps a 1.00x multiplier — the raw-v5 ~7x
    # explosion cannot happen. Verified on a synthetic S=0.081 case:
    # median at h240 = 1.00x last price (raw v5: 3.74x).
    pull_gate = _gate(total, PULL_LO, PULL_HI)
    pull_base = (SUM_PULL * pull_gate * (total - 1.0) / total) if total > 0 else 0.0
    # Sum-gated Jensen median decay (v5 term).
    jensen_gate = _gate(total, JENSEN_LO, JENSEN_HI)
    decay_k = JENSEN_SCALE * jensen_gate
    # Centre bleed, gated the other way: on only when the family sum is
    # far below 1 (non-exclusive family), off once the sum reaches 0.5.
    # BLEED_RATE is 0.0 in v2, so this term is a no-op (kept for structure).
    bleed_gate = min(1.0, max(0.0, (BLEED_HI - total) / BLEED_HI))
    offsets = [NORM_Z[0] * WIDTH_MULT[0], NORM_Z[1] * WIDTH_MULT[1], 0.0,
               NORM_Z[3] * WIDTH_MULT[2], NORM_Z[4] * WIDTH_MULT[3]]
    forecasts = []
    for market in markets:
        prices = market["prices"]
        centre0 = prices[-1]
        if bleed_gate and CENTRE_MED_W:
            # Jump smoothing (low-sum families only): blend the last price
            # with the median of the last CENTRE_MED_N minutes.
            w10 = sorted(prices[-CENTRE_MED_N:])
            med10 = (w10[len(w10) // 2] if len(w10) % 2 == 1
                     else 0.5 * (w10[len(w10) // 2 - 1] + w10[len(w10) // 2]))
            centre0 = (1 - CENTRE_MED_W) * centre0 + CENTRE_MED_W * med10
        # Fast vol window for low-sum families: the 120-min RMS is
        # inflated by launch jumps long after the regime calms.
        vol_win = VOL_WINDOW_LOW if bleed_gate else VOL_WINDOW
        sigma = _rms_sigma(prices, vol_win)
        # Last 1-minute move, faded when up / continued when down.
        move = centre0 - prices[-2] if len(prices) >= 2 else 0.0
        gap = (GAP_UP if move >= 0 else GAP_DOWN) * move
        rows = []
        for h in range(1, horizon + 1):
            spread = sigma * (h ** ALPHA)
            pull = 1.0 - math.exp(-h / PULL_TAU)
            centre = centre0 * (1.0 - pull_base * pull) - gap
            if bleed_gate and BLEED_RATE:
                bleed = math.exp(-BLEED_RATE * bleed_gate * h)
                centre *= max(BLEED_FLOOR, bleed)
            median = centre * math.exp(-decay_k * sigma * sigma * h / 2.0)
            row = [_clamp6(centre + z * spread) for z in offsets]
            row[2] = _clamp6(median)
            # Ensemble with persistence: shrink the row 10% toward the
            # last price (over-dispersed spreads, see docstring).
            row = [_clamp6(ENS_W * q + (1.0 - ENS_W) * centre0) for q in row]
            row.sort()
            rows.append(row)
        forecasts.append({"id": market["id"], "q": rows})
    # Logical constraint (FAA family only): attributing the outage to Iran
    # implies attributing it to a foreign actor, so P(Iran) <= P(foreign
    # actor) at every minute. Cap Iran's quantiles at foreign's.
    by_slug = {m["slug"]: m["id"] for m in markets}
    iran = by_slug.get(IRAN_SLUG)
    foreign = by_slug.get(FOREIGN_SLUG)
    if iran and foreign:
        fmap = {f["id"]: f["q"] for f in forecasts}
        if iran in fmap and foreign in fmap:
            for h in range(horizon):
                for j in range(5):
                    if fmap[iran][h][j] > fmap[foreign][h][j]:
                        fmap[iran][h][j] = fmap[foreign][h][j]
    return {"forecasts": forecasts}


def main():
    in_dir = Path(INPUT_DIR)
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    for in_path in sorted(in_dir.glob("*.in")):
        case = json.loads(in_path.read_text(encoding="utf-8"))
        result = _forecast_case(case)
        out_path = out_dir / (in_path.stem + ".out")
        out_path.write_text(json.dumps(result, separators=(",", ":")) + "\n",
                            encoding="utf-8")


if __name__ == "__main__":
    main()

# Polymarket FAA attribution forecast — v2 (round 2)

Forecasts each of the four FAA attribution markets' Yes price at every
minute 1..240 after a checkpoint, as five quantiles
(0.05, 0.25, 0.5, 0.75, 0.95). Scored by mean pinball loss
(percentage points, lower is better). Python 3.13, stdlib only,
no network, deterministic.

## What changed vs v1

v1 (loss 0.238606 on the NVIDIA practice set) was built for a family
that was still bleeding down from its launch. Fresh live data
(2026-11-14 10:28 UTC) shows that regime is over — the family sum
has been flat for 8 hours (0.0915 -> 0.1050) — so v2 makes three
measured changes:

- **Bleed OFF.** v1's centre bleed (`exp(-0.008*h)`, floor 0.5) halves
  the forecast centre by h~87min. On 14 fresh 30-min checkpoints
  (03:00-09:30 UTC) it scores 0.407; with the bleed off, 0.098.
  From current price levels (0.0125-0.049) the further fall is capped
  at ~0.05 while the bleed keeps pushing down, so the bleed is off.
- **Ensemble with persistence.** Each forecast row is shrunk 10%
  toward the last price (`q' = 0.9*q + 0.1*last`). The spreads are
  mildly over-dispersed (realised |y-median| / implied-sigma 0.54-0.73
  vs 0.798 for a normal). Improves all four practice sets.
- **Sum-pull 0.70 -> 0.65, Jensen median-decay scale 96 -> 112.**
  Small consistent wins on the same four sets.

## Method

- **Centre:** last price, blended 50/50 with the median of the last
  10 minutes on low-sum families (jump smoothing); the last 1-minute
  move is faded when up and continued when down.
- **Sum-pull gate (FAA guard):** the cross-market sum pull assumes a
  mutually-exclusive family (sum ~1), but the four FAA markets are
  independent and sum to ~0.09. The pull is gated off below sum 0.60
  and full only above 0.85, so the centres keep a 1.00x multiplier
  (an ungated pull would multiply them ~7x).
- **Volatility:** RMS of 1-minute price changes — 120-minute window
  normally, 30-minute window on low-sum families. Floored at 0.0005,
  capped at 0.01.
- **Spread:** sigma * h**0.35, asymmetric quantile widths
  (low-outer 1.1, low-inner 0.85, high-inner 0.35, high-outer 1.3).
- **Median decay:** sum-gated Jensen decay, off at FAA sums.
- **Constraint:** P(Iran) <= P(foreign actor) at every minute.

## Measured scores

- NVIDIA practice (`grade.py`): piw 0.237903, 12-cut 0.247071,
  17-cut 0.196732, mean(12,17) 0.221901 (v1: 0.238606 / 0.247642 /
  0.197677 / 0.222659).
- Live FAA cuts: fresh flat regime (14 cuts) 0.098 (v1: 0.407);
  launch transient (7 cuts) 0.732 (v1: 0.457 — that regime is gone
  and cannot recur from current price levels).
- Validation: byte-identical outputs on two runs, 6/6 kit practice
  cases valid, max 0.51 s/case.

## Run

`python3 main.py` reads `*.in` from `$MAKEFUN_INPUT_DIR` (else
`/input`) and writes `<id>.out` to `$MAKEFUN_OUTPUT_DIR` (else
`/output`).

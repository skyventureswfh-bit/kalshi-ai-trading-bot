from pathlib import Path

path = Path("src/jobs/live_trade.py")
text = path.read_text(encoding="utf-8")

old = '''            "The market price is usually close to fair: if your estimate is within ~5 cents of the "
            "midpoint, there is no edge â€” use WATCH or SKIP.\\n"
            "Kalshi taker fees are about 0.07 x P x (1-P) per contract (~1.75c at mid prices); "
            "your edge must clearly exceed fees after entry at the ask.\\n"
            "Trade only when liquidity, catalyst, and edge are all present. Use QUICK_FLIP only for sub-30-minute holds.\\n"
'''

new = '''            "Estimate fair value independently from the current market price. Do not automatically "
            "downgrade a candidate to WATCH or SKIP merely because the estimated edge is modest. "
            "If the evidence supports a positive actionable edge, return TRADE and let the downstream "
            "deterministic EV/risk gates decide whether that edge is sufficient after fees.\\n"
            "Kalshi taker fees are about 0.07 x P x (1-P) per contract (~1.75c at mid prices); "
            "include realistic entry economics in your estimate, but do not duplicate the final EV gate.\\n"
            "Require a real thesis, usable liquidity, and an actionable time window. Use QUICK_FLIP only for sub-30-minute holds.\\n"
'''

count = text.count(old)
if count != 1:
    raise SystemExit(f"Expected exactly one specialist prompt block, found {count}")

path.write_text(text.replace(old, new, 1), encoding="utf-8")
print("Participation gate prompt updated exactly once.")

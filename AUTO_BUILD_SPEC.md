# Beast Auto — Build Contract

## Protected baseline
- Manual Beast remains the active trading system.
- Do not modify the working Manual flow while Auto is being built.
- Auto work happens on a separate branch created from the known-good Manual baseline.

## Auto adds only
1. Replace the final manual-approval hold with controlled order submission.
2. Preserve every existing safety gate before execution.
3. Add a hard kill switch that blocks all new orders immediately.
4. Add strict per-trade and session loss/exposure caps.
5. Fail closed on stale data, API uncertainty, auth failure, order-state ambiguity, or safety-state uncertainty.
6. Log every proposed, blocked, submitted, filled, cancelled, and failed order path.

## Must not change
- Scout/specialist/final-synth decision pipeline.
- Existing EV/confidence/liquidity/guardrail logic unless a failing test proves a required change.
- Manual Beast branch or its running process.
- Dashboard/cockpit design.
- Unrelated indicators, visual polish, or feature additions.

## Acceptance tests before any live rollout
- Auto can be enabled/disabled without restarting the whole application.
- Kill switch prevents all new submissions.
- Order submission requires every existing safety gate to pass.
- Duplicate-order protection prevents a second order for the same intent/market.
- Stale market data blocks execution.
- API timeout/unknown order state fails closed and does not blindly retry into a duplicate position.
- Exposure cap blocks oversized or excess concurrent positions.
- Loss cap halts new entries when reached.
- Restart/recovery does not forget an existing open/pending order.
- Full audit log identifies why each candidate was blocked or executed.
- Manual Beast behavior remains unchanged.

## Rollout
1. Tests only.
2. Dry/shadow validation against live market data.
3. Tiny live canary with the smallest practical exposure.
4. Expand only after clean observed behavior.

## Team flow
Sage (architect/orchestrator) -> Builder -> independent review -> Sage acceptance -> Jeff go/no-go.

No cockpit redesign, Coderick work, or extra features until the Auto acceptance tests pass.

# BEAST STATE

## KNOWN GOOD
- Project: Beast Manual
- Repo: skyventureswfh-bit/kalshi-ai-trading-bot
- Manual baseline branch: beast-manual-baseline
- Protected source baseline starts from clean commit e39de90
- Running laptop copy is the operational reference until the clean telemetry fix is synced back to GitHub

## RUNNING NOW
- Dashboard: running
- Hunter: continuous manual-approval mode
- Real automatic order execution: hard-blocked
- Approval page: http://localhost:3000/live-trade/approval

## PROVEN
- End-to-end candidate pipeline confirmed
- Genuine pending candidates confirmed
- Worker telemetry observed as completed after the local repair
- Hunter can stop cleanly with Ctrl+C
- No real order was sent during validation

## CURRENT ISSUE
- GitHub main commit 20f5c50 contains encoding damage and must not be treated as the known-good baseline
- The local working copy is repaired and running, but that exact repaired source has not yet been synced to this branch

## NEXT ACTION
- Sync only the clean local telemetry repair into beast-manual-baseline without touching the running Beast
- Verify the diff contains only the intended telemetry logic change and no mojibake
- Then freeze/tag Beast Manual and branch Beast Auto from that verified baseline

## PROJECT CONTROL RULE
Current machine evidence outranks historical warnings. Do not reopen known-good work unless current evidence proves it is broken.

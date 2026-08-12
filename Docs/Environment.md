# Environment

`risk/game/environment.py` is the rules-engine entry point. It owns the
mutable `State`, exposes `legal_actions()` and `step(action)`, validates every
action, advances phases and turns, calculates rewards, and determines game
completion.

## Card-trade flow

Every turn begins in `TRADE_IN`. A player with five or more cards must trade
before reinforcement; otherwise a skip action advances to reinforcement.

An ordinary first-conquest card that changes a hand from four to five does not
interrupt the current attack turn. The player completes occupy/fortify and
trades at the beginning of their next turn. An elimination is different: if
the transferred cards leave the attacker with five or more cards,
`pending_attack` preserves the conquest while required trades and their
reinforcement placement complete before `OCCUPY` resumes.

## Fortify candidates

`_legal_fortify(...)` emits one skip action and, for every reachable owned
source/destination pair, the deduplicated transfer amounts `1`, half, and the
maximum (`armies_from - 1`). Directly submitted actions remain valid for every
positive amount through that maximum; bucketing limits only the candidates
offered to an agent.

## Related code

- `risk/game/state.py`: mutable game state and `PendingAttack`.
- `risk/game/actions.py`: action data and structural validation.
- `risk/learning/reward.py`: learner reward calculations called by `step()`.
- `Temp/tests/test_environment.py`: deterministic rules coverage.

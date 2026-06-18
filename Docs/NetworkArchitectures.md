# Network architectures — DQN × PPO, injection × lookup (plan)

Planning doc for the network(s) that sit on top of
[`GraphAdapter`](GraphAdapter.md) and [`Action.md`](Action.md) to choose
actions. Nothing here is implemented yet — no `risk/learning/encoder.py`,
`q_network.py`, or `policy_network.py` exist. This supersedes the earlier
split into `QNetwork.md` (DQN) and `PPOPolicy.md` (PPO) — those covered one
combination each and would have repeated the shared structure twice; this
doc covers all four combinations against one shared foundation instead.

---

## Two independent axes, four nets

Two separate design questions turned out to be orthogonal:

1. **Algorithm** — DQN (`Q(s,a)`, off-policy, replay buffer, target network)
   vs. PPO (`π(a|s)` + `V(s)`, on-policy, rollout buffer, GAE + clipped
   surrogate).
2. **How the action enters the network** — **inject** it into the graph
   before the GNN runs (modify `x`/`edge_attr` per candidate, one GNN pass
   per candidate), vs. **look it up** from one GNN pass over the
   unmodified state graph (read `H[t1]`/`H[t2]` after the fact, one GNN
   pass per decision, regardless of how many candidates exist).

Neither axis implies the other — the algorithm decides what the output
scalar(s) mean and how training works; the action representation decides
how a candidate gets *into* the network and at what cost. That's four nets:

| | **Inject** | **Lookup** |
|---|---|---|
| **DQN** | **Net A** — `Q(s,a)` from a per-candidate modified graph | **Net B** — `Q(s,a)` from one shared graph + embedding lookup |
| **PPO** | **Net C** — `logit(s,a)` from a per-candidate modified graph, `V(s)` from the base graph | **Net D** — `logit(s,a)` and `V(s)` both from one shared graph |

The plan is to build and train all four against the same self-play setup
and find out empirically which fits Risk best, rather than deciding by
argument — see "Experiment plan" at the end.

---

## Shared foundation (all four nets)

This is the structure that would otherwise be duplicated four times. Build
it once, in its own module(s), and have all four nets import it.

### Base graph — unchanged, from `GraphAdapter`

`Data(x=[42,13], edge_index=[2,166], u=[1,33])`, exactly as documented in
[GraphAdapter.md](GraphAdapter.md). None of the four nets touch
`GraphAdapter` itself — it stays the action-independent state encoding.

**One addition all four nets benefit from:** append a `num_legal_actions`
scalar (raw count, or `log1p`-scaled) to `u`. This was a real gap in the
lookup-based design specifically — a per-candidate head reading `H[t1]`/
`H[t2]` has no way to know "is this one of 3 options or one of 40" unless
that's surfaced somewhere global — but since it lives in `u`, every net
gets it for free (pooled into the critic/Q-head for Nets A-C, and available
to Net D's lookup heads via the same `u` they already read). Goes in
`GraphAdapter._global_features`, sourced from `len(env.legal_actions())` at
adapter-call time — a small, deliberate exception to `GraphAdapter`'s
"per-state, not per-action" framing, justified because it's a *property of
the state* (how open the position is) rather than of any one candidate.

### `Encoder` — one shared module, `risk/learning/encoder.py`

```python
class Encoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, edge_dim, n_layers=4):
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.convs = nn.ModuleList([
            TransformerConv(hidden_dim, hidden_dim, edge_dim=edge_dim)
            for _ in range(n_layers)
        ])

    def forward(self, x, edge_index, edge_attr):
        h = self.input_proj(x)
        for conv in self.convs:
            h = h + F.relu(conv(h, edge_index, edge_attr))   # residual
        return h   # [num_nodes, hidden_dim]
```

Identical module for all four nets. What differs is only what gets passed
in:

- **Inject (A, C):** `edge_attr` carries the real per-candidate marker for
  `ATTACK` (`[selected_attack, dice_count]`, zero on every other edge —
  see "Edge injection" below); `x` carries the per-candidate node
  perturbation for `REINFORCE_PLACE`/`OCCUPY`/`FORTIFY`. Called once per
  *candidate* (batched via `Batch.from_data_list`).
- **Lookup (B, D):** `edge_attr` is zero-filled (or omitted, `edge_dim=0`)
  and `x` is the unmodified base graph's. Called once per *decision*.
  `TransformerConv`'s attention still runs — it just has nothing from the
  edge term to contribute, degrading to plain neighbor-attention over `h`,
  not a no-op.

One encoder, reused, regardless of which net is being trained — this was
already the settled answer to "do we need two encoders for attack vs.
other stages" (we don't; zero `edge_attr` degrades gracefully) and it
applies the same way to "do we need different encoders for injection vs.
lookup" (we don't; the encoder doesn't know or care where its `x`/`edge_attr`
came from).

### Action injection (used by Nets A, C only) — `CandidateGraphBuilder`

For a candidate `AttackAction(from=A, to=B, dice=d)`:

```python
edge_attr.shape = [166, 2]                  # [selected_attack, dice_count]
edge_attr[index_of(A -> B)] = [1, d / MAX_ATTACK_DICE]
edge_attr[index_of(B -> A)] = [0, 0]        # reverse edge unmarked -> direction
# every other edge: [0, 0]
```

For `REINFORCE_PLACE` / `OCCUPY` / `FORTIFY`, the affected territory row(s)
in a **copy** of `x` get a perturbation (open question: write into the
existing army-count column directly, or add a parallel `proposed_delta`
column so the signal stays legible to attention rather than pre-merged
into the real count — lean toward the parallel column, revisit
empirically). `StopAttackAction`/skip-`FortifyAction` are the unmodified
base graph itself — no perturbation, nothing to encode.

`edge_index` never changes per candidate, only `edge_attr`/`x` do — and
`topology.edge_index()`'s row order is stable (`BoardTopology` sorts once
at load time), so `index_of(A -> B)` can be precomputed once, not searched
per candidate.

`CandidateGraphBuilder` never mutates the base `Data` — every candidate
needs its own unmodified copy to diverge from.

### Action lookup (used by Nets B, D only) — embedding heads

For a candidate `(stage, t1, t2, n)` (from the already-implemented
`Action.dqn_index()` / `ActionEncoder`, see [Action.md](Action.md)):

```python
def head_input(stage, t1, t2, n, H, none_vec):
    h1 = none_vec if t1 == Action.NONE_INDEX else H[t1]
    h2 = none_vec if t2 == Action.NONE_INDEX else H[t2]
    return torch.cat([h1, h2, encode_n(stage, n)])
```

`none_vec` is one learned `nn.Parameter([hidden_dim])`, shared across every
stage's sentinel case (`StopAttackAction`, skip-`FortifyAction`) — the
`-1 -> fixed learned "none" vector` idea from `Action.md`. `TRADE_IN`
doesn't read `H` at all regardless of net — its `t1`/`t2`/`n` are
card-hand-slot indices, so its head reads `nn.Embedding(MAX_CARDS_IN_HAND,
d)` lookups concatenated with pooled global context instead.

### Per-stage heads (all four nets)

Five small heads keyed by `ActionStage` (`TRADE_IN, REINFORCE_PLACE,
ATTACK, OCCUPY, FORTIFY`), same reasoning in all four nets: the stages are
different decisions with different `n`-semantics and value scales, so one
head per stage avoids forcing a shared tail to disambiguate that
implicitly. What differs per net is only the **input** to each head
(pooled candidate-graph embedding for A/C, `head_input(...)` lookup for
B/D) and the **output's meaning** (`Q` for A/B, `logit` for C/D).

```python
def make_head(in_dim: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(in_dim, 256), nn.ReLU(),
        nn.Linear(256, 128), nn.ReLU(),
        nn.Linear(128, 1),
    )

heads = nn.ModuleDict({
    "REINFORCE_PLACE": make_head(in_dim),
    "ATTACK":          make_head(in_dim),
    "OCCUPY":          make_head(in_dim),
    "FORTIFY":         make_head(in_dim),
    "TRADE_IN":        make_head(in_dim + 3 * card_embed_dim),
})
```

Mixed-stage batches (`TRADE_IN` + `REINFORCE_PLACE` candidates can appear
together during `Phase.REINFORCE` — `Action.md` already notes this): split
candidates by stage, run each group through its own head, merge the
per-candidate scalars back into `legal_actions()`'s original order before
the final `argmax` (A/B) or `softmax` (C/D).

### Pooling (used wherever a whole-graph scalar is needed)

```python
g = torch.cat([global_mean_pool(H), global_max_pool(H), u], dim=-1)
```

`u` concatenated in after pooling, not fed through the GNN — it's already
global (phase, budget, eliminations, now `num_legal_actions`), nothing for
message passing to refine, and pure node pooling alone is blind to it.

---

## Net A — DQN + injection

```
base Data(state)
   │  + candidate action (per legal candidate)
   ▼
CandidateGraphBuilder  ──▶  N modified Data graphs
   ▼
Batch.from_data_list
   ▼
Encoder (batched)      ──▶  H per graph
   ▼
pool(H, u)              ──▶  g  [N, g_dim]
   ▼
heads[stage](g)          ──▶  Q(s, a)  [N]
   ▼
argmax -> chosen action
```

`N` GNN forward passes per decision (batched into one call, but still `N`
node-sets through the encoder). Loss: standard DQN TD error against a
target network, off-policy replay buffer. This is the design originally in
`QNetwork.md`.

## Net B — DQN + lookup

```
base Data(state)
   ▼
Encoder (once)          ──▶  H  [42, hidden_dim]
   │
   ├─ pool(H, u) ──▶ available if a head wants global context too
   ▼
for each candidate (stage, t1, t2, n):
   head_input(stage, t1, t2, n, H, none_vec)
   ▼
heads[stage](head_input)  ──▶  Q(s, a)  [N]
   ▼
argmax -> chosen action
```

**1** GNN forward pass per decision, regardless of `N`. Same DQN loss/target
network/replay buffer as Net A — only the action representation changes.
Cheaper per step; the open empirical question is whether `Q` estimates
degrade without the GNN seeing the action during message passing.

## Net C — PPO + injection

```
base Data(state)                                    base Data(state)
   │  + candidate action (per legal candidate)            │  (unmodified)
   ▼                                                       ▼
CandidateGraphBuilder ──▶ N modified graphs          Encoder (once) ──▶ H_base
   ▼                                                       ▼
Batch.from_data_list                                pool(H_base, u) ──▶ g_base
   ▼                                                       ▼
Encoder (batched) ──▶ H per graph                    critic_head(g_base) ──▶ V(s)
   ▼
pool(H, u) ──▶ g  [N, g_dim]
   ▼
heads[stage](g) ──▶ logit(s, a)  [N]
   ▼
softmax -> Categorical -> sample
```

`N + 1` GNN forward passes per decision — the `+1` is real and worth
calling out: `V(s)` doesn't depend on any candidate, so it has to come from
a pass over the *unmodified* base graph, separate from the `N` candidate
passes the actor needs. This is the most expensive of the four nets.

## Net D — PPO + lookup

```
base Data(state)
   ▼
Encoder (once) ──▶ H  [42, hidden_dim]
   │
   ├──────────────────────────┬───────────────────────┐
   ▼                          ▼                         ▼
for each candidate:      pool(H, u) ──▶ g          (g reused, no extra pass)
  head_input(...)              ▼
   ▼                     critic_head(g) ──▶ V(s)
heads[stage](...)
   ▼ logit(s,a) [N]
softmax -> Categorical -> sample
```

**1** GNN forward pass per decision, shared between the actor (via `H`
lookups) and the critic (via pooling the same `H`). Cheapest of the four —
the actor and critic aren't just both cheap, they share the *same* encoder
output, so there's no `+1` the way Net C has. This was the design
originally in `PPOPolicy.md`.

---

## Comparing the four

| | Algorithm | Action entry | GNN passes / decision | What the GNN "sees" |
|---|---|---|---|---|
| **A** | DQN | inject | `N` | board + this specific candidate, jointly, during message passing |
| **B** | DQN | lookup | `1` | board only; candidate enters after, at the head |
| **C** | PPO | inject | `N + 1` | board + candidate jointly (actor); board alone (critic) |
| **D** | PPO | lookup | `1` | board only; candidate enters after, at the head |

The honest trade is expressiveness (inject) vs. throughput (lookup) — not
"PPO vs. DQN" by itself, since both algorithms can pair with either action
representation. DQN's off-policy replay buffer means each transition gets
reused across many gradient steps, partially amortizing inject's per-step
cost; PPO is on-policy and typically more sample-hungry, so inject's `N+1`
cost is paid more often per unit of policy improvement. That's a reason to
expect B/D to train faster wall-clock-for-wall-clock, but it's exactly the
kind of claim this experiment plan exists to check rather than assume.

---

## Experiment plan

1. **Build the shared foundation first** — `Encoder`, `CandidateGraphBuilder`,
   the lookup `head_input`/`none_vec` helper, the per-stage `heads`
   `ModuleDict` pattern, and the `num_legal_actions` addition to
   `GraphAdapter`. Verify each in isolation against real `Environment`
   states (same style as `GraphAdapter.md`'s/`Action.md`'s "Verified"
   sections) before any of the four nets are assembled — a bug shared by
   all four is much cheaper to catch once than four times.
2. **Assemble all four nets as thin wrappers** over the shared pieces —
   ideally one `risk/learning/` module per net (or one configurable class
   with `algorithm: Literal["dqn","ppo"]` / `action_entry:
   Literal["inject","lookup"]` flags, if the four wrappers turn out to be
   mostly boilerplate around the same shared calls). Confirm each one plays
   a full legal game via `SelfPlay.play_headless` with an untrained network
   before training anything — sampling/arg-maxing from random weights
   should still only ever produce legal moves.
3. **Train all four against the same self-play setup** — same opponent
   roster (`RaiderAgent`/`SentinelAgent`/`EmpireAgent`/`RandomAgent`, as in
   `self_play.py`'s `main()`), same `n_players`/seeds where feasible.
   Track wall-clock time and env-step count separately, since A/C and B/D
   have different per-decision cost — comparing by wall-clock alone would
   conflate "trains better" with "computes cheaper per step."
4. **Evaluate head-to-head** — round-robin tournament among the four
   trained agents plus the existing heuristic baselines via
   `SelfPlay.play_headless`/`play_rendered`, tracking win rate. This is the
   actual answer to "which one is better" — empirical, not argued from
   architecture diagrams.
5. **Only then** tune hyperparameters (clip `eps`, GAE `λ`/`γ` for C/D;
   target-update frequency, replay size for A/B; learning rate, entropy
   coefficient, epochs per batch) — on whichever net(s) the tournament
   says are worth the investment, not all four equally.

Steps 1-2 are the actual subject of this doc; 3-5 are the experiment that
decides which of the four nets to keep building on.

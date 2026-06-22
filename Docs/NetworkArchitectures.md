# Network architectures — DQN × PPO, injection × lookup (plan)

Planning doc for the network(s) that sit on top of
[`GraphAdapter`](GraphAdapter.md) and [`Action.md`](Action.md) to choose
actions. Shared-foundation pieces are being built incrementally — see
[ActionGraphBuilder.md](ActionGraphBuilder.md) (implemented) — but
none of the four nets themselves exist yet: no `risk/learning/encoder.py`,
`q_network.py`, or `policy_network.py`. This supersedes the earlier
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
   before the GNN runs (modify `x`/`edge_attr` per action, one GNN pass
   per action), vs. **look it up** from one GNN pass over the
   unmodified state graph (read `H[t1]`/`H[t2]` after the fact, one GNN
   pass per decision, regardless of how many legal actions exist).

Neither axis implies the other — the algorithm decides what the output
scalar(s) mean and how training works; the action representation decides
how an action gets *into* the network and at what cost. That's four nets:

| | **Inject** | **Lookup** |
|---|---|---|
| **DQN** | **Net A** — `Q(s,a)` from a per-action modified graph | **Net B** — `Q(s,a)` from one shared graph + embedding lookup |
| **PPO** | **Net C** — `logit(s,a)` from a per-action modified graph, `V(s)` from the base graph | **Net D** — `logit(s,a)` and `V(s)` both from one shared graph |

The plan is to build and train all four against the same self-play setup
and find out empirically which fits Risk best, rather than deciding by
argument — see "Experiment plan" at the end.

---

## Shared foundation (all four nets)

This is the structure that would otherwise be duplicated four times. Build
it once, in its own module(s), and have all four nets import it.

### Base graph — unchanged, from `GraphAdapter`

`Data(x=[42,13], edge_index=[2,166], u=[1,34])`, exactly as documented in
[GraphAdapter.md](GraphAdapter.md). None of the four nets touch
`GraphAdapter` itself — it stays the action-independent state encoding.

**One addition all four nets benefit from:** append a `num_legal_actions`
scalar (raw count, or `log1p`-scaled) to `u`. This was a real gap in the
lookup-based design specifically — a per-action head reading `H[t1]`/
`H[t2]` has no way to know "is this one of 3 options or one of 40" unless
that's surfaced somewhere global — but since it lives in `u`, every net
gets it for free (pooled into the critic/Q-head for Nets A-C, and available
to Net D's lookup heads via the same `u` they already read). Goes in
`GraphAdapter._global_features`, sourced from `len(env.legal_actions())` at
adapter-call time — a small, deliberate exception to `GraphAdapter`'s
"per-state, not per-action" framing, justified because it's a *property of
the state* (how open the position is) rather than of any one action.

### `Encoder` — one shared module, `risk/learning/encoder.py`

**Implemented**, exactly as planned below — `TransformerConv` is a graph
*attention* layer, not a spectral GCN, so the class keeps the name
`Encoder` rather than something GCN-flavored:

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

Verified against real action batches from `ActionGraphBuilder`
(`in_dim=13`, `edge_dim=2`, `hidden_dim=64`, `n_layers=4`): encodes a full
74-action `ATTACK`-phase batch to `[3108, 64]` (3108 = sum of nodes
across all action graphs), encodes a single base graph with zero
`edge_attr` (the lookup-style call) to `[42, 64]`, and gradients flow back
through every layer.

Identical module for all four nets. What differs is only what gets passed
in:

- **Inject (A, C):** `edge_attr` carries the real per-action marker for
  `ATTACK` (`[selected_attack, dice_count]`, zero on every other edge —
  see "Edge injection" below); `x` carries the per-action node
  perturbation for `REINFORCE_PLACE`/`OCCUPY`/`FORTIFY`. Called once per
  *action* (batched via `Batch.from_data_list`).
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

### Action injection (used by Nets A, C only) — `ActionGraphBuilder`

**Implemented** — `risk/learning/action_graph_builder.py`, reference doc
[ActionGraphBuilder.md](ActionGraphBuilder.md). One difference from
the original plan below: the per-territory perturbation writes directly
into `x`'s existing army-count column rather than a parallel
`proposed_delta` column (see that doc's "Design decision" section for why,
and when to revisit).

For an `AttackAction(from=A, to=B, dice=d)`:

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

`edge_index` never changes per action, only `edge_attr`/`x` do — and
`topology.edge_index()`'s row order is stable (`BoardTopology` sorts once
at load time), so `index_of(A -> B)` can be precomputed once, not searched
per action.

`ActionGraphBuilder` never mutates the base `Data` — every action
needs its own unmodified copy to diverge from.

### Action lookup (used by Nets B, D only) — embedding heads

For one legal action's `(stage, t1, t2, n)` (from the already-implemented
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

**Implemented** — `risk/learning/heads.py`'s `ScoringHead`/`TradeInHead`
classes, one *named submodule per stage* rather than a `ModuleDict` with
internal stage-keyed dispatch:

```python
class ScoringHead(nn.Module):       # REINFORCE_PLACE/ATTACK/OCCUPY/FORTIFY —
    def __init__(self, g_dim):      # identical shape, 4 separate instances
        self.net = _mlp(g_dim)      # (independent weights, shared class)

    def forward(self, g, card_indices):   # card_indices unused — kept only
        return self.net(g).squeeze(-1)    # so every head shares one call shape

class TradeInHead(nn.Module):       # different input shape -> its own class
    def __init__(self, g_dim, card_embed_dim=8):
        self.card_embedding = nn.Embedding(MAX_CARDS_IN_HAND + 1, card_embed_dim)
        self.net = _mlp(g_dim + 3 * card_embed_dim)

    def forward(self, g, card_indices):
        ...
```

Both heads now share one call signature, `(g, card_indices) -> Q`, even
though `ScoringHead` ignores the second argument — purely so
`GNN_DQN.forward` (see Net A below) can route *every* row through
`self._heads_by_phase[stage](g[mask], card_indices[mask])` in a single
uniform loop, with no `if`/branch distinguishing `TRADE_IN` from the rest.
`card_indices` itself is `[N, 3]` long and not optional at that call site
— rows that aren't `TRADE_IN` just carry zeros (or anything; ignored), so
the tensor's shape never depends on which stages are actually present.
Each head class is still just `(its input) -> scalar` with no dispatch
logic of its own — deciding *which* head scores a given row stays
`GNN_DQN.forward`'s job, not something the agent has to do by hand. Five
small heads, same reasoning in all four nets: the stages are different
decisions with different
`n`-semantics and value scales, so one head per stage avoids forcing a
shared tail to disambiguate that implicitly. What differs per net is only
the **input** to each head (pooled action-graph embedding for A/C,
`head_input(...)` lookup for B/D) and the **output's meaning** (`Q` for
A/B, `logit` for C/D).

`TradeInHead`'s embedding table has one extra row reserved for the
`SkipTradeAction` sentinel (`dqn_index()`'s `(-1, -1, 0)`) — detected per
*row* (`t1 < 0`), not per element, since `n == 0` is `SkipTradeAction`'s
placeholder but a legitimate real card-slot index for `TradeInAction`;
per-element masking would silently mis-embed that slot. Verified directly:
a real rollout exercising the sentinel row through `TradeInHead` produces
the dedicated "none" embedding rather than colliding with card slot 0.

`Phase` doubles as the DQN action-representation stage directly
(`Docs/Action.md`), so any one decision's legal actions are always a
single stage now — the agent never needs to handle a mixed-stage batch
*within one decision* (it still will across a sampled replay-buffer
minibatch, which spans many decisions — `Docs/RL-Prep-Changes.md`).

### Pooling (used wherever a whole-graph scalar is needed)

**Implemented** — `risk/learning/pooling.py`'s `pool(h, batch, u)` function
(needs the PyG `batch` index tensor too, to know which nodes belong to
which graph — omitted from the snippet below for brevity).

```python
g = torch.cat([global_mean_pool(H), global_max_pool(H), u], dim=-1)
```

`u` concatenated in after pooling, not fed through the GNN — it's already
global (phase, budget, eliminations, now `num_legal_actions`), nothing for
message passing to refine, and pure node pooling alone is blind to it.

---

## Net A — DQN + injection

The network and the game-logic orchestration around it are two separate
classes (mirroring `Temp/Examples/DQN_Agent.py` wrapping a bare `DQN`):

- **`risk/learning/gnn_dqn.py`'s `GNN_DQN`** — **implemented**, all 5
  stages in one call, one uniform loop: `forward(state, phase,
  card_indices) -> Q`.
  - `state` — always a `Batch`, one graph per row being scored, even for
    `N=1` (`Batch.from_data_list([one_graph])`) — same convention as
    carrying a batch dimension of 1 in any other PyTorch net; `forward`
    doesn't detect/special-case a bare single `Data`, that's the caller's
   job. *Injected* (`ActionGraphBuilder`) for the 4 graph-based stages;
   for `TRADE_IN` the builder returns an unmodified copy of
   `GraphAdapter`'s base graph (repeated once per candidate in that
   decision, since none of them perturb the graph) — `GraphAdapter`
   already gives every base graph a zero-filled `edge_attr` of its own
   (`Docs/GraphAdapter.md`), so `forward` doesn't need to default
   anything; every graph that reaches it already has the same fields
   regardless of stage.
  - `phase` — `[N]` long, one `Phase` value per row, same convention as
    `ReplayBuffer`'s `stage`/`next_stage`.
  - `card_indices` — `[N, 3]` long, **not optional**: each row's
    `dqn_index()` `(t1, t2, n)` for `TRADE_IN` rows, zeros (or anything;
    ignored) for every other row. Required unconditionally — not just
    whenever a `TRADE_IN` row happens to be present — specifically so its
    shape never depends on which stages are actually in the batch.

  Internally: `Encoder` + `pool` once for the *whole* batch regardless of
  stage mix, then **one loop, no branching**: over the 5 DQN phase/head
  slots (`TRADE_IN`, `REINFORCE_PLACE`, `ATTACK`, `OCCUPY`, `FORTIFY`),
  scoring only rows whose mask is present with
  `self._heads_by_phase[stage](g[mask], card_indices[mask])`.
  This only works because every head — `ScoringHead` and `TradeInHead`
  alike — shares the same `(g, card_indices) -> Q` call signature now (see
  "Per-stage heads" above); there's no separate `if is_trade_in` path
  anymore. The dict lookup is built once in `__init__`, aliasing the same
  submodule instances already registered as named attributes (not a
  second registration). The only game-adjacent import is the leaf `Phase`
  enum itself (no dependencies of its own) — not `Action`/
  `ActionGraphBuilder`/`Environment`.

  Verified against a real 400-step rollout (driven manually, playing the
  agent's role by hand — see below, with a uniform `[N, 3]` `card_indices`
  passed every call, zeros for non-`TRADE_IN` decisions) plus a fabricated,
  fully mixed batch — `TRADE_IN` rows (including a `SkipTradeAction`
  sentinel) *and* graph-based rows from a different decision, scored
  together in one `forward` call through the same uniform loop, the shape
  a sampled replay-buffer minibatch would actually be: every row scored
  correctly, gradients confirmed flowing.
- **`GNN_DQN_Agent`** — **implemented**, including `train_step`. Owns
  `ActionGraphBuilder`, builds/batches the per-candidate graphs, builds the
  `phase`/`card_indices` tensors, merges scores back to `legal_actions()`'s
  order, `argmax`s. This is the piece that actually knows what a
  `State`/`Action` is.
  - `train_step(batch_size)` — one DQN TD-error update: samples a
    minibatch from `self.replay_buffer`, scores the taken `(state,
    action)` pairs with the online net, scores every legal action of each
    `next_state` with the target net (one batched forward pass over *all*
    transitions' candidates at once, reduced back to a per-transition max
    via `torch_geometric.utils.scatter(..., reduce="max")` keyed by a
    per-row transition index — cheaper than one forward pass per
    transition), Huber loss against `r + gamma * (1 - done) * max_a'
    Q_target(s', a')`, one optimizer step. Target net is hard-synced to
    the online net every `target_update_every` calls (an `Adam` optimizer
    and `gamma`/`lr`/`target_update_every` are now `GNN_DQN_Agent`
    constructor kwargs). Verified against a real self-play rollout: online
    net params change after `train_step`, target net is unchanged between
    syncs and exactly matches the online net immediately after one.
  - `risk/learning/trainer.py`'s `Trainer` (not part of this doc's shared
    foundation, but the thing that actually runs Net A's training step
    from "Experiment plan" step 3) reuses one `GNN_DQN_Agent` across many
    self-play episodes, reassigning it to a random seat/`n_players` every
    episode via `attach(...)` — see `GraphAdapter`'s `perspective`
    parameter (`Docs/GraphAdapter.md`) for how the net still sees one
    consistent "this is me" frame despite the seat changing, and
    `Docs/RL-Prep-Changes.md`'s "perspective-relative encoding + `Trainer`"
    entry for the full writeup.

```
base Data(state)
   │  + one legal action (per action)            <- the agent's job from here down
   ▼
ActionGraphBuilder  ──▶  N graphs (injected, or base copy for TRADE_IN), phase + card_indices tensors
   ▼
Batch.from_data_list
   ▼
GNN_DQN.forward(state, phase, card_indices) ──▶ Q(s, a)  [N]   <- the net's job: encode + pool + route to head
   ▼
argmax -> chosen action                                          <- back to the agent
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
for each legal action (stage, t1, t2, n):
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
   │  + one legal action (per action)                     │  (unmodified)
   ▼                                                       ▼
ActionGraphBuilder ──▶ N modified graphs             Encoder (once) ──▶ H_base
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
calling out: `V(s)` doesn't depend on any action, so it has to come from
a pass over the *unmodified* base graph, separate from the `N` action
passes the actor needs. This is the most expensive of the four nets.

## Net D — PPO + lookup

```
base Data(state)
   ▼
Encoder (once) ──▶  H  [42, hidden_dim]
   │
   ├──────────────────────────┬───────────────────────┐
   ▼                          ▼                         ▼
for each legal action:   pool(H, u) ──▶ g          (g reused, no extra pass)
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
| **A** | DQN | inject | `N` | board + this specific action, jointly, during message passing |
| **B** | DQN | lookup | `1` | board only; the action enters after, at the head |
| **C** | PPO | inject | `N + 1` | board + action jointly (actor); board alone (critic) |
| **D** | PPO | lookup | `1` | board only; the action enters after, at the head |

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

1. **Build the shared foundation first** — `Encoder`, `ActionGraphBuilder`,
   the lookup `head_input`/`none_vec` helper, the per-stage scoring heads
   (`ScoringHead`/`TradeInHead`), and the `num_legal_actions` addition to
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

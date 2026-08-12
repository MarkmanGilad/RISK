# Learning to Play Risk with Action-Injected Graph Neural Networks

A graph-attention reinforcement-learning approach to Risk, compared across
DQN, Dueling DQN, and PPO.

Gilad Markman · Practical Deep Learning for Science · 2026

> **Instead of scoring from one enormous fixed list of moves, the system
> injects each legal Risk move into the board graph and scores that
> state–action graph with one shared graph-attention encoder.**
>
> **That encoder is trained through reinforcement learning — compared across
> DQN, Dueling DQN, and PPO — so only the learning objective changes between
> them.**

---

## How Risk is played

Risk is played on a 42-territory world map, and **the goal is to conquer the
entire board by eliminating every opponent**. Territories change hands
through dice combat: the attacker and defender each roll one die per army
committed (up to three for the attacker, two for the defender), the highest
dice are compared pair by pair, and the losing side removes one army each
round — repeated until the attacker retreats or the defending territory is
emptied and conquered. Each turn steps through five phases in order:
**trade-in** cards for bonus armies, **reinforce** with new armies,
**attack** neighbouring territories, **occupy** a territory just conquered,
and **fortify** by regrouping armies before ending the turn.

<img src="../Assets/RiskMap/image.png" alt="Risk game board" width="760">

**Figure 1.** The playable Risk board.

---

## The problem: an enormous, mostly-illegal, structured action space

An attack chooses a source territory, an adjacent enemy target, and a dice
count; reinforce, occupy, and fortify each choose a territory and an army
amount. Enumerated over 42 territories and their borders, the nominal action
space is huge — and in any given state, almost all of it is illegal. A
fixed-size Q-output network must still carry one output per nominal action,
learn over combinations that are rarely or never legal, and mask out
everything else at every decision.

The state itself resists a flat vector, too: ownership, armies, continents,
cards, and borders are relational — whether a move is good depends on who is
adjacent, who is strong, and which continent is contested. Flattening that
into one fixed-length vector throws the relational structure away.

---

## The idea: represent the board as a graph, inject the action, score only what's legal

Represent the board as a graph — nodes are territories, edges are borders —
so the relational structure survives. For each **legal** action, inject it
directly into that graph as a state–action pair and let one shared encoder
score it. No fixed action-output layer, no masking: the network only ever
sees candidates that can actually be taken.

<img src="../Assets/RiskMap/map_graph_nodes_edges.png" alt="Risk board with territory nodes and border edges" width="760">

**Figure 2.** The board as a graph: 42 territory nodes and 83 undirected
borders, stored as 166 directed edges for message passing.

~~~text
X: node features       [42 × 15]
E: edge features       [166 × 2]
u: global features     [1 × 35]
~~~

---

## A controlled environment supplies legal actions

The environment owns Risk's rules and legal-action validation. At each
decision it exposes the state and the valid actions for the current phase;
the agent chooses one; the environment applies it, opponents respond, and
returns the next state, reward, and terminal signal. Heuristic and learning
agents share this same interface.

<img src="../Assets/RiskMap/start%20UI.png" alt="Risk player-selection screen" width="620">

**Figure 3.** The start screen selects human, heuristic, and learning agents
through the same game interface.

| Agent | What it emphasizes |
|---|---|
| Random | Uniformly samples a legal move. |
| Raider | Aggressive expansion and marginal-odds attacks. |
| Sentinel | Border defence and safer attacks. |
| Empire | Capturing and protecting continents. |
| Killbot | Continent strategy plus weak-player elimination. |

---

## Injecting a candidate action into the graph

For an attack, the selected directed border receives an orange edge feature
such as `[attack = 1, dice = 2/3]`. For reinforce, occupy, and fortify,
affected territory rows receive a proposed army-change feature. Skip actions
use an unmodified graph copy. The network sees **the board plus the specific
candidate move**.

<img src="../Assets/RiskMap/partial_graph_attributes.png" alt="Partial Risk graph with node, edge, global attributes, and an injected attack" width="1250">

**Figure 4.** One legal attack changes the selected border before graph
attention (orange = injected candidate action). Node features describe
territories, edge features describe borders and attacks, and global features
describe phase-level constraints.

1. **Rules enumerate legal candidate actions.**
2. **Injection marks one candidate action in the graph.**
3. **The GNN returns one score for that state–action graph.**

---

## One shared graph encoder, five action-phase heads

Every legal candidate enters the same graph-attention encoder. Four residual
`TransformerConv` layers exchange information along Risk borders, so a
territory can weigh different neighbours differently. Mean and max pooling
summarize territory embeddings, global state is appended, and the current
phase chooses one small MLP head. The representation is shared while the
learning objective changes.

<img src="../Assets/RiskMap/network_phase_heads_v2.png" alt="Shared Risk graph-attention encoder with five phase-specific heads" width="1250">

**Figure 5.** A legal action becomes one graph row. The shared encoder is
followed by the relevant trade-in, reinforce, attack, occupy, or fortify
head. DQN treats the scalar as `Q(s, a)`; a policy learner uses legal-action
logits and a value estimate.

| Learner | Legal-action output | Training signal |
|---|---|---|
| DQN | `Q(s, a)` | replayed Double-DQN targets |
| Dueling DQN | state value + relative advantage | replayed Double-DQN targets |
| PPO | policy logit and state value | on-policy clipped updates |

---

## Where the injected edge enters the calculation

<img src="../Assets/encoder_matrix_summary.png" alt="Matrix summary of the sparse Risk graph-attention encoder" width="1550">

**Figure 6.** Attention is computed only on 166 directed Risk borders. The
selected attack changes one projected edge row, which changes attention and
the neighbour message for that candidate graph.

---

## Results: comparing learners at a matched data budget

Same legal-action generator · Same injected graph representation
Same heuristic-opponent roster · Randomized learner seat and player count
Compared against cumulative learner turns · Held-out evaluation reported separately

| Learner | Chart colour |
|---|---|
| DQN | blue |
| Dueling DQN | purple |
| PPO | teal |

**Figure 7.** Rolling training win rate vs. cumulative learner turns — DQN ·
Dueling DQN · PPO. *[reserved for the final training-curve chart]*

**Figure 8.** Balanced held-out evaluation win rate at matched training
budgets, with uncertainty bars and the number of games stated. *[reserved
for the final evaluation chart]*

**Figure 9 (optional).** Territories conquered or agent-turn survival vs.
learner turns. *[reserved for a supporting-evidence chart]*

---

> **Action injection converts a changing legal-action set into a graph
> scoring problem: the GNN evaluates the board and the proposed move
> together.**

**Limitations**

- Risk outcomes are stochastic — held-out evaluation and uncertainty matter
  more than a smoothed training curve.
- One injected graph per legal candidate is expressive but costs more than a
  single fixed action-output layer.
- DQN, Dueling DQN, and PPO are compared only at matched training budgets,
  under the same opponent protocol.

**Footer**

- Repository QR code · experiment-tracking QR/link
- Course and author line
- Citations: DQN/Double DQN, Dueling DQN, PPO, GAT, PyTorch-Geometric `TransformerConv`

---

## Layout

Visual-first upper half, full-width results band, compact footer.

~~~text
┌──── Header: title + claim + game primer (Figure 1) ──────────────────────────────────────────────────┐
├───────────────────────────────────────┬────────────────────────────────┬───────────────────────────┤
│ Problem → idea → board-graph (Fig. 2)  │ Action injection (Figure 4)    │ Shared encoder (Figure 5) │
├───────────────────────────────────────┴────────────────────────────────┴───────────────────────────┤
│ Player-selection UI + opponent roster (Figure 3)   │ Sparse-attention technical inset (Figure 6)     │
├──────────────────────────────────────── Results: Figures 7–9, full width ───────────────────────────┤
├──────────────────────────────────── Footer: take-home · limitations · QR · citations ────────────────┤
└────────────────────────────────────────────────────────────────────────────────────────────────────┘
~~~

| Region | Share |
|---|---:|
| Header | 11% |
| Problem + idea + method visuals | 41% |
| UI + technical inset | 14% |
| Results | 26% |
| Footer | 8% |

# Poster design brief — GNN reinforcement learning for Risk

This document is the writing and layout plan for an **A0 landscape** scientific
poster. It is not the final poster artwork. It keeps the technical claims,
draft text, figure placeholders, and evidence standards in one place so the
later design pass can focus on visual clarity.

## Poster identity

**Course:** Practical Deep Learning for Science

**Author:** Gilad Markman

**Year:** 2026

**Lecturer:** Prof. Eilam Gross

**Teaching assistants:** Dmitrii Kobylianskii, Alon Levi, and Etienne Dreyer

**Working title:**

> **Learning to Play Risk with Graph Neural Networks and Reinforcement Learning**

**One-sentence message:**

> We represent the Risk board as a graph and evaluate each currently legal
> move by injecting that candidate move into a graph before a graph-attention
> network scores it through a head specialized for that action phase.

**Supporting line for the header/subtitle:**

> A shared graph-attention encoder learns strategic board context; separate
> heads score trade-in, reinforcement, attack, occupy, and fortify decisions.

**Audience:** course staff and students who understand basic deep learning,
but may not know Risk, graph neural networks, or reinforcement learning.

**Recommended layout:** A0 landscape, three columns, read left to right. Use a
large central pipeline figure as the visual anchor; do not make the poster a
wall of text.

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│ Title · Gilad Markman · Practical Deep Learning for Science (2026)          │
├───────────────────────┬───────────────────────────┬─────────────────────────┤
│ 1. Problem            │ 2. Method (largest area)  │ 3. Experiments/results  │
│ 2. Why graphs          │ 3. Action injection       │ 4. What we learned       │
│ [game-board image]    │ [map + graph overlay]      │ [W&B charts]             │
│                        │ [network pipeline]         │ [comparison table]       │
├───────────────────────┴───────────────────────────┴─────────────────────────┤
│ Take-home message · limitations · QR/repository link · acknowledgements     │
└─────────────────────────────────────────────────────────────────────────────┘
```

Use one accent color for the proposed action/injection throughout the poster
(for example orange), one for the state graph (blue/grey), and one for results
(green). Keep all other board territory colours subdued so the proposed action
is visually dominant.

---

## 1. Left column — problem and graph representation

### Heading

## Risk is a graph decision problem with a changing legal action set

### How the game works

**Draft text (about 85 words):**

Risk is a turn-based strategy game played on a world map of connected
territories. Each territory belongs to one player and contains an army count.
On a turn, a player places reinforcement armies, attacks adjacent enemy
territories, moves armies into conquered territory, and may fortify between
connected territories they own. In an attack, the attacker rolls up to three
dice and the defender up to two; sorted dice are compared pairwise and each
lost comparison removes one army. When the defender reaches zero armies, the
attacker occupies that territory. Controlling every territory in a continent
grants extra reinforcement armies on later turns. The objective is to eliminate
every opponent and control the board.

**Suggested visual:** a compact five-step strip directly under Figure A:

```text
Reinforce → Attack neighbours with dice → Occupy conquered territory → Fortify
```

Use one sentence below the strip: “Winning dice comparisons remove armies;
reaching zero defenders conquers a territory. Continents earn future
reinforcement armies.” This connects the rules directly to the features and
reward-shaping panels without requiring the reader to know Risk.

### Foundation: the game as an RL environment

**Draft text (about 80 words):**

Risk was implemented as a reproducible reinforcement-learning environment,
not only as a game interface. At every decision, the environment exposes the
current state and the exact set of legal actions for the active phase. An agent
selects one legal action; the environment applies the rules, advances the game,
returns the next state and reward, and signals a terminal win or loss. This
environment–agent separation lets the same validated game rules train and
evaluate a random baseline, heuristic agents, DQN, Dueling DQN, and PPO
agents. The game also includes heuristic opponents with aggressive, defensive,
continent-focused, and weak-player-elimination strategies, giving the learner
varied opponents through that same interface.

**Suggested visual:** a small four-step loop placed above Figure A:

```text
Environment: state + legal actions → Agent: choose action → Environment: rules
and opponents → next state + reward + done
```

**Caption:**

> **Figure 0.** The RL loop. The environment owns Risk rules and legal-action
> validation; heuristic and learning agents share the same decision interface.

This section should make clear that the project did not assume an existing
Risk/Gym environment. Do not spend poster space on every game rule; the key
claim is the clean, testable interface needed to compare learning agents.

### Draft text (about 70 words)

Risk combines a fixed geographical board with a changing multi-player state.
Territories are connected by borders; ownership, armies, cards, turn phase,
and reinforcement budget change after every move. The number and meaning of
available actions also change by phase and state. A standard neural network
with one fixed output for every possible move would be wasteful and would score
many illegal actions. We instead enumerate only legal actions and score each
one in its board context.

### Figure A — game board

**Reserve:** upper half of left column, approximately 25% of total poster
area.

**Asset:** `Assets/RiskMap/map_grey_new.jpg`.

**What to add later:** a small legend explaining ownership colours and army
counts. This figure should introduce the game, not attempt to explain every
network feature.

**Caption:**

> **Figure 1.** The Risk board contains 42 territories connected by borders.
> A decision depends on both a territory's local state and its neighbours.

### Figure B — board becomes a graph

Build this on top of a copy of `map_grey_new.jpg`, preferably immediately
beside or below Figure A:

- place one visible node at the centre of each territory;
- draw a thin edge for each border; arrows are unnecessary here because the
  map would become cluttered;
- colour one owned territory, one opponent territory, and one proposed-action
  pair as examples;
- add a small callout, not a full tensor dump.

**Caption:**

> **Figure 2.** The board is encoded as a graph: territories are nodes and
> borders are edges. The same topology is reused while the node, edge, and
> global state features change each turn.

### Compact feature callout next to Figure B

Use a small three-row table or labelled boxes:

| Graph part | Examples of information | Why it matters |
|---|---|---|
| Territory node | continent, relative owner, armies, unfinished attack target, proposed army change | local strength and local action effect |
| Border edge | adjacency; selected attack and dice count when applicable | which territories can interact |
| Global state | phase, current player, cards, reinforcement budget, eliminated players, continent values | game-wide constraints and turn context |

**Technical footnote, optional:** The current implementation uses 42 nodes,
166 directed edges, 15 node features, 2 edge features, and 35 global features.
This is useful as a small reproducibility detail, not as the main message.

---

## 2. Centre column — method (make this the largest column)

### Heading

## Action injection turns “state + move” into a graph the GNN can understand

### The action-space problem

**Draft text (about 75 words):**

The action space is combinatorial. An attack chooses a source territory, a
neighbouring target, and a dice count; reinforcement and fortification choose
territories and army amounts. Most combinations are illegal in a given state.
Rather than predict over a huge fixed action vector and then mask invalid
outputs, the game rules enumerate the legal candidates. The network scores
only these candidates, so every score corresponds to an executable action.

### Discretising quantitative actions without losing strategic choice

**Draft text (about 85 words):**

Some legal moves also require choosing an integer number of armies. Enumerating
every possible reinforcement amount would make the action set grow with the
reinforcement budget. We therefore offer three representative reinforcement
amounts for each owned territory: **one army**, **half of the remaining
budget**, or **the whole remaining budget**. Reinforcement is multi-step, so
the agent can still construct a finer split through successive decisions. This
keeps the candidate set proportional to the number of territories rather than
to the potentially large army count, while preserving small, medium, and
committed strategic choices.

**Suggested visual:** add three small branches beneath a reinforced territory
in Figure C: `+1`, `+½ budget`, and `+all budget`. Label the benefit:
“bounded legal action set; multi-step placement retains flexibility.”

**Accuracy note for the final poster:** fortification uses the same bounded
`1 / half / maximum` candidate principle as reinforcement. For each reachable
owned source–destination pair, the legal actions include those deduplicated
transfer amounts plus skip. The environment still accepts any smaller valid
fortification amount when explicitly submitted.

### Figure C — action injection (the main figure)

**Reserve:** centre of the poster, roughly 30% of total poster area.

Create a three-panel figure using the graph-overlay map:

```text
Current state graph        Candidate action              Injected action graph
[grey/blue territories]   attack A → B, 3 dice           [same graph + orange A→B]
                                                            selected edge = 1
                                                            dice feature = 3 / max dice
```

For a reinforcement, occupy, or fortify example, show an orange `+k`/`−k` on
the affected nodes instead. Do not try to show all action types in the main
figure; an attack example is the clearest. Add a small side note:

> Other phase actions inject proposed army changes into the affected territory
> nodes. Skip/stop actions remain an unmodified graph copy.

**Caption:**

> **Figure 3.** Action injection. For each legal candidate, we clone the base
> state graph and mark the proposed action. Message passing can therefore
> evaluate the *consequence-relevant context* of a specific move, not only the
> board and the move separately.

### Network pipeline

Place directly beneath Figure C. Prefer this short visual rather than a
layer-by-layer implementation diagram:

```text
Legal actions from rules
          ↓
One injected graph per legal action
          ↓
Shared graph-attention encoder
          ↓
Mean + max graph pooling  ⊕  global state features
          ↓
Phase-specific scoring head
          ↓
Q(s,a) for DQN   or   policy logit + V(s) for PPO
```

### Why graph attention?

**Draft text (about 90 words):**

Risk interactions are relational: the importance of a territory depends on
which neighbouring territories are friendly, hostile, weak, or part of a
continent. We use residual `TransformerConv` layers, a graph-attention
operator. Attention lets the model learn different weights for different
neighbours instead of treating every border identically. Edge features also
allow the selected attack edge and its dice count to influence message
passing. Four residual graph-attention layers produce territory embeddings;
mean and max pooling summarize the board before a phase-specific head produces
one scalar score.

**Be precise:** call it *graph attention* or `TransformerConv`; do not call it
a plain GCN. Do not claim attention makes the model interpretable unless you
include a real attention analysis.

### Small algorithm box

Keep this short:

| Learner | Network score | Learning signal |
|---|---|---|
| DQN | `Q(s, a)` | replayed transitions and Double-DQN targets |
| Dueling DQN | `V(s) + A(s,a) − mean(A)` | replayed transitions and Double-DQN targets |
| PPO | policy logit for each legal action + `V(s)` | on-policy rollouts and clipped policy updates |

**Draft text:**

The representation is held constant across learners. This isolates the effect
of the reinforcement-learning objective and value decomposition: DQN learns
action values from replay; Dueling DQN separates state value from relative
action advantage; PPO optimizes a legal-action policy from fresh rollouts.

---

## 3. Right column — experiments and evidence

### Heading

## Early experiments: injected-action DQN is currently the stronger baseline

Use cautious wording. The current evidence supports DQN_105 as the strongest
completed baseline; PPO_200 showed meaningful learning signals but has not
demonstrated DQN parity in win rate. PPO_201 tested a lower learning rate and
was weaker. PPO_202 is a fresh midpoint learning-rate experiment and should
be labelled **running** until enough data is collected. A new Dueling DQN run
is planned: earlier behaviour was promising, but it must be reported as
preliminary until the new controlled run is complete.

### Experiment protocol box

**Draft text (about 65 words):**

Agents train through self-play against a changing roster of heuristic
opponents, with randomized learner seat and player count. All learners use the
same board encoding and legal-action generation. We log game outcomes, dense
behavioural measures, optimizer diagnostics, and deterministic evaluation.
Because policy optimization methods consume data differently, comparisons must
state their x-axis: learner turns or processed samples for sample efficiency;
optimizer steps only for update-count efficiency.

### Training opponents — why not only random play?

**Draft text (about 80 words):**

The learner does not train only against random agents. Each non-learner seat is
filled from a roster of fixed, rule-based opponents with different strategic
styles: aggressive expansion (**Raider**), defensive border protection
(**Sentinel**), continent control (**Empire**), and a stronger Killbot-inspired
opponent that combines continent strategy with weak-player elimination. These
opponents create varied tactical situations and prevent the learner from
overfitting to one predictable policy. Learner seat and number of players are
randomized, while the graph representation keeps a consistent “me versus
others” perspective.

**Suggested visual:** four small labelled icons or simple strategy arrows,
not four separate board screenshots. Place this beside the experiment protocol
box.

### Reward shaping — making long-horizon learning practical

**Draft text (about 105 words):**

Winning or losing Risk is a delayed, sparse signal. We therefore use terminal
rewards of **+300** for a win and **−300** for a loss, together with bounded
dense reward shaping. Shaping rewards strategically useful intermediate
behaviour: favourable army ratios and exchanges in attacks, conquering
territories, completing continents, eliminating opponents, reinforcing exposed
frontiers, moving armies forward after conquest, and fortifying toward borders.
At turn boundaries it also measures changes in territory share, army share, and
continent control after opponents respond. The per-action shaping signal is
clipped and scaled by 0.3, so it guides exploration without replacing the game
outcome. Win rate remains the primary success measure.

**Suggested visual:** a small horizontal reward timeline:

```text
reinforce frontier → favourable attack → conquest → occupy/fortify → win/loss
      dense reward        dense reward       dense reward             ±300
```

**Important interpretation note:** do not place `reward_per_agent_turn` alone
as a headline result. Dense reward can improve before win rate, and it can also
reflect active local play that does not convert into a game win.

### Figure D — main W&B result

**Reserve:** largest chart in right column.

**Use:** rolling training win rate for DQN_105 and PPO_200, plotted against
**cumulative learner turns** or **cumulative samples processed**.

**Do not use raw optimizer steps as the main fairness chart.** PPO uses a
256-sample minibatch while DQN uses a 64-sample batch, so a raw update count
does not mean equal data exposure. A small supplementary chart may show raw
optimizer steps if it is explicitly labelled with both batch sizes.

**Caption template:**

> **Figure 4.** Rolling training win rate versus [chosen fair x-axis]. DQN_105
> is the current performance baseline. PPO_200 learns more slowly but shows
> improving mid-game behaviour; results are preliminary because rolling
> training wins are noisy and evaluation currently uses a small game set.

When the new Dueling DQN run has a matched comparison window, add it as a third
curve rather than making a separate incomparable chart. Until then, include a
small labelled box: “Dueling DQN: next controlled experiment; preliminary
behaviour promising.”

### Figure E — behavioural learning signals

Use two aligned W&B plots, each with DQN_103/105 and PPO_200/201 only if the
reward regimes are clearly labelled. Prefer PPO_200 vs PPO_201 for a controlled
hyperparameter comparison.

- `territories_conquered`
- `agent_turns_survived`

**Caption template:**

> **Figure 5.** Dense behavioural measures reveal partial progress that win
> rate alone misses: an agent can conquer more territories or survive longer
> without yet converting those advantages into full-game wins.

### Figure F — PPO diagnostic inset (optional)

Use a compact panel containing:

- approximate KL divergence;
- PPO epochs completed per rollout or KL early-stop fraction;
- normalized entropy.

**Caption:**

> **Figure 6.** PPO diagnostics distinguish learning progress from an unstable
> policy update. PPO_200 showed frequent KL early stopping at `1e-4`; PPO_201
> reduced drift at `5e-5` but learned more conservatively. PPO_202 tests the
> midpoint `7.5e-5` setting.

### Results summary table

Avoid unverified final numbers. Fill this table only after choosing a fixed
comparison window and a larger evaluation suite.

| Question | Current evidence | Poster wording |
|---|---|---|
| Does graph/action injection train? | Yes; DQN_105 learns a strong policy. | “The representation supports successful value-based RL.” |
| Does the dueling value/advantage split help? | Earlier signs were promising; a new controlled run is planned. | “Dueling DQN is the next experiment.” |
| Is PPO competitive? | PPO_200 shows useful behavioural improvement but lower win rate. | “Promising but not yet competitive with the DQN baseline.” |
| Is `5e-5` better than `1e-4` for PPO? | It reduces KL drift but was weaker in observed gameplay metrics. | “Lower KL alone did not improve the observed learning outcome.” |

---

## Footer — conclusion, limitations, and next steps

### Take-home message (large type)

> **Injecting each legal action into the board graph avoids a fixed enormous
> action output and lets graph attention score moves in their local strategic
> context.**

### Limitations and next steps (short bullets)

- Training games and evaluation games are stochastic; rolling win rate is
  noisy.
- The present evaluation set is too small for a final claim of algorithm
  superiority; use at least 100 balanced held-out games for promotion claims.
- Action injection evaluates one GNN graph per legal candidate, which is more
  expressive but computationally costly.
- Continue PPO_202 as a controlled learning-rate test; then compare methods at
  matched learner turns and processed samples.
- Run Dueling DQN again with the same reward and opponent protocol, then add it
  to the matched-budget comparison rather than relying on the earlier run.
- Investigate action-set size, per-phase performance, and policy failure modes
  in games where the agent gains territory but does not win.

### Footer placeholders

- Gilad Markman;
- Practical Deep Learning for Science (2026);
- Lecturer: Prof. Eilam Gross;
- Teaching assistants: Dmitrii Kobylianskii, Alon Levi, and Etienne Dreyer;
- GitHub/repository QR code;
- course team acknowledgement;
- one short reproducibility line: “Code: Python, PyTorch, PyTorch Geometric,
  Weights & Biases.”

---

## Visual production checklist

1. Build Figure B from `Assets/RiskMap/map_grey_new.jpg`; retain the map's
   geography but make the overlaid graph readable at A0 viewing distance.
2. Reuse the same map crop and colours in Figure C so the reader immediately
   recognizes that action injection modifies the same graph.
3. Export W&B charts as vector PDF/SVG when possible; otherwise use a
   high-resolution PNG. Remove W&B interface chrome before placing them.
4. Use one consistent legend for DQN/PPO colours across every result figure.
5. Put units and the comparison axis directly on each chart.
6. Keep body text around 28–32 pt, section headings around 44–56 pt, and the
   title around 90–110 pt; verify readability from roughly 1–1.5 metres.
7. Before finalizing, check every numerical claim against the selected W&B
   comparison window and distinguish training metrics from held-out evaluation.
8. Check that the reward panel says both *why shaping is needed* and *why it is
   not the final objective*; the poster should not imply that a high shaped
   reward proves strong play.
9. Keep the environment–agent loop visible: it establishes that legal-action
   generation and rule validation come from the environment, not from an
   unconstrained network output.
10. Show the reinforcement `1 / half / all` choice as an action-space
    discretisation. Keep the fortify description separate unless its legal
    action generator is intentionally changed.
11. Keep the “How the game works” panel concise: objective, territories,
    turn phases, cards, and continent bonuses are sufficient context.

## Sources to cite on the poster

Keep citations compact, in a small footer block:

1. Schulman et al., *Proximal Policy Optimization Algorithms* (2017).
2. Veličković et al., *Graph Attention Networks* (2018).
3. Shi et al., *Masked Label Prediction: Unified Message Passing Model for
   Semi-Supervised Classification* / TransformerConv reference as appropriate
   to the PyTorch Geometric implementation used. Verify the exact preferred
   citation before print.
4. The project repository and experiment tracking run links/QR code.

# Graph-attention encoder: the actual project computation

This guide describes the shared encoder in
[`risk/learning/encoder.py`](../risk/learning/encoder.py) and PyTorch
Geometric's `TransformerConv` exactly as this project configures them. The
encoder has four graph-attention layers, each with one attention head and a
hidden width of 64.

## Board graph input

```text
X: node-feature matrix          [42 x 15]
edge_index: directed borders    [2 x 166]
target_index = edge_index[1]    [166]
E: edge-feature matrix          [166 x 2]
u: global-feature matrix        [1 x 35]
```

`X` has one row per Risk territory. Its 15 columns encode continent, relative
owner, army count, unfinished-attack status, and action-injected
proposed-army delta.

`edge_index` stores the 166 directed board borders. Each column identifies one
source territory and one target territory. The code performs attention only on
these borders; it does **not** build a dense `[42 x 42]` score matrix or a
separate border-mask matrix.

`target_index` is the second row of `edge_index`. For directed border `r`, it
is the index of the territory receiving that border's message. It groups the
166 border scores by receiving territory for `SegmentSoftmax` and `SegmentSum`.

`E` has one row for each directed border:

```text
E[row] = [selected_attack, dice_count]
```

`u` is the 35-value global state vector. It is appended after pooling, not
used in territory-to-territory attention.

## What `H` means

`H` is the current learned territory-representation matrix. One row is one
territory.

```text
H^0 = X                         [42 x 15]

H = X W_in                      [42 x 15] x [15 x 64]
                                = [42 x 64]
```

`W_in` is the first learned mapping from raw board features to 64 learned
values per territory. Every later encoder layer receives and returns an
`H [42 x 64]` matrix.

## One actual `TransformerConv` layer

The layer has one attention head, so all query, key, and value widths are 64:

```text
d_hidden = 64
d_att    = 64
d_value  = 64
```

First, it projects every territory representation three ways:

```text
Q = H W_Q                       [42 x 64] x [64 x 64] = [42 x 64]
K = H W_K                       [42 x 64] x [64 x 64] = [42 x 64]
V = H W_V                       [42 x 64] x [64 x 64] = [42 x 64]
```

- `Q`: what a target territory is looking for.
- `K`: what a source territory offers for attention scoring.
- `V`: the information a source territory can send.

### Edge action injection

The two features on each directed border are projected into the same 64-value
space:

```text
E_projected = E W_E             [166 x 2] x [2 x 64]
                                = [166 x 64]
```

This is the attack-action injection. For a candidate attack, exactly one
directed-border row changes:

```text
selected border:     [1, dice_count / max_attack_dice]
all other borders:   [0, 0]
```

Therefore one row of `E_projected [166 x 64]` changes. The GNN can evaluate
the board together with that proposed attack before the phase head scores it.

### Sparse attention over the 166 borders

The code uses `edge_index` to gather one node representation for each directed
border. It does not form all 42 x 42 territory pairs:

```text
Q_target = GatherTarget(Q, edge_index)         [166 x 64]
K_source = GatherSource(K, edge_index)         [166 x 64]
V_source = GatherSource(V, edge_index)         [166 x 64]
```

The projected edge information is added only to the matching border rows:

```text
K_edge = K_source + E_projected                [166 x 64]
V_edge = V_source + E_projected                [166 x 64]
```

The code then produces one score for each directed border. `RowSum` means sum
across the 64 columns of each row:

```text
scores = RowSum(Q_target * K_edge) / sqrt(64)  [166]
alpha  = SegmentSoftmax(scores, target_index)  [166 x 1]
```

`SegmentSoftmax` normalizes only among borders entering the same target
territory. A non-neighbour has no directed-border row, so it can never receive
an attention weight. This is how the real code enforces the Risk board
topology; it does not need `Mask(S, M)`. In the actual code,
`TransformerConv.message(...)` computes this segment softmax while
`propagate(...)` is running.

The weighted messages are aggregated back to one row per territory:

```text
weighted_messages = alpha * V_edge              [166 x 64]
neighbour_message = SegmentSum(weighted_messages, target_index)
                                [42 x 64]
```

`SegmentSum` adds all incoming directed-border rows into the matching target
territory row. This sum is automatic: PyTorch Geometric's `propagate(...)`
performs it with this layer's `aggr="add"` setting.

## The two skip paths

Inside each `TransformerConv`, the root territory is transformed and added to
the neighbour message:

```text
R = H W_skip                    [42 x 64] x [64 x 64] = [42 x 64]
conv_output = neighbour_message + R             [42 x 64]
```

`W_skip` is not an attention head. It is the layer's internal root/skip
projection, preserving the territory's own current information.

Then `Encoder.forward` applies the outer residual connection:

```text
H_next = H + ReLU(conv_output)  [42 x 64]
```

The encoder repeats this four times:

```text
H^0 = X -> H^1 -> H^2 -> H^3 -> H^4
```

## Non-attack action injection

Reinforce, occupy, and fortify actions do not mark an edge. They modify the
proposed-army-delta feature in the affected rows of `X` before `W_in` creates
`H`. Thus these actions affect the node representations from the first layer.

## Final board embedding

After the fourth layer, the project pools the territory rows and appends the
global vector:

```text
mean(H_final)                    [1 x 64]
max(H_final)                     [1 x 64]
u                                [1 x 35]

g = [mean(H_final) | max(H_final) | u]
                                  [1 x 163]
```

The phase-specific MLP head maps `g` to `Q(s, a)` for DQN or a policy logit
for PPO.

## Learned parameters in this encoder

`V` is not a learned parameter; it is the calculated matrix `V = H W_V`.
PyTorch stores linear weights as `[output_features x input_features]`, the
transpose of the mathematical orientation used in the equations above.

| Weight symbol | PyTorch weight | Weight shape | Bias | Total count | Purpose |
|---|---|---:|---|---:|---|
| `W_in` | `input_proj.weight` | `[64 x 15]` | `input_proj.bias`: `[64]` | 1,024 | Maps raw `X` into `H`. |
| `W_Q` (each of 4 layers) | `lin_query.weight` | `[64 x 64]` | `lin_query.bias`: `[64]` | 4,160 | Creates queries. |
| `W_K` (each of 4 layers) | `lin_key.weight` | `[64 x 64]` | `lin_key.bias`: `[64]` | 4,160 | Creates keys. |
| `W_V` (each of 4 layers) | `lin_value.weight` | `[64 x 64]` | `lin_value.bias`: `[64]` | 4,160 | Creates values. |
| `W_E` (each of 4 layers) | `lin_edge.weight` | `[64 x 2]` | None | 128 | Projects action/edge features. |
| `W_skip` (each of 4 layers) | `lin_skip.weight` | `[64 x 64]` | `lin_skip.bias`: `[64]` | 4,160 | Internal root/skip projection. |

Each attention layer has 16,768 learned scalar parameters. The four layers
have 67,072; with `W_in`, the shared encoder has **68,096 learned scalar
parameters**. This excludes the separate phase-specific MLP heads.

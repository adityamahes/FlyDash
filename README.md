# TetriFLY

Playing Tetris with the wiring of a real fly brain.

The connectome of the male fruit fly (~166,700 neurons, ~125M synapses) is
loaded from neuPrint, turned into a fixed matrix of signed synaptic strengths,
and simulated as leaky integrate-and-fire neurons. **That matrix is never
trained.** The only things that learn are two small linear layers bolted onto
either end of it:

```
 board ──► Encoder ──► visual neurons ┐
                                       │  frozen connectome, simulated as
                                       │  spiking neurons for 20 timesteps
           Action ◄── Decoder ◄── motor neurons ┘
              │
              └──► Tetris ──► outcome ──► reward current ──► dopamine neurons
```

There is no backpropagation through the brain: a spike is a hard threshold
comparison, which has no usable gradient. The encoder and decoder are optimised
as a black box with evolutionary strategies.

## Running it

### Setup

Everything runs on the real connectome, so the token and a one-off build are
required; there is no synthetic fallback.

You need a free [neuPrint](https://neuprint.janelia.org) token (log in, then
your account page). Put it in `tetrifly/local_config.py`, which is gitignored
so the token is never committed:

```python
TOKEN = 'paste-your-token-here'
```

Two other ways to supply it, if you would rather not keep it on disk — an
environment variable, or a one-off flag:

```bash
export NEUPRINT_TOKEN=paste-your-token-here   # Windows: $env:NEUPRINT_TOKEN="..."
python main.py build --token paste-your-token-here
```

A `token=...` argument wins over `$NEUPRINT_TOKEN`, which wins over
`local_config.TOKEN`.

Check that the token works and that the dataset still matches what the
pipeline expects — this takes seconds, and is worth doing before the build:

```bash
python check_access.py
```

Then fetch and cache the connectome once. This is a large download and the
preprocessing takes a while:

```bash
python main.py build
```

That writes `data/connectome.pt`, which `serve` and `train` load by default:

```bash
python main.py serve                    # then open http://127.0.0.1:8000
python main.py train --generations 25
```

`serve` plays episode after episode in a background thread and streams each
frame to the browser. `--delay` is the pause after every frame, in seconds:

```bash
python main.py serve --delay 0.4        # slow, easy to follow
python main.py serve --delay 0          # as fast as the simulation runs
python main.py serve --port 9000
```

The viewer binds to loopback only; it is for this machine, not the network.

Expect it to be slow per frame: each Tetris frame is 25 sparse matrix
multiplies against ~166,700 neurons. Start with `--delay 0` and a low
`--frames` to see how fast your machine manages.

### Watching it

The browser view draws the board as a grid of square cells (styled after
[robertrypula/simple-tetris](https://github.com/robertrypula/simple-tetris)) and,
alongside it, what the brain is doing: the size of each neuron population, how
hard the motor neurons are firing, the five decoder logits as bars with the
winning one highlighted, and the reward rule's verdict on the last frame
alongside the dopamine current it became. Green cells are locked; amber is the piece the brain
still controls.

## The code

One class per concept:

| File | Class | What it does |
|---|---|---|
| `tetrifly/brain.py` | `Connectome` | neuPrint tables → frozen sparse wiring matrix |
| | `SpikingNetwork` | leaky integrate-and-fire simulation over that matrix |
| `tetrifly/tetris.py` | `Piece`, `TetrisGame` | the game |
| `tetrifly/agent.py` | `Encoder`, `Decoder` | the two trainable layers |
| | `Reward` | board outcome -> one scalar, for brain and trainer alike |
| | `Agent` | one frame: see, simulate, act, be rewarded |
| | `Trainer` | evolutionary strategies |
| `tetrifly/web.py` | `GameRunner` | plays in a thread, publishes a snapshot |
| | `Handler` | serves the page and `/state` (stdlib only) |

Tunable constants live as class attributes next to the code that reads them —
`SpikingNetwork.THRESHOLD`, `Connectome.WEIGHT_PER_SYNAPSE`, `Trainer.SIGMA`.
The neuPrint token is the exception: it lives in the gitignored
`tetrifly/local_config.py`.

`check_access.py` verifies the token, the dataset schema, the cell-type
regexes and the synapse signs against the live server, without downloading
the adjacency table.

### Building the wiring matrix

`Connectome.__init__` runs the pipeline as one method per step, so each stage
can be called and inspected on its own:

1. `_collapse_roi_duplicates` — neuPrint reports one row per (sender, receiver,
   brain region), so the same pair appears many times. Sum their synapse counts.
2. `_signed_strengths` — the sender's predicted neurotransmitter decides whether
   its synapses excite (+1) or inhibit (−1). Low-confidence predictions and
   neuromodulators fall back to the default sign.
3. `_build_matrix` — map arbitrary body IDs onto contiguous indices and assemble
   the sparse matrix. `matrix[i, j]` is how hard a spike of neuron `i` pushes
   neuron `j`.
4. `_normalise_input` — equalise how hard each neuron is driven. See below.
5. `_find_population` — locate the visual, motor and dopamine neurons by cell
   type.

## Two things that are not in the design

Both were needed to make it actually run, and both were measured rather than
assumed. They are recorded in `SIMPLIFICATIONS.txt` along with everything that
was cut.

**Input normalisation.** The design gives every neuron the same firing
threshold, but real neurons differ by orders of magnitude in how many synapses
they receive. Under one threshold the well-connected majority is driven far past
it by recurrent activity alone, the network saturates, and the sensory current
from the encoder is lost entirely. Measured: without normalisation, perturbing
the encoder had *no effect at all* on the motor neurons — every candidate in a
generation produced a byte-identical episode and evolution saw a perfectly flat
landscape. Normalising divides each neuron's incoming column by its own total,
leaving the relative strength and sign of every synapse untouched. Pass
`normalise=False` for the literal model.

**Encoder gain.** A fresh linear layer reading a sparse binary board emits
currents near 0.1, against a threshold of 1.0. `Encoder.SCALE` lifts that into a
range that can modulate spiking; training can shrink it back down.

## Status

The machinery runs end to end and the search landscape is non-degenerate — the
agent moves pieces around, fitness varies between candidates, and short runs
occasionally clear a line. It does **not** yet learn to play: evolutionary
strategies over ~40,000 parameters needs far more episodes than a handful of
generations, and each candidate is scored on a single noisy episode.

See `SIMPLIFICATIONS.txt` for what this first iteration deliberately leaves out
(mushroom-body plasticity, next-piece preview, hold, wall kicks, scoring tiers)
and what it would take to add each back. Hard drop and a graded reward have
since been added; see `Action.DROP` and `Reward`.

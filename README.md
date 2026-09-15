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

Nothing here needs a neuPrint token to run — `play` and `train` fall back to a
small synthetic brain built through the exact same pipeline.

```bash
python main.py play --render          # watch one episode
python main.py train --generations 25 # optimise the encoder and decoder
python -m unittest test_tetrifly -v   # 65 tests
```

For the real thing you need a [neuPrint](https://neuprint.janelia.org) token.
The download is large, so it is cached:

```bash
export NEUPRINT_TOKEN=...
python main.py build --output data/connectome.pt
python main.py play --connectome data/connectome.pt --render
```

## The code

Three modules, one class per concept:

| File | Class | What it does |
|---|---|---|
| `tetrifly/brain.py` | `Connectome` | neuPrint tables → frozen sparse wiring matrix |
| | `SpikingNetwork` | leaky integrate-and-fire simulation over that matrix |
| `tetrifly/tetris.py` | `Piece`, `TetrisGame` | the game |
| `tetrifly/agent.py` | `Encoder`, `Decoder` | the two trainable layers |
| | `Agent` | one frame: see, simulate, act, be rewarded |
| | `Trainer` | evolutionary strategies |

Tunable constants live as class attributes next to the code that reads them —
`SpikingNetwork.THRESHOLD`, `Connectome.WEIGHT_PER_SYNAPSE`, `Trainer.SIGMA`.

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
(mushroom-body plasticity, next-piece preview, hold, drops, wall kicks, scoring
tiers, graded reward) and what it would take to add each back.

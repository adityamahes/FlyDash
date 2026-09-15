"""The fly half of TetriFLY: a frozen connectome and the simulation that runs it.

Two classes live here:

``Connectome``
    Turns the two tables neuPrint returns into a fixed sparse matrix of signed
    synaptic strengths, and finds the neuron populations the agent is allowed to
    touch. Nothing in it ever changes during training.

``SpikingNetwork``
    Runs that matrix forward as leaky integrate-and-fire neurons: accumulate
    charge, leak it away, spike at threshold, reset.
"""

import os
from pathlib import Path

import pandas as pd
import torch

try:
    from .local_config import TOKEN as _LOCAL_TOKEN
except ImportError:
    # local_config.py is gitignored, so a fresh clone will not have it.
    _LOCAL_TOKEN = ''


class Connectome:
    """A frozen fly brain built from neuPrint tables.

    The build pipeline runs in ``__init__`` and is split into one method per
    step, so each stage can be inspected on its own:

    1. ``_collapse_roi_duplicates`` - neuPrint reports one row per (pre, post,
       brain region), so the same pair of neurons appears several times. Sum
       their synapse counts into a single row per pair.
    2. ``_signed_strengths`` - a neuron's predicted neurotransmitter decides
       whether its synapses excite (+1) or inhibit (-1).
    3. ``_build_matrix`` - map the arbitrary body IDs onto contiguous indices
       and assemble the sparse N x N matrix.
    4. ``_normalise_input`` - equalise how hard each neuron is driven (see the
       method's docstring for why this is necessary).
    5. ``_find_population`` - locate the visual, motor and reward neurons by
       cell type.

    Attributes:
        matrix: Sparse (N, N) tensor. ``matrix[i, j]`` is how strongly a spike
            of neuron i pushes the membrane potential of neuron j: positive
            excites, negative inhibits. Row = sender, column = receiver.
        types: Cell type name per matrix index.
        positions: Optional (N, 3) soma coordinates in nanometres, one row per
            matrix index, or None. NaN for a cell type neuPrint has no soma
            location for. Used only for drawing; the simulation ignores it.
        visual: Indices of the visual neurons, where the encoder injects.
        motor: Indices of the descending neurons, which the decoder reads.
        reward: Indices of the dopamine neurons, where reward is injected.
    """

    # --- data source ---
    SERVER = 'neuprint.janelia.org'
    DATASET = 'male-cns:v1.0'

    # Auth token, needed only by ``from_neuprint``. The first non-empty of
    # these wins: the ``token=...`` argument, then $NEUPRINT_TOKEN, then
    # TOKEN, which comes from the gitignored local_config.py.
    TOKEN_ENV_VAR = 'NEUPRINT_TOKEN'
    TOKEN = _LOCAL_TOKEN

    # --- synapse signs ---
    # Neuromodulators (dopamine, octopamine, serotonin) are deliberately absent:
    # their effect is not a simple fast excitation or inhibition, so they fall
    # back to DEFAULT_SIGN, as do low-confidence predictions.
    TRANSMITTER_SIGNS = {'acetylcholine': +1, 'gaba': -1, 'glutamate': -1}
    DEFAULT_SIGN = +1
    MIN_CONFIDENCE = 0.5

    #: Membrane potential delivered by one synapse. Ten synapses therefore add
    #: up to 1.0, which is exactly the firing threshold.
    WEIGHT_PER_SYNAPSE = 0.1

    #: Net input a neuron gets when all of its senders fire at once, in units
    #: of the firing threshold. Net, not excitatory: see _normalise_input.
    TARGET_INPUT = 2.0

    #: Ceiling on the gain _normalise_input may apply. A neuron whose
    #: excitation and inhibition nearly cancel has a net drive near zero, and
    #: asking for TARGET_INPUT would multiply its synapses by millions. Those
    #: neurons stay under-driven instead, which is the honest outcome: the
    #: anatomy really does leave them balanced.
    MAX_GAIN = 20.0

    # --- cell types of the populations the agent uses ---
    VISUAL_TYPES = r'^(?:R[1-6]|Mi1)'    # photoreceptors and early medulla
    MOTOR_TYPES = r'^(?:DNp01|DNg)'      # descending neurons that drive movement
    REWARD_TYPES = r'^PPL1'              # dopaminergic reward/punishment neurons

    def __init__(self, neurons, connections, normalise=True):
        """Build the connectome from raw neuPrint tables.

        Args:
            neurons: One row per neuron, with columns ``bodyId``, ``type``,
                ``predictedNt`` and ``predictedNtProb``.
            connections: One row per (pre, post, brain region), with columns
                ``bodyId_pre``, ``bodyId_post`` and ``weight``.
            normalise: Whether to equalise each neuron's total input drive.
        """
        self.types = neurons['type'].reset_index(drop=True)

        pairs = self._collapse_roi_duplicates(connections)
        strengths = self._signed_strengths(neurons, pairs)
        self.matrix = self._build_matrix(neurons, pairs, strengths)
        if normalise:
            self.matrix = self._normalise_input(self.matrix)

        self.visual = self._find_population(self.VISUAL_TYPES)
        self.motor = self._find_population(self.MOTOR_TYPES)
        self.reward = self._find_population(self.REWARD_TYPES)

    def __len__(self):
        """Return the number of neurons N."""
        return self.matrix.shape[0]

    def __repr__(self):
        return (f'<Connectome {len(self)} neurons, {self.matrix._nnz()} connections, '
                f'{len(self.visual)} visual, {len(self.motor)} motor, '
                f'{len(self.reward)} reward>')

    # ------------------------------------------------------------------
    # Build steps
    # ------------------------------------------------------------------

    @staticmethod
    def _collapse_roi_duplicates(connections):
        """Sum a connection's synapse counts across the brain regions it spans.

        Args:
            connections: Raw per-region connection rows.

        Returns:
            One row per unique (pre, post) pair, with ``weight`` summed.
        """
        return connections.groupby(['bodyId_pre', 'bodyId_post'],
                                   as_index=False)['weight'].sum()

    @classmethod
    def _signed_strengths(cls, neurons, pairs):
        """Turn synapse counts into signed membrane-potential contributions.

        A neuron releases the same transmitter at all of its output synapses, so
        the sign is a property of the sending neuron alone.

        Args:
            neurons: Neuron rows, used for the transmitter lookup.
            pairs: Collapsed connection rows.

        Returns:
            One float per connection row: synapse count x sign x
            WEIGHT_PER_SYNAPSE.
        """
        transmitters = neurons['predictedNt']
        # Gate on confidence only where there is confidence data to gate on.
        # male-cns:v1.0 ships the predictedNtProb column but leaves it null for
        # every neuron; an all-null gate would discard every transmitter and
        # fall the whole brain back to DEFAULT_SIGN, losing all inhibition.
        confidence = neurons.get('predictedNtProb')
        if confidence is not None and confidence.notna().any():
            confident = (confidence >= cls.MIN_CONFIDENCE).fillna(False)
            transmitters = transmitters.where(confident)

        signs = (transmitters.str.lower()
                 .map(cls.TRANSMITTER_SIGNS)
                 .fillna(cls.DEFAULT_SIGN))
        sign_by_id = pd.Series(signs.to_numpy(), index=neurons['bodyId'].to_numpy())

        return (pairs['weight']
                * pairs['bodyId_pre'].map(sign_by_id).fillna(cls.DEFAULT_SIGN)
                * cls.WEIGHT_PER_SYNAPSE)

    @staticmethod
    def _build_matrix(neurons, pairs, strengths):
        """Assemble the sparse N x N matrix of signed synaptic strengths.

        Body IDs are arbitrary large integers, so they are first mapped onto
        contiguous positions 0..N-1.

        Args:
            neurons: Neuron rows; their order defines the matrix indices.
            pairs: Collapsed connection rows.
            strengths: One signed strength per connection row.

        Returns:
            A coalesced sparse COO tensor holding only the real connections.

        Raises:
            ValueError: If strengths are not aligned with the connection rows.
        """
        if len(strengths) != len(pairs):
            raise ValueError(f'Got {len(strengths)} strengths for {len(pairs)} '
                             f'connections; they must line up row for row.')

        index_by_id = pd.Series(range(len(neurons)), index=neurons['bodyId'].to_numpy())
        rows = pairs['bodyId_pre'].map(index_by_id)
        cols = pairs['bodyId_post'].map(index_by_id)

        # Drop connections whose endpoints are missing from the neuron table.
        known = (rows.notna() & cols.notna()).to_numpy()

        positions = torch.stack([
            torch.as_tensor(rows.to_numpy()[known].astype('int64')),
            torch.as_tensor(cols.to_numpy()[known].astype('int64')),
        ])
        values = torch.as_tensor(strengths.to_numpy()[known].astype('float32'))

        return torch.sparse_coo_tensor(positions, values,
                                       size=(len(neurons), len(neurons))).coalesce()

    @classmethod
    def _normalise_input(cls, matrix):
        """Rescale each neuron's incoming synapses to a common total drive.

        This is a deliberate departure from the raw anatomy, and it is needed
        because every neuron here shares one firing threshold. Real neurons
        differ by orders of magnitude in how many synapses they receive, so
        under a single threshold the well-connected majority is pushed far past
        it by recurrent activity alone. The network then saturates and the
        sensory current the encoder injects is lost entirely - measured on the
        connectome, the encoder has no detectable effect on the motor neurons,
        and there is nothing for training to optimise.

        Dividing each neuron's incoming column by its own total leaves the
        relative strength and the sign of every synapse it receives untouched;
        only the overall gain of its input is set to a common value.

        The total that gets equalised is the NET one, excitation plus
        inhibition, not excitation alone. Normalising excitation by itself
        looks right and is not: 37% of the synapses here are inhibitory, so
        setting excitation to 2.0 leaves a median net drive of 0.52 and an
        all-neuron mean of 0.007 - below the firing threshold of 1.0 even if
        every presynaptic partner fired at once. Nothing could reach threshold
        from recurrent input at all, and the descending neurons the decoder
        reads sat permanently just under it, firing 2 out of 862. That was
        invisible while a bug made every synapse excitatory, because then
        excitation and net were the same number.

        Neurons whose inhibition outweighs their excitation (29.6% of this
        connectome) have no positive gain that would reach threshold. They are
        left as they are and stay quiet unless something injects current
        directly.

        Args:
            matrix: Sparse (N, N) matrix in [sender, receiver] orientation.

        Returns:
            A matrix of the same shape and sparsity.

        Note:
            Not idempotent. For a neuron whose gain hit MAX_GAIN the first
            pass leaves net drive short of TARGET_INPUT, so a second pass
            scales it again. Re-running this on an already normalised matrix
            is therefore only approximately the same as building from raw
            synapse counts, and differs exactly on the gain-capped neurons.
        """
        receivers = matrix.indices()[1]
        values = matrix.values()
        size = matrix.shape[0]

        net = torch.zeros(size).index_add_(0, receivers, values)
        gain = (cls.TARGET_INPUT / net.clamp(min=1e-6)).clamp(max=cls.MAX_GAIN)
        scale = torch.where(net > 0, gain, torch.ones(size))

        return torch.sparse_coo_tensor(matrix.indices(), values * scale[receivers],
                                       size=matrix.shape).coalesce()

    def _find_population(self, pattern):
        """Find the matrix indices of every neuron whose cell type matches.

        Args:
            pattern: Regular expression matched against the cell type name.

        Returns:
            A 1-D long tensor of matrix indices.

        Raises:
            ValueError: If nothing matches, which would leave the agent blind,
                paralysed or unable to feel reward.
        """
        matches = self.types.str.contains(pattern, na=False, regex=True)
        indices = torch.as_tensor(matches.to_numpy().nonzero()[0], dtype=torch.long)
        if len(indices) == 0:
            raise ValueError(f'No neuron type matches {pattern!r}. Check the '
                             f'patterns against the cell types in this dataset.')
        return indices

    # ------------------------------------------------------------------
    # Loading and saving
    # ------------------------------------------------------------------

    @classmethod
    def from_neuprint(cls, token=None):
        """Download the connectome from neuPrint and build it.

        Args:
            token: neuPrint auth token. Defaults to the NEUPRINT_TOKEN
                environment variable, then to ``TOKEN`` from local_config.py.

        Returns:
            The built Connectome.

        Raises:
            RuntimeError: If no token is available.
        """
        # Imported here so the rest of the package works without neuprint
        # installed or a network connection.
        from neuprint import Client, fetch_adjacencies, fetch_neurons

        token = token or os.environ.get(cls.TOKEN_ENV_VAR) or cls.TOKEN
        if not token:
            raise RuntimeError(f'No neuPrint token. Set TOKEN in '
                               f'tetrifly/local_config.py, export '
                               f'${cls.TOKEN_ENV_VAR}, or pass token=...')

        client = Client(cls.SERVER, dataset=cls.DATASET, token=token)
        neurons, _ = fetch_neurons(client=client)
        # fetch_adjacencies returns (neurons, connections); we want the second.
        _, connections = fetch_adjacencies(sources=neurons.bodyId,
                                           targets=neurons.bodyId, client=client)
        return cls(neurons, connections)

    def save(self, path):
        """Write the built connectome to disk.

        Refetching ~125M connection rows is far slower than reloading them.

        Args:
            path: Destination file; parent directories are created.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        torch.save({'matrix': self.matrix, 'types': self.types.tolist(),
                    'visual': self.visual, 'motor': self.motor,
                    'reward': self.reward, 'positions': self.positions}, path)

    @classmethod
    def load(cls, path):
        """Reload a connectome written by :meth:`save`.

        Args:
            path: File previously written by :meth:`save`.

        Returns:
            The deserialised Connectome.
        """
        data = torch.load(path, weights_only=False)
        connectome = cls.__new__(cls)          # skip the build pipeline
        connectome.matrix = data['matrix'].coalesce()
        connectome.types = pd.Series(data['types'])
        connectome.visual = data['visual']
        connectome.motor = data['motor']
        connectome.reward = data['reward']
        # Optional, and absent from files built before positions were stored.
        connectome.positions = data.get('positions')
        return connectome


class SpikingNetwork:
    """Leaky integrate-and-fire simulation over a frozen connectome.

    Every neuron is the same simple unit, so all the structure in the agent's
    behaviour comes from the real wiring rather than from per-neuron tuning.
    One timestep is::

        potential = potential * (1 - DECAY) + recurrent + external
        spike     = potential > THRESHOLD
        potential = 0 where it spiked

    Spikes are binary, so there is no usable gradient through this class. That
    is why the encoder and decoder around it are trained without backpropagating
    through the brain.

    Attributes:
        potential: Continuous (N,) membrane potential of every neuron.
        spikes: Binary (N,) output of the most recent timestep.
    """

    #: Fraction of the membrane potential lost per timestep (= dt / tau).
    #: tau = 5 ms (published) with dt = 0.25 ms gives 0.05.
    DECAY = 0.05

    #: Potential a neuron must exceed to fire.
    THRESHOLD = 1.0

    def __init__(self, connectome):
        """Prepare a simulation over one connectome's wiring.

        Args:
            connectome: The frozen brain to simulate.
        """
        self.size = len(connectome)
        # The simulation needs total input *per receiving* neuron, a column sum
        # of the [sender, receiver] matrix. Transposing once here makes each
        # timestep an ordinary matrix-vector product.
        #
        # CSR, not COO: this matmul is the hot loop of the whole project, run
        # 25 times per game frame, and torch's COO kernel chases ~25.9M
        # unordered (row, col, value) triples through scattered memory. CSR
        # groups them by row so the same arithmetic walks memory in order --
        # measured at 73ms against 1264ms per step on male-cns:v1.0, for
        # bitwise-identical potentials. Build and storage stay COO, because
        # _normalise_input needs .indices() and .values().
        self._incoming = connectome.matrix.t().coalesce().to_sparse_csr()
        self.reset()

    def reset(self):
        """Clear all potentials and spikes, as between episodes."""
        self.potential = torch.zeros(self.size)
        self.spikes = torch.zeros(self.size)

    def step(self, external=None):
        """Advance the whole network by one timestep.

        Args:
            external: Optional (N,) current injected from outside the brain -
                sensory drive or reward. ``None`` means no external input.

        Returns:
            The binary (N,) spike vector produced this timestep.

        Raises:
            ValueError: If the external current is not shaped (N,).
        """
        if external is None:
            external = torch.zeros(self.size)
        elif external.shape != (self.size,):
            raise ValueError(f'External current must be shape ({self.size},), '
                             f'got {tuple(external.shape)}.')

        recurrent = torch.sparse.mm(self._incoming, self.spikes.unsqueeze(1)).squeeze(1)
        self.potential = self.potential * (1 - self.DECAY) + recurrent + external

        self.spikes = (self.potential > self.THRESHOLD).float()
        self.potential = self.potential * (1 - self.spikes)   # reset what fired
        return self.spikes

    def run(self, external, timesteps):
        """Hold an input steady for several timesteps and measure firing rates.

        The timestep is much shorter than the membrane time constant, so a
        single step says almost nothing; one game frame is worth many steps.

        Args:
            external: Current injected on every one of those timesteps.
            timesteps: How many steps to simulate. Must be positive.

        Returns:
            An (N,) tensor of firing rates in [0, 1]: the fraction of timesteps
            on which each neuron spiked.

        Raises:
            ValueError: If timesteps is not positive.
        """
        if timesteps <= 0:
            raise ValueError(f'timesteps must be positive, got {timesteps}.')

        total = torch.zeros(self.size)
        for _ in range(timesteps):
            total += self.step(external)
        return total / timesteps

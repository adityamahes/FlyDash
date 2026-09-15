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
import random
from pathlib import Path

import pandas as pd
import torch


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
        visual: Indices of the visual neurons, where the encoder injects.
        motor: Indices of the descending neurons, which the decoder reads.
        reward: Indices of the dopamine neurons, where reward is injected.
    """

    # --- data source ---
    SERVER = 'neuprint.janelia.org'
    DATASET = 'male-cns:v1.0'
    TOKEN_ENV_VAR = 'NEUPRINT_TOKEN'

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

    #: Total excitatory input a neuron gets when all of its senders fire at
    #: once, in units of the firing threshold. See _normalise_input.
    TARGET_INPUT = 2.0

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
        if 'predictedNtProb' in neurons:
            confident = (neurons['predictedNtProb'] >= cls.MIN_CONFIDENCE).fillna(False)
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

        Args:
            matrix: Sparse (N, N) matrix in [sender, receiver] orientation.

        Returns:
            A matrix of the same shape and sparsity. Neurons with no excitatory
            input are left alone.
        """
        senders, receivers = matrix.indices()
        values = matrix.values()
        size = matrix.shape[0]

        excitation = torch.zeros(size).index_add_(0, receivers, values.clamp(min=0))
        scale = torch.where(excitation > 0,
                            cls.TARGET_INPUT / excitation.clamp(min=1e-6),
                            torch.ones(size))

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
                environment variable.

        Returns:
            The built Connectome.

        Raises:
            RuntimeError: If no token is available.
        """
        # Imported here so the rest of the package works without neuprint
        # installed or a network connection.
        from neuprint import Client, fetch_adjacencies, fetch_neurons

        token = token or os.environ.get(cls.TOKEN_ENV_VAR)
        if not token:
            raise RuntimeError(f'No neuPrint token. Set ${cls.TOKEN_ENV_VAR} '
                               f'or pass token=...')

        client = Client(cls.SERVER, dataset=cls.DATASET, token=token)
        neurons, _ = fetch_neurons(client=client)
        # fetch_adjacencies returns (neurons, connections); we want the second.
        _, connections = fetch_adjacencies(sources=neurons.bodyId,
                                           targets=neurons.bodyId, client=client)
        return cls(neurons, connections)

    @classmethod
    def synthetic(cls, seed=0):
        """Build a small fake brain, so the project runs with no token.

        The generated tables have the same schema as neuPrint's, duplicate
        region rows and all, and go through the ordinary build pipeline. Nothing
        is shortcut, so a synthetic run exercises the same code as a real one.

        Args:
            seed: Seed for reproducible fixtures.

        Returns:
            A Connectome of roughly 200 neurons.
        """
        rng = random.Random(seed)
        counts = {'R1-R6': 60, 'Mi1': 40, 'DNp01': 4, 'DNg100': 8,
                  'PPL1-a': 6, 'Unknown': 82}
        transmitters = ['acetylcholine', 'gaba', 'glutamate', 'dopamine', 'unclear']
        regions = ['ME', 'LO', 'MB', 'CX']

        body_ids, types = [], []
        for cell_type, count in counts.items():
            for _ in range(count):
                body_ids.append(1_000_000 + 7 * len(body_ids))  # sparse, like real IDs
                types.append(cell_type)

        neurons = pd.DataFrame({
            'bodyId': body_ids,
            'type': types,
            'predictedNt': [rng.choice(transmitters) for _ in body_ids],
            'predictedNtProb': [rng.uniform(0.2, 1.0) for _ in body_ids],
        })
        rows = 8 * len(body_ids)
        connections = pd.DataFrame({
            'bodyId_pre': [rng.choice(body_ids) for _ in range(rows)],
            'bodyId_post': [rng.choice(body_ids) for _ in range(rows)],
            'roi': [rng.choice(regions) for _ in range(rows)],
            'weight': [rng.randint(1, 30) for _ in range(rows)],
        })
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
                    'reward': self.reward}, path)

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
        self._incoming = connectome.matrix.t().coalesce()
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

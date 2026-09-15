"""The agent: Tetris wired into the fly brain, and the trainer that tunes it.

Four classes:

``Encoder``   board -> current injected at the visual neurons (trainable)
``Decoder``   motor neuron firing rates -> four action logits (trainable)
``Agent``     runs one frame: see, simulate, act, be rewarded
``Trainer``   evolutionary strategies over the encoder and decoder

The encoder and decoder are the only trainable parts. The connectome between
them is frozen anatomy, and the reward injection is a fixed rule.
"""

import torch
from torch import nn

from .brain import SpikingNetwork
from .tetris import Action, TetrisGame


class Encoder(nn.Module):
    """Learns to turn a Tetris board into current for the visual neurons.

    Nothing about the mapping is hand-designed: which neurons a given board cell
    drives is exactly what training has to discover.
    """

    #: Gain on the raw layer output. A freshly initialised linear layer reading
    #: a sparse binary board emits currents near 0.1, far below the firing
    #: threshold of 1.0, so the network simply ignores it. This lifts the
    #: injection into a range that can actually modulate spiking; training is
    #: free to shrink the weights back down.
    SCALE = 10.0

    def __init__(self, connectome):
        """Create an encoder for one connectome's visual population.

        Args:
            connectome: The brain to inject into.
        """
        super().__init__()
        self.register_buffer('visual', connectome.visual)
        self.size = len(connectome)
        self.layer = nn.Linear(TetrisGame.OBSERVATION_SIZE, len(connectome.visual))

    def forward(self, observation):
        """Encode one board into a full-brain injection vector.

        Args:
            observation: The binary board state.

        Returns:
            An (N,) current vector, zero everywhere except the visual neurons.
        """
        current = torch.zeros(self.size)
        current[self.visual] = self.layer(observation) * self.SCALE
        return current


class Decoder(nn.Module):
    """Learns to read descending-neuron activity as one of four controls.

    The chosen action is the argmax of the logits, so only their ordering
    matters.
    """

    def __init__(self, connectome):
        """Create a decoder for one connectome's motor population.

        Args:
            connectome: The brain to read from.
        """
        super().__init__()
        self.register_buffer('motor', connectome.motor)
        self.layer = nn.Linear(len(connectome.motor), len(Action))

    def forward(self, rates):
        """Decode brain-wide firing rates into action logits.

        Args:
            rates: An (N,) tensor of per-neuron firing rates. Only the motor
                subset is read.

        Returns:
            A (4,) tensor of logits, ordered to match Action.
        """
        return self.layer(rates[self.motor])


class Agent:
    """Couples the trainable encoder and decoder to a frozen brain.

    One frame of play is the loop the design describes:

    1. Render the board.
    2. Encode it into current at the visual neurons.
    3. Simulate the connectome for many timesteps under that steady input.
    4. Decode the motor firing rates into one of four controls.
    5. Apply that control to the game.
    6. Turn the outcome into current at the dopamine neurons and simulate a few
       more timesteps, so the teaching signal reaches the brain.

    Attributes:
        connectome: The frozen brain.
        network: The spiking simulation over it.
        encoder: Trainable board -> visual current mapping.
        decoder: Trainable motor rates -> action logits mapping.
        rates: Firing rates from the most recent frame, kept for inspection.
        logits: Action logits from the most recent frame.
    """

    #: Timesteps simulated per Tetris frame. The timestep is far shorter than
    #: the membrane time constant, so the network needs several to settle.
    FRAME_TIMESTEPS = 20

    #: Extra timesteps after the reward current is injected, so the dopamine
    #: signal actually propagates.
    REWARD_TIMESTEPS = 5

    #: Fixed current injected at the dopamine neurons. Not learned: real
    #: dopamine is an unconditioned "that was good / bad" teaching signal, so it
    #: stays as external ground truth. Large enough to drive those neurons
    #: regardless of what the rest of the brain is doing.
    LINE_REWARD = 2.0
    GAME_OVER_PUNISHMENT = -2.0

    def __init__(self, connectome):
        """Build an agent around one connectome.

        Args:
            connectome: The frozen brain to run on.
        """
        self.connectome = connectome
        self.network = SpikingNetwork(connectome)
        self.encoder = Encoder(connectome)
        self.decoder = Decoder(connectome)
        self.rates = torch.zeros(len(connectome))
        self.logits = torch.zeros(len(Action))

    def reset(self):
        """Clear the simulation, as at the start of an episode."""
        self.network.reset()

    def play_frame(self, game):
        """Perceive, act, and be taught, for a single frame.

        Args:
            game: The game to observe and step. Mutated in place.

        Returns:
            The Action chosen this frame.
        """
        with torch.no_grad():
            sensory = self.encoder(game.observation())
            self.rates = self.network.run(sensory, self.FRAME_TIMESTEPS)
            self.logits = self.decoder(self.rates)

        action = Action(int(self.logits.argmax()))
        lines, over = game.step(action)
        self._inject_reward(lines, over)
        return action

    def _inject_reward(self, lines, over):
        """Deliver the teaching signal and let it propagate.

        Args:
            lines: Rows cleared this frame.
            over: Whether the game ended this frame.
        """
        amount = self.LINE_REWARD * lines
        if over:
            amount += self.GAME_OVER_PUNISHMENT

        current = torch.zeros(len(self.connectome))
        if amount:
            current[self.connectome.reward] = amount
        self.network.run(current, self.REWARD_TIMESTEPS)

    def play_episode(self, game, max_frames=500):
        """Play until the game ends or the frame budget runs out.

        This runs flat out, for training. The browser view drives
        :meth:`play_frame` itself so it can pace the game and publish each frame.

        Args:
            game: A fresh game to play. Mutated in place.
            max_frames: Frame budget, so a policy that never tops out still
                terminates.

        Returns:
            (lines_cleared, frames_played) for the episode.
        """
        self.reset()

        frames = 0
        for _ in range(max_frames):
            self.play_frame(game)
            frames += 1
            if game.over:
                break

        return game.lines, frames

    # ------------------------------------------------------------------
    # Parameters, as one flat vector for the trainer
    # ------------------------------------------------------------------

    def _parameters(self):
        """Yield every trainable tensor, encoder first then decoder."""
        yield from self.encoder.parameters()
        yield from self.decoder.parameters()

    def get_parameters(self):
        """Flatten every trainable parameter into one vector.

        Returns:
            A 1-D copy of all encoder and decoder parameters.
        """
        return torch.cat([p.detach().reshape(-1) for p in self._parameters()])

    def set_parameters(self, vector):
        """Write a flat vector back into the trainable parameters.

        Args:
            vector: A vector laid out as :meth:`get_parameters` produces.

        Raises:
            ValueError: If the vector is the wrong length.
        """
        expected = sum(p.numel() for p in self._parameters())
        if vector.numel() != expected:
            raise ValueError(f'Expected {expected} parameters, got {vector.numel()}.')

        at = 0
        with torch.no_grad():
            for parameter in self._parameters():
                chunk = vector[at:at + parameter.numel()]
                parameter.copy_(chunk.reshape(parameter.shape))
                at += parameter.numel()


class Trainer:
    """Trains the encoder and decoder with evolutionary strategies.

    Backpropagation is not available. The encoder reaches the decoder only
    through thousands of spiking neurons whose hard threshold has a gradient of
    zero almost everywhere, so the agent is treated as a black box: perturb the
    parameters, play a game, keep what scored better.

    Each generation draws a few random directions, evaluates the agent at
    ``+sigma`` and ``-sigma`` along each one, and steps toward the better side.
    Mirroring the samples cancels most of the noise in that estimate.

    Attributes:
        agent: The agent being trained; its parameters are updated in place.
    """

    #: Mirrored perturbation pairs per generation; each costs two episodes.
    PAIRS = 8

    #: Standard deviation of the Gaussian parameter noise.
    SIGMA = 0.05

    #: Step size along the estimated gradient.
    LEARNING_RATE = 0.02

    #: Frame budget per episode.
    MAX_FRAMES = 500

    #: Weight on cleared lines in the fitness. Surviving frames are worth 1
    #: each. Cleared lines alone would be useless as a training signal: an
    #: untrained policy never clears one, so every candidate would score zero
    #: and there would be nothing to climb. Survival time gives a gradient from
    #: the first generation, and this weight keeps a cleared line worth far more
    #: than the frames it took.
    LINE_WEIGHT = 200.0

    def __init__(self, agent, pairs=PAIRS, sigma=SIGMA,
                 learning_rate=LEARNING_RATE, max_frames=MAX_FRAMES, seed=0):
        """Set up a training run.

        Args:
            agent: The agent to train.
            pairs: Mirrored perturbation pairs per generation.
            sigma: Scale of the parameter noise.
            learning_rate: Step size along the estimated gradient.
            max_frames: Frame budget per episode.
            seed: Seed for the perturbations and the piece sequences.

        Raises:
            ValueError: If pairs is not positive.
        """
        if pairs <= 0:
            raise ValueError(f'pairs must be positive, got {pairs}.')

        self.agent = agent
        self.pairs = pairs
        self.sigma = sigma
        self.learning_rate = learning_rate
        self.max_frames = max_frames
        self.seed = seed
        self._rng = torch.Generator().manual_seed(seed)

    def evaluate(self, vector):
        """Play one episode with one candidate parameter vector.

        Every candidate shares the trainer's seed, so differences in fitness
        reflect the parameters rather than the luck of the piece sequence.

        Args:
            vector: Candidate encoder and decoder parameters.

        Returns:
            (fitness, lines_cleared, frames_played).
        """
        self.agent.set_parameters(vector)
        lines, frames = self.agent.play_episode(TetrisGame(seed=self.seed),
                                                max_frames=self.max_frames)
        return self.LINE_WEIGHT * lines + frames, lines, frames

    def train(self, generations, on_generation=None):
        """Run evolutionary strategies for a number of generations.

        Args:
            generations: How many generations to run.
            on_generation: Optional callable ``on_generation(index, fitness,
                lines, frames)`` called after each generation with the results
                of the updated parameters.

        Returns:
            A list of (fitness, lines, frames) tuples, one per generation.
        """
        vector = self.agent.get_parameters()
        history = []

        for generation in range(generations):
            # Every episode of the run uses the same piece sequence. That lets
            # the agent overfit to it, but it is the only way the reported
            # fitness is comparable from one generation to the next - with a
            # fresh sequence each time, the luck of the draw swamps any real
            # progress. Vary `seed` across runs to check it generalises.
            directions = torch.randn(self.pairs, vector.numel(), generator=self._rng)

            scores = torch.empty(self.pairs, 2)
            for i, direction in enumerate(directions):
                offset = self.sigma * direction
                scores[i, 0] = self.evaluate(vector + offset)[0]
                scores[i, 1] = self.evaluate(vector - offset)[0]

            vector = vector + self._update_step(directions, scores)

            result = self.evaluate(vector)
            history.append(result)
            if on_generation:
                on_generation(generation, *result)

        self.agent.set_parameters(vector)
        return history

    def _update_step(self, directions, scores):
        """Combine one generation's results into a parameter update.

        Args:
            directions: (pairs, parameters) unit-scale random directions.
            scores: (pairs, 2) fitnesses, plus-direction in column 0.

        Returns:
            The step to add to the parameter vector, or zeros if every
            candidate scored the same and there is nothing to climb.
        """
        spread = scores.std()
        if spread < 1e-8:
            return torch.zeros(directions.shape[1])

        # Standardising makes the step size independent of the fitness scale.
        standardised = (scores - scores.mean()) / spread
        advantage = standardised[:, 0] - standardised[:, 1]

        gradient = (advantage.unsqueeze(1) * directions).mean(dim=0)
        return (self.learning_rate / (2 * self.sigma)) * gradient

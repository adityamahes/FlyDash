"""The agent: Tetris wired into the fly brain, and the trainer that tunes it.

Five classes:

``Encoder``   board -> current injected at the visual neurons (trainable)
``Decoder``   motor neuron firing rates -> five action logits (trainable)
``Reward``    board outcome -> one scalar, for both the brain and the trainer
``Agent``     runs one frame: see, simulate, act, be rewarded
``Trainer``   evolutionary strategies over the encoder and decoder

The encoder and decoder are the only trainable parts. The connectome between
them is frozen anatomy, and the reward rule is fixed.
"""

import math

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
    """Learns to read descending-neuron activity as one of five controls.

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
            A (len(Action),) tensor of logits, ordered to match Action.
        """
        return self.layer(rates[self.motor])


class Reward:
    """Scores what one frame did, on a scale of consequence.

    This is the objective the trainer climbs *and* the teaching signal the
    dopamine neurons receive: one number, so the brain is told the same thing
    selection acts on.

    Two properties it is built for:

    **Something is scored on every frame.** An earlier version only paid out
    when a piece came to rest, which with a falling piece is one frame in
    nineteen - the brain received a flat zero the rest of the time and had
    nothing to associate with anything. Now a frame that walks into a wall is
    punished and a frame that does something legal is credited, however
    slightly, so the signal is continuous.

    **Consequences are graded across four orders of magnitude.** A wasted
    frame costs 2, burying a hole costs 5, coming to rest costs or earns tens,
    clearing four rows at once earns 8000. Those ratios are the design: they
    are what makes a clear worth the dozen careful placements it takes to set
    up, rather than something to be traded away for a tidier board.

    The board terms are charged as the *change* they undergo, not their
    absolute size. Charging absolute badness per placement made a long good
    episode score worse than a short bad one purely for containing more
    placements - a lookahead policy that cleared a line over 34 pieces scored
    -3700 against always-drop's -870. As a difference they telescope over the
    episode to the badness of the final board, so tidiness is measured once
    however long the agent survives. This is potential-based shaping (Ng,
    Harada & Russell 1999), which is why it can be added without changing
    which policy is best.
    """

    # --- every frame ---

    #: An action that actually moved or rotated the piece. Small and positive,
    #: so there is something to feel on an ordinary frame.
    LEGAL = 0.2

    #: An action the rules rejected - into a wall, or a rotation that does not
    #: fit. Ten times the size of LEGAL: a wasted frame is a real cost, and
    #: this is the most frequent mistake available to an untrained policy.
    ILLEGAL = -2.0

    # --- when a piece comes to rest ---

    #: Paid per piece placed. The progress signal: more pieces fit before
    #: topping out only if they are stacked well. Per placement rather than
    #: per frame, because with a drop action a per-frame bonus would reward an
    #: agent that drifts sideways forever and never commits.
    PLACEMENT = 10.0

    #: Per row above the floor the piece came to rest. Prefers filling the
    #: bottom first, and makes a drop into a deep gap better than one onto a
    #: peak.
    LANDING = -1.0

    # --- when rows clear ---

    #: Per cleared row, multiplied by the number cleared: one row earns 500,
    #: four at once earn 8000, as in real Tetris scoring.
    LINE = 500.0

    #: Per cell of the just-placed piece that a clear consumed. Credits the
    #: piece that completed the row, not merely the board that resulted.
    ERODED = 20.0

    # --- once ---

    #: Topping out. Larger than a whole episode of LEGAL frames, so stalling
    #: is never a way to avoid it.
    GAME_OVER = -200.0

    # --- board badness, charged as the change it undergoes ---

    #: Empty cells with something filled above them. The expensive mistake:
    #: only a clear above can free them, so this is the heaviest board term.
    HOLES = 5.0

    #: One-wide gaps, which only an I piece can fill.
    WELLS = 1.0

    #: Height difference between neighbouring columns. A flat surface accepts
    #: whatever piece comes next; a jagged one does not.
    BUMPINESS = 0.5

    #: Total stack height. Weak on its own - height is only bad because it
    #: runs out of room - but it discourages building upward early.
    AGGREGATE = 0.2

    def __init__(self):
        """Start from the badness of an empty board."""
        self.reset()

    def reset(self):
        """Forget the previous board, as at the start of an episode."""
        self.previous = 0.0

    def badness(self, game):
        """Score how awkward a board is to keep playing on.

        Args:
            game: The game whose board to judge.

        Returns:
            A non-negative number; 0 is an empty board.
        """
        features = game.features()
        return (self.HOLES * features['holes']
                + self.WELLS * features['wells']
                + self.BUMPINESS * features['bumpiness']
                + self.AGGREGATE * features['aggregate_height'])

    def __call__(self, game, cleared, over):
        """Score one frame.

        Args:
            game: The game, already stepped. Its placement attributes describe
                the piece that just landed, if one did.
            cleared: Rows cleared this frame.
            over: Whether the game ended this frame.

        Returns:
            The scalar reward for this frame. Negative is punishment.
        """
        score = self.LEGAL if game.applied else self.ILLEGAL

        # The stack only changes on the frame a piece locks, so that is the
        # only frame with a placement to judge.
        if game.locked:
            current = self.badness(game)
            score += (self.PLACEMENT
                      + self.LANDING * game.landing_height
                      + self.LINE * cleared ** 2
                      + self.ERODED * game.eroded_cells
                      - (current - self.previous))
            self.previous = current

        if over:
            score += self.GAME_OVER
        return score


class Agent:
    """Couples the trainable encoder and decoder to a frozen brain.

    One frame of play is the loop the design describes:

    1. Render the board.
    2. Encode it into current at the visual neurons.
    3. Simulate the connectome for many timesteps under that steady input.
    4. Decode the motor firing rates into one of five controls.
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
        reward: The reward rule's verdict on the most recent frame.
        dopamine: The current that reward became at the dopamine neurons.
    """

    #: Timesteps simulated per Tetris frame. The timestep is far shorter than
    #: the membrane time constant, so the network needs several to settle.
    FRAME_TIMESTEPS = 20

    #: Extra timesteps after the reward current is injected, so the dopamine
    #: signal actually propagates.
    REWARD_TIMESTEPS = 5

    #: Current injected at the dopamine neurons when the reward saturates.
    #: Not learned: real dopamine is an unconditioned "that was good / bad"
    #: teaching signal, so it stays external ground truth. Large enough to
    #: drive those neurons regardless of what the rest of the brain is doing.
    DOPAMINE_CURRENT = 2.0

    #: Reward magnitude that maps to tanh(1), i.e. about 76% of saturation.
    #: The reward itself ranges over hundreds, and a current of 500 against a
    #: firing threshold of 1.0 would say nothing except "enormous", so it is
    #: squashed. Real dopamine neurons saturate the same way.
    DOPAMINE_SCALE = 100.0

    def __init__(self, connectome):
        """Build an agent around one connectome.

        Args:
            connectome: The frozen brain to run on.
        """
        self.connectome = connectome
        self.network = SpikingNetwork(connectome)
        self.encoder = Encoder(connectome)
        self.decoder = Decoder(connectome)
        self.reward_rule = Reward()
        self.rates = torch.zeros(len(connectome))
        self.logits = torch.zeros(len(Action))
        self.reward = 0.0
        self.dopamine = 0.0

    def reset(self):
        """Clear the simulation and the reward history, as between episodes."""
        self.network.reset()
        self.reward_rule.reset()

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
        cleared, over = game.step(action)

        self.reward = self.reward_rule(game, cleared, over)
        self.dopamine = self._inject_reward(self.reward)
        return action

    def _inject_reward(self, reward):
        """Deliver the teaching signal to the dopamine neurons.

        The scalar is squashed onto a bounded current and applied to every
        dopamine neuron at once, then simulated for a few more timesteps so it
        actually reaches the rest of the brain before the next frame is
        perceived. Because the network is not reset between frames, that
        activity is still present when the next board arrives - which is the
        only route by which reward reaches behaviour at all.

        Args:
            reward: This frame's scalar reward. Negative is punishment.

        Returns:
            The current that was injected, for inspection.
        """
        amount = self.DOPAMINE_CURRENT * math.tanh(reward / self.DOPAMINE_SCALE)

        current = torch.zeros(len(self.connectome))
        current[self.connectome.reward] = amount
        self.network.run(current, self.REWARD_TIMESTEPS)
        return amount

    def play_episode(self, game, max_frames=500):
        """Play until the game ends or the frame budget runs out.

        This runs flat out, for training. The browser view drives
        :meth:`play_frame` itself so it can pace the game and publish each frame.

        Args:
            game: A fresh game to play. Mutated in place.
            max_frames: Frame budget, so a policy that never tops out still
                terminates.

        Returns:
            (total_reward, lines_cleared, frames_played) for the episode.
        """
        self.reset()

        total = 0.0
        frames = 0
        for _ in range(max_frames):
            self.play_frame(game)
            total += self.reward
            frames += 1
            if game.over:
                break

        return total, game.lines, frames

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
            (fitness, lines_cleared, frames_played). Fitness is the episode's
            total reward, so :class:`Reward` alone decides what "better" means
            - the trainer holds no scoring weights of its own.
        """
        self.agent.set_parameters(vector)
        return self.agent.play_episode(TetrisGame(seed=self.seed),
                                       max_frames=self.max_frames)

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

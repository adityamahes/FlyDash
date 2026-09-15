"""Tests for TetriFLY. Run with: python -m unittest test_tetrifly -v"""

import tempfile
import unittest
from pathlib import Path

import pandas as pd
import torch

from tetrifly import Action, Agent, Connectome, Piece, SpikingNetwork, TetrisGame, Trainer
from tetrifly.tetris import _rotate


def make_neurons():
    """A four-neuron table covering each transmitter case."""
    return pd.DataFrame({
        'bodyId': [100, 200, 300, 400],
        'type': ['R1-R6', 'Mi1', 'DNp01', 'PPL1-a'],
        'predictedNt': ['acetylcholine', 'gaba', 'dopamine', 'glutamate'],
        'predictedNtProb': [0.99, 0.95, 0.90, 0.20],
    })


def chain_network(strength=2.0):
    """A three-neuron chain 0 -> 1 -> 2, bypassing the build pipeline."""
    connectome = Connectome.__new__(Connectome)
    connectome.matrix = torch.sparse_coo_tensor(
        torch.tensor([[0, 1], [1, 2]]), torch.tensor([strength, strength]), (3, 3),
    ).coalesce()
    return SpikingNetwork(connectome)


def fill_row(game, row, empty=()):
    """Fill a board row, optionally leaving some columns empty."""
    for col in range(game.WIDTH):
        game.board[row][col] = 0 if col in empty else 1


# ---------------------------------------------------------------------------
# Connectome: neuPrint tables -> frozen wiring matrix
# ---------------------------------------------------------------------------

class TestConnectomeBuild(unittest.TestCase):

    def test_roi_duplicates_are_summed_into_one_row(self):
        connections = pd.DataFrame({
            'bodyId_pre': [100, 100, 100, 200],
            'bodyId_post': [200, 200, 300, 300],
            'roi': ['ME', 'LO', 'ME', 'MB'],
            'weight': [5, 7, 3, 11],
        })

        pairs = Connectome._collapse_roi_duplicates(connections)

        self.assertEqual(len(pairs), 3)
        weights = {(r.bodyId_pre, r.bodyId_post): r.weight for r in pairs.itertuples()}
        self.assertEqual(weights[(100, 200)], 12)   # 5 + 7, summed across regions
        self.assertEqual(weights[(100, 300)], 3)

    def test_acetylcholine_excites_and_gaba_inhibits(self):
        pairs = pd.DataFrame({'bodyId_pre': [100, 200], 'bodyId_post': [200, 100],
                              'weight': [10, 20]})

        strengths = Connectome._signed_strengths(make_neurons(), pairs)

        self.assertAlmostEqual(strengths.iloc[0], +10 * Connectome.WEIGHT_PER_SYNAPSE)
        self.assertAlmostEqual(strengths.iloc[1], -20 * Connectome.WEIGHT_PER_SYNAPSE)

    def test_unmapped_transmitter_falls_back_to_the_default_sign(self):
        # Neuron 300 is dopamine, a neuromodulator with no simple sign.
        pairs = pd.DataFrame({'bodyId_pre': [300], 'bodyId_post': [100], 'weight': [10]})

        strengths = Connectome._signed_strengths(make_neurons(), pairs)

        self.assertAlmostEqual(strengths.iloc[0],
                               Connectome.DEFAULT_SIGN * 10 * Connectome.WEIGHT_PER_SYNAPSE)

    def test_low_confidence_prediction_falls_back_to_the_default_sign(self):
        # Neuron 400 is glutamate (inhibitory) but only 0.20 confident.
        pairs = pd.DataFrame({'bodyId_pre': [400], 'bodyId_post': [100], 'weight': [10]})

        strengths = Connectome._signed_strengths(make_neurons(), pairs)

        self.assertGreater(strengths.iloc[0], 0)

    def test_confident_prediction_is_trusted(self):
        neurons = make_neurons()
        neurons.loc[neurons.bodyId == 400, 'predictedNtProb'] = 0.99
        pairs = pd.DataFrame({'bodyId_pre': [400], 'bodyId_post': [100], 'weight': [10]})

        strengths = Connectome._signed_strengths(neurons, pairs)

        self.assertLess(strengths.iloc[0], 0)

    def test_entry_sits_at_sender_row_and_receiver_column(self):
        neurons = make_neurons()
        pairs = pd.DataFrame({'bodyId_pre': [100, 200], 'bodyId_post': [300, 400],
                              'weight': [10, 10]})
        strengths = Connectome._signed_strengths(neurons, pairs)

        matrix = Connectome._build_matrix(neurons, pairs, strengths).to_dense()

        self.assertAlmostEqual(matrix[0, 2].item(), +1.0)   # 100 -> 300, excitatory
        self.assertAlmostEqual(matrix[1, 3].item(), -1.0)   # 200 -> 400, inhibitory
        self.assertAlmostEqual(matrix[2, 0].item(), 0.0)    # no reverse connection

    def test_connections_to_unknown_neurons_are_dropped(self):
        neurons = make_neurons()
        pairs = pd.DataFrame({'bodyId_pre': [100, 999], 'bodyId_post': [200, 200],
                              'weight': [10, 10]})
        strengths = Connectome._signed_strengths(neurons, pairs)

        matrix = Connectome._build_matrix(neurons, pairs, strengths)

        self.assertEqual(matrix._nnz(), 1)

    def test_misaligned_strengths_are_rejected(self):
        neurons = make_neurons()
        pairs = pd.DataFrame({'bodyId_pre': [100, 200], 'bodyId_post': [300, 400],
                              'weight': [10, 10]})

        with self.assertRaises(ValueError):
            Connectome._build_matrix(neurons, pairs, pd.Series([1.0]))

    def test_normalising_equalises_total_excitatory_input(self):
        # Neuron 2 receives 1.0 and 3.0; neuron 3 receives 0.5. Both should end
        # up with the same total drive.
        matrix = torch.sparse_coo_tensor(
            torch.tensor([[0, 1, 0], [2, 2, 3]]),
            torch.tensor([1.0, 3.0, 0.5]), (4, 4),
        ).coalesce()

        normalised = Connectome._normalise_input(matrix).to_dense()

        self.assertAlmostEqual(normalised[:, 2].sum().item(), Connectome.TARGET_INPUT)
        self.assertAlmostEqual(normalised[:, 3].sum().item(), Connectome.TARGET_INPUT)

    def test_normalising_keeps_relative_strengths_and_signs(self):
        matrix = torch.sparse_coo_tensor(
            torch.tensor([[0, 1], [2, 2]]), torch.tensor([1.0, -3.0]), (3, 3),
        ).coalesce()

        normalised = Connectome._normalise_input(matrix).to_dense()

        self.assertGreater(normalised[0, 2].item(), 0)
        self.assertLess(normalised[1, 2].item(), 0)
        self.assertAlmostEqual(normalised[1, 2].item() / normalised[0, 2].item(), -3.0)

    def test_neuron_with_no_excitatory_input_is_left_alone(self):
        matrix = torch.sparse_coo_tensor(
            torch.tensor([[0], [1]]), torch.tensor([-2.0]), (2, 2),
        ).coalesce()

        normalised = Connectome._normalise_input(matrix).to_dense()

        self.assertAlmostEqual(normalised[0, 1].item(), -2.0)


class TestConnectomePopulations(unittest.TestCase):

    def test_each_population_is_found_by_cell_type(self):
        connectome = Connectome.synthetic()

        for name, indices in (('visual', connectome.visual),
                              ('motor', connectome.motor),
                              ('reward', connectome.reward)):
            self.assertGreater(len(indices), 0, name)

    def test_populations_do_not_overlap(self):
        connectome = Connectome.synthetic()

        all_indices = torch.cat([connectome.visual, connectome.motor, connectome.reward])

        self.assertEqual(len(all_indices), len(set(all_indices.tolist())))

    def test_a_missing_population_is_an_error(self):
        connectome = Connectome.__new__(Connectome)
        connectome.types = pd.Series(['R1-R6', 'Mi1'])

        with self.assertRaises(ValueError):
            connectome._find_population(Connectome.MOTOR_TYPES)

    def test_pattern_is_anchored_so_unrelated_types_do_not_match(self):
        connectome = Connectome.__new__(Connectome)
        connectome.types = pd.Series(['LPLC2_R1-R6_downstream', 'R1-R6'])

        self.assertEqual(connectome._find_population(Connectome.VISUAL_TYPES).tolist(), [1])


class TestConnectomeEndToEnd(unittest.TestCase):

    def test_synthetic_brain_builds_a_square_coalesced_matrix(self):
        connectome = Connectome.synthetic()

        self.assertEqual(connectome.matrix.shape, (len(connectome), len(connectome)))
        self.assertTrue(connectome.matrix.is_coalesced())

    def test_one_matrix_entry_per_connected_pair(self):
        connectome = Connectome.synthetic()
        # Rebuild the same fixture to count its unique pairs.
        other = Connectome.synthetic()

        self.assertEqual(connectome.matrix._nnz(), other.matrix._nnz())
        self.assertLess(connectome.matrix._nnz(), len(connectome) ** 2)

    def test_unnormalised_build_keeps_raw_synapse_scale(self):
        neurons = make_neurons()
        connections = pd.DataFrame({
            'bodyId_pre': [100], 'bodyId_post': [200], 'roi': ['ME'], 'weight': [10],
        })

        connectome = Connectome(neurons, connections, normalise=False)

        self.assertAlmostEqual(connectome.matrix.to_dense()[0, 1].item(), 1.0)

    def test_saving_and_loading_round_trips(self):
        connectome = Connectome.synthetic()

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'brain.pt'
            connectome.save(path)
            reloaded = Connectome.load(path)

        self.assertTrue(torch.equal(connectome.matrix.to_dense(),
                                    reloaded.matrix.to_dense()))
        self.assertTrue(torch.equal(connectome.motor, reloaded.motor))


# ---------------------------------------------------------------------------
# SpikingNetwork: leaky integrate-and-fire dynamics
# ---------------------------------------------------------------------------

class TestSpikingNetwork(unittest.TestCase):

    def test_neuron_fires_once_it_crosses_threshold(self):
        network = chain_network()

        spikes = network.step(torch.tensor([SpikingNetwork.THRESHOLD + 0.5, 0.0, 0.0]))

        self.assertEqual(spikes.tolist(), [1.0, 0.0, 0.0])

    def test_neuron_stays_silent_below_threshold(self):
        network = chain_network()

        spikes = network.step(torch.tensor([SpikingNetwork.THRESHOLD - 0.5, 0.0, 0.0]))

        self.assertEqual(spikes.tolist(), [0.0, 0.0, 0.0])

    def test_firing_resets_the_potential(self):
        network = chain_network()

        network.step(torch.tensor([5.0, 0.0, 0.0]))

        self.assertAlmostEqual(network.potential[0].item(), 0.0)

    def test_potential_leaks_away_when_nothing_is_injected(self):
        network = chain_network()
        network.step(torch.tensor([0.5, 0.0, 0.0]))
        before = network.potential[0].item()

        network.step()

        self.assertAlmostEqual(network.potential[0].item(),
                               before * (1 - SpikingNetwork.DECAY))

    def test_subthreshold_input_accumulates_until_it_fires(self):
        network = chain_network()
        drip = torch.tensor([0.3, 0.0, 0.0])

        fired_on = [i for i in range(10) if network.step(drip)[0] == 1.0]

        self.assertTrue(fired_on, 'repeated subthreshold input should eventually fire')

    def test_a_spike_propagates_along_the_wiring_next_step(self):
        network = chain_network(strength=SpikingNetwork.THRESHOLD + 1.0)

        network.step(torch.tensor([5.0, 0.0, 0.0]))
        spikes = network.step()

        # Neuron 0 fired first, so neuron 1 now receives its synapse.
        self.assertEqual(spikes.tolist(), [0.0, 1.0, 0.0])

    def test_inhibition_suppresses_a_neuron_that_would_otherwise_fire(self):
        connectome = Connectome.__new__(Connectome)
        connectome.matrix = torch.sparse_coo_tensor(
            torch.tensor([[0], [1]]), torch.tensor([-5.0]), (2, 2)).coalesce()
        network = SpikingNetwork(connectome)

        network.step(torch.tensor([5.0, 0.0]))
        spikes = network.step(torch.tensor([0.0, 2.0]))

        self.assertEqual(spikes[1].item(), 0.0)

    def test_run_returns_firing_rates_between_zero_and_one(self):
        network = chain_network()

        rates = network.run(torch.tensor([5.0, 0.0, 0.0]), timesteps=10)

        self.assertEqual(rates[0].item(), 1.0)
        self.assertTrue(((rates >= 0) & (rates <= 1)).all())

    def test_reset_clears_everything(self):
        network = chain_network()
        network.step(torch.tensor([5.0, 0.0, 0.0]))

        network.reset()

        self.assertEqual(network.potential.abs().sum().item(), 0.0)
        self.assertEqual(network.spikes.sum().item(), 0.0)

    def test_wrong_shaped_input_is_rejected(self):
        with self.assertRaises(ValueError):
            chain_network().step(torch.zeros(5))

    def test_non_positive_timestep_count_is_rejected(self):
        with self.assertRaises(ValueError):
            chain_network().run(None, timesteps=0)


# ---------------------------------------------------------------------------
# Tetris
# ---------------------------------------------------------------------------

class TestPiece(unittest.TestCase):

    def test_every_piece_has_four_rotations_of_four_cells(self):
        for name, rotations in Piece.ROTATIONS.items():
            self.assertEqual(len(rotations), 4, name)
            for cells in rotations:
                self.assertEqual(len(cells), 4, name)

    def test_four_rotations_return_a_piece_to_its_start(self):
        for name, rotations in Piece.ROTATIONS.items():
            self.assertEqual(_rotate(rotations[3], Piece.BOX_SIZES[name]),
                             rotations[0], name)

    def test_o_piece_never_changes_shape(self):
        self.assertEqual(len(set(Piece.ROTATIONS['O'])), 1)

    def test_moving_and_rotating_do_not_mutate_the_original(self):
        piece = Piece('T', rotation=0, row=5, col=5)

        piece.moved(1, 1)
        piece.rotated(1)

        self.assertEqual((piece.row, piece.col, piece.rotation), (5, 5, 0))

    def test_cells_are_offset_by_the_piece_position(self):
        at_origin = Piece('T', 0, 0, 0).cells()
        moved = Piece('T', 0, 3, 2).cells()

        self.assertEqual(moved, [(r + 3, c + 2) for r, c in at_origin])


class TestTetrisRules(unittest.TestCase):

    def setUp(self):
        self.game = TetrisGame(seed=1)

    def test_moving_left_shifts_one_column(self):
        before = self.game.piece.col

        self.game.step(Action.LEFT)

        self.assertEqual(self.game.piece.col, before - 1)

    def test_gravity_drops_one_row_per_frame(self):
        before = self.game.piece.row

        self.game.step(Action.LEFT)

        self.assertEqual(self.game.piece.row, before + 1)

    def test_move_into_the_wall_is_ignored_but_time_passes(self):
        self.game.piece = Piece('O', 0, 0, 0)

        self.game.step(Action.LEFT)

        self.assertEqual(self.game.piece.col, 0)
        self.assertEqual(self.game.piece.row, 1)

    def test_rotation_that_does_not_fit_is_ignored(self):
        self.game.piece = Piece('I', rotation=1, row=0, col=5)
        self.game.board[2][5] = 1        # blocks the flat orientation

        self.game.step(Action.ROTATE_RIGHT)

        self.assertEqual(self.game.piece.rotation, 1)

    def test_piece_locks_when_it_reaches_the_floor(self):
        self.game.piece = Piece('O', 0, self.game.HEIGHT - 2, 0)

        self.game.step(Action.ROTATE_LEFT)

        self.assertEqual(self.game.board[self.game.HEIGHT - 1][0], 1)

    def test_a_completed_row_is_cleared_and_counted(self):
        bottom = self.game.HEIGHT - 1
        fill_row(self.game, bottom, empty=(0, 1))
        self.game.piece = Piece('O', 0, bottom - 1, 0)

        lines, over = self.game.step(Action.ROTATE_LEFT)

        self.assertEqual(lines, 1)
        self.assertFalse(over)
        self.assertEqual(self.game.lines, 1)
        self.assertEqual(sum(self.game.board[bottom]), 2)

    def test_rows_above_a_clear_fall_down(self):
        bottom = self.game.HEIGHT - 1
        fill_row(self.game, bottom, empty=(0, 1))
        self.game.board[bottom - 2][5] = 1
        self.game.piece = Piece('O', 0, bottom - 1, 0)

        self.game.step(Action.ROTATE_LEFT)

        self.assertEqual(self.game.board[bottom - 1][5], 1)

    def test_topping_out_ends_the_episode(self):
        for row in range(4):
            fill_row(self.game, row, empty=(self.game.WIDTH - 1,))
        self.game.piece = Piece('O', 0, self.game.HEIGHT - 2, 0)

        _, over = self.game.step(Action.ROTATE_LEFT)

        self.assertTrue(over)
        self.assertTrue(self.game.over)
        with self.assertRaises(RuntimeError):
            self.game.step(Action.LEFT)

    def test_same_seed_gives_the_same_piece_sequence(self):
        first = [TetrisGame(seed=3).piece.name for _ in range(3)]
        second = [TetrisGame(seed=3).piece.name for _ in range(3)]

        self.assertEqual(first, second)


class TestObservation(unittest.TestCase):

    def setUp(self):
        self.game = TetrisGame(seed=1)

    def test_observation_has_the_documented_length(self):
        self.assertEqual(self.game.observation().numel(), TetrisGame.OBSERVATION_SIZE)

    def test_observation_is_binary(self):
        observation = self.game.observation()

        self.assertTrue(((observation == 0) | (observation == 1)).all())

    def test_locked_blocks_and_falling_piece_are_kept_apart(self):
        fill_row(self.game, self.game.HEIGHT - 1)
        half = TetrisGame.OBSERVATION_SIZE // 2

        observation = self.game.observation()

        self.assertEqual(observation[:half].sum().item(), float(self.game.WIDTH))
        self.assertEqual(observation[half:].sum().item(), 4.0)

    def test_the_board_changes_what_the_agent_sees(self):
        before = self.game.observation()
        fill_row(self.game, self.game.HEIGHT - 1)

        self.assertFalse(torch.equal(before, self.game.observation()))


# ---------------------------------------------------------------------------
# Encoder, decoder, agent, trainer
# ---------------------------------------------------------------------------

class TestAgent(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.connectome = Connectome.synthetic()

    def test_encoder_injects_only_at_the_visual_neurons(self):
        agent = Agent(self.connectome)

        current = agent.encoder(TetrisGame(seed=0).observation())

        elsewhere = torch.ones(len(self.connectome), dtype=torch.bool)
        elsewhere[self.connectome.visual] = False
        self.assertEqual(current.shape, (len(self.connectome),))
        self.assertEqual(current[elsewhere].abs().sum().item(), 0.0)

    def test_different_boards_produce_different_sensory_input(self):
        agent = Agent(self.connectome)
        empty = torch.zeros(TetrisGame.OBSERVATION_SIZE)
        full = torch.ones(TetrisGame.OBSERVATION_SIZE)

        self.assertFalse(torch.equal(agent.encoder(empty), agent.encoder(full)))

    def test_decoder_emits_one_logit_per_control(self):
        agent = Agent(self.connectome)

        logits = agent.decoder(torch.rand(len(self.connectome)))

        self.assertEqual(logits.shape, (len(Action),))

    def test_decoder_reads_only_the_motor_population(self):
        agent = Agent(self.connectome)
        rates = torch.zeros(len(self.connectome))
        before = agent.decoder(rates)

        rates[self.connectome.visual] = 1.0

        self.assertTrue(torch.equal(before, agent.decoder(rates)))

    def test_one_frame_picks_an_action_and_advances_the_game(self):
        agent = Agent(self.connectome)
        game = TetrisGame(seed=0)
        before = game.piece.row

        action = agent.play_frame(game)

        self.assertIsInstance(action, Action)
        self.assertEqual(game.piece.row, before + 1)

    def test_an_episode_stays_within_its_frame_budget(self):
        agent = Agent(self.connectome)

        lines, frames = agent.play_episode(TetrisGame(seed=0), max_frames=30)

        self.assertLessEqual(frames, 30)
        self.assertGreaterEqual(lines, 0)

    def test_playing_never_modifies_the_frozen_wiring(self):
        agent = Agent(self.connectome)
        before = self.connectome.matrix.values().clone()

        agent.play_episode(TetrisGame(seed=0), max_frames=20)

        self.assertTrue(torch.equal(before, self.connectome.matrix.values()))

    def test_same_parameters_and_seed_replay_identically(self):
        agent = Agent(self.connectome)
        parameters = agent.get_parameters()

        first = agent.play_episode(TetrisGame(seed=7), max_frames=25)
        agent.set_parameters(parameters)
        second = agent.play_episode(TetrisGame(seed=7), max_frames=25)

        self.assertEqual(first, second)

    def test_parameter_vector_round_trips(self):
        agent = Agent(self.connectome)
        original = agent.get_parameters()

        agent.set_parameters(torch.zeros_like(original))

        self.assertEqual(agent.get_parameters().abs().sum().item(), 0.0)

    def test_parameter_vector_covers_encoder_and_decoder(self):
        agent = Agent(self.connectome)
        expected = (sum(p.numel() for p in agent.encoder.parameters())
                    + sum(p.numel() for p in agent.decoder.parameters()))

        self.assertEqual(agent.get_parameters().numel(), expected)

    def test_wrong_length_parameter_vector_is_rejected(self):
        with self.assertRaises(ValueError):
            Agent(self.connectome).set_parameters(torch.zeros(3))

    def test_different_parameters_produce_different_behaviour(self):
        agent = Agent(self.connectome)
        base = agent.get_parameters()
        generator = torch.Generator().manual_seed(0)

        behaviours = set()
        for _ in range(6):
            agent.set_parameters(base + torch.randn(base.numel(), generator=generator))
            game = TetrisGame(seed=0)
            agent.reset()
            behaviours.add(tuple(int(agent.play_frame(game)) for _ in range(15)))

        # If the encoder could not influence the motor neurons, every candidate
        # would behave identically and training would have nothing to optimise.
        self.assertGreater(len(behaviours), 1)


class TestTrainer(unittest.TestCase):

    def test_training_reports_one_result_per_generation(self):
        trainer = Trainer(Agent(Connectome.synthetic()), pairs=2, max_frames=20)

        history = trainer.train(generations=2)

        self.assertEqual(len(history), 2)
        for fitness, lines, frames in history:
            self.assertGreaterEqual(frames, 1)
            self.assertGreaterEqual(lines, 0)

    def test_cleared_lines_count_for_more_than_survival(self):
        trainer = Trainer(Agent(Connectome.synthetic()), pairs=1, max_frames=10)

        cleared_one_line = Trainer.LINE_WEIGHT * 1 + 10
        survived_a_full_episode = Trainer.LINE_WEIGHT * 0 + trainer.max_frames

        self.assertGreater(cleared_one_line, survived_a_full_episode)

    def test_a_flat_generation_produces_no_update(self):
        trainer = Trainer(Agent(Connectome.synthetic()), pairs=3)
        identical = torch.full((3, 2), 5.0)

        step = trainer._update_step(torch.randn(3, 20), identical)

        self.assertEqual(step.abs().sum().item(), 0.0)

    def test_the_update_moves_toward_the_better_side(self):
        trainer = Trainer(Agent(Connectome.synthetic()), pairs=1)
        direction = torch.tensor([[1.0, 0.0]])

        better_plus = trainer._update_step(direction, torch.tensor([[10.0, 0.0]]))
        better_minus = trainer._update_step(direction, torch.tensor([[0.0, 10.0]]))

        self.assertGreater(better_plus[0].item(), 0)
        self.assertLess(better_minus[0].item(), 0)

    def test_non_positive_pair_count_is_rejected(self):
        with self.assertRaises(ValueError):
            Trainer(Agent(Connectome.synthetic()), pairs=0)


if __name__ == '__main__':
    unittest.main()

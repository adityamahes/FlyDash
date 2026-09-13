import torch
from neuprint import Client, fetch_neurons, fetch_adjacencies

NEUPRINT_AUTH_TOKEN = 'YOUR_TOKEN_HERE'

MEMBRANE_DECAY_RATE = 0.05      # fraction of membrane potential lost per timestep   (= dt/tau) or tau = 5ms (published) dt = 0.25ms
FIRING_THRESHOLD = 1.0          # potential required for a neuron to spike    (if SYNAPTIC_WEIGHT_PER_SYNAPSE is 0.1, needs 10 synapses to activate)

VISUAL_NEURON_TYPE_PATTERN = 'R1-R6|Mi1'      # visual pathway
MOTOR_NEURON_TYPE_PATTERN = 'DNp01|DNg'       # driving movement
REWARD_NEURON_TYPE_PATTERN = 'PPL1'           # dopaminergic reward/punishment

REWARD_CURRENT_MAGNITUDE = 1.0
PUNISHMENT_CURRENT_MAGNITUDE = -1.0

TRANSMITTER_SIGN_MAP = {
    'acetylcholine': +1,
    'gaba': -1,
    'glutamate': -1,
}

def load_connectome():
    neuprint_client = Client('neuprint.janelia.org', dataset='male-cns:v1.0', token=NEUPRINT_AUTH_TOKEN)

    neuron_table, _ = fetch_neurons(client=neuprint_client)  #each row is neuron with info about it
    connection_table = fetch_adjacencies(   # bodyId_pre   bodyId_post   roi (region)     weight (# synapses)
        sources=neuron_table.bodyId,
        targets=neuron_table.bodyId,
        client=neuprint_client,
    )

    return neuron_table, connection_table


# Assign a synaptic sign (excitatory/inhibitory) to each connection. Determine whether each connection excites or inhibits its target,
#     based on the predicted neurotransmitter of the presynaptic neuron

def compute_signed_connection_strengths(neuron_table, connection_table):

    connection_table = connection_table.groupby(['bodyId_pre', 'bodyId_post'])['weight'].sum().reset_index()
    neuron_id_to_transmitter = dict(zip(neuron_table.bodyId, neuron_table.predictedNt))

    presynaptic_transmitters = connection_table.bodyId_pre.map(neuron_id_to_transmitter)
    connection_signs = presynaptic_transmitters.map(TRANSMITTER_SIGN_MAP).fillna(1)    #fills not define nts with +1

    signed_connection_strengths = connection_table.weight * connection_signs
    return signed_connection_strengths


# Make matrix where [i, j] is the signed connection strengths of synapse of neuron i on neuron j. (fixed constant biologically derived matrix)

def build_connectivity_matrix(neuron_table, connection_table, signed_connection_strengths):
    """Build an (N x N) sparse matrix where entry """

    total_neuron_count = len(neuron_table)
    neuron_id_to_index = {
        neuron_id: index for index, neuron_id in enumerate(neuron_table.bodyId)
    }

    presynaptic_indices = connection_table.bodyId_pre.map(neuron_id_to_index)
    postsynaptic_indices = connection_table.bodyId_post.map(neuron_id_to_index)

    connectivity_matrix = torch.sparse_coo_tensor(
        torch.tensor([presynaptic_indices.values, postsynaptic_indices.values]),
        torch.tensor(signed_connection_strengths.values, dtype=torch.float32),
        size=(total_neuron_count, total_neuron_count),
    )

    return connectivity_matrix, neuron_id_to_index


# A leaky integrate-and-fire simulation running over the fixed connectivity matrix

class ConnectomeSpikingNetwork:
    def __init__(self, connectivity_matrix, total_neuron_count):
        self.connectivity_matrix = connectivity_matrix
        self.membrane_potential = torch.zeros(total_neuron_count)
        self.neuron_spiked = torch.zeros(total_neuron_count)

    def advance_one_timestep(self, external_input_current):
        # Update every neuron's membrane potential by one simulation step, given the influence of neurons that spiked last step plus any externally injected current (sensory or reward)

        recurrent_input_current = torch.sparse.mm(self.connectivity_matrix, self.neuron_spiked.unsqueeze(1)).squeeze()

        self.membrane_potential = (
            self.membrane_potential * (1 - MEMBRANE_DECAY_RATE)
            + recurrent_input_current
            + external_input_current
        )

        self.neuron_spiked = (self.membrane_potential > FIRING_THRESHOLD).float()
        self.membrane_potential = self.membrane_potential * (1 - self.neuron_spiked)

        return self.neuron_spiked


def locate_functional_neuron_groups(neuron_table, neuron_id_to_index):

    def indices_matching(type_pattern):
        matching_neurons = neuron_table[
            neuron_table.type.str.contains(type_pattern, na=False)
        ]
        return matching_neurons.bodyId.map(neuron_id_to_index).values

    visual_neuron_indices = indices_matching(VISUAL_NEURON_TYPE_PATTERN)
    motor_neuron_indices = indices_matching(MOTOR_NEURON_TYPE_PATTERN)
    reward_neuron_indices = indices_matching(REWARD_NEURON_TYPE_PATTERN)

    return visual_neuron_indices, motor_neuron_indices, reward_neuron_indices


def convert_game_frame_to_sensory_input(game_frame, visual_neuron_indices, total_neuron_count):
    """
    Placeholder: convert a raw Geometry Dash frame into an input-current
    vector aligned with the visual neuron population.

    To be implemented later. Should return a tensor of shape
    (total_neuron_count,) that is zero everywhere except at
    visual_neuron_indices.
    """
    sensory_input_current = torch.zeros(total_neuron_count)
    # TODO: extract obstacle distance / player position / jump timing
    # features from game_frame and assign them into
    # sensory_input_current[visual_neuron_indices]
    return sensory_input_current


def convert_motor_spikes_to_action(neuron_spiked, motor_neuron_indices):
    """Translate motor neuron activity into a jump / no-jump decision."""
    motor_neuron_activity = neuron_spiked[motor_neuron_indices]
    jump_signal_strength = motor_neuron_activity.mean().item()
    should_jump = jump_signal_strength > 0.5
    return should_jump


def build_reward_input_current(outcome, reward_neuron_indices, total_neuron_count):
    """Construct the current injected into reward neurons this timestep,
    based on the outcome of the most recent action."""

    reward_input_current = torch.zeros(total_neuron_count)

    if outcome == 'survived_obstacle':
        reward_input_current[reward_neuron_indices] = REWARD_CURRENT_MAGNITUDE
    elif outcome == 'collided_with_obstacle':
        reward_input_current[reward_neuron_indices] = PUNISHMENT_CURRENT_MAGNITUDE

    return reward_input_current


# ---------------------------------------------------------------------------
# Main game loop
# ---------------------------------------------------------------------------

def run_geometry_dash_episode(spiking_network, visual_neuron_indices,
                               motor_neuron_indices, reward_neuron_indices,
                               total_neuron_count, geometry_dash_environment):

    game_is_running = True

    while game_is_running:
        current_game_frame = geometry_dash_environment.get_current_frame()

        sensory_input_current = convert_game_frame_to_sensory_input(
            current_game_frame, visual_neuron_indices, total_neuron_count
        )
        neuron_spiked = spiking_network.advance_one_timestep(sensory_input_current)

        should_jump = convert_motor_spikes_to_action(neuron_spiked, motor_neuron_indices)
        step_result = geometry_dash_environment.apply_action(jump=should_jump)

        reward_input_current = build_reward_input_current(
            step_result.outcome, reward_neuron_indices, total_neuron_count
        )
        spiking_network.advance_one_timestep(reward_input_current)

        game_is_running = not step_result.episode_ended


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def main():
    neuron_table, connection_table = load_connectome()
    signed_connection_strengths = compute_signed_connection_strengths(
        neuron_table, connection_table
    )
    connectivity_matrix, neuron_id_to_index = build_connectivity_matrix(
        neuron_table, connection_table, signed_connection_strengths
    )

    total_neuron_count = len(neuron_table)
    spiking_network = ConnectomeSpikingNetwork(connectivity_matrix, total_neuron_count)

    visual_neuron_indices, motor_neuron_indices, reward_neuron_indices = (
        locate_functional_neuron_groups(neuron_table, neuron_id_to_index)
    )

    # geometry_dash_environment = YourGeometryDashInterface()
    # run_geometry_dash_episode(
    #     spiking_network, visual_neuron_indices, motor_neuron_indices,
    #     reward_neuron_indices, total_neuron_count, geometry_dash_environment
    # )


if __name__ == '__main__':
    main()
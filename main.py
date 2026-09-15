"""Command line entry point for TetriFLY.

Three subcommands::

    python main.py build --output data/connectome.pt   # fetch the real brain
    python main.py play --render                       # watch one episode
    python main.py train --generations 20              # optimise the agent

``play`` and ``train`` use a small synthetic brain unless ``--connectome``
points at a file built by ``build``, so the project runs with no neuPrint token
and no download.
"""

import argparse
import sys

from tetrifly import Agent, Connectome, TetrisGame, Trainer


def load_connectome(path):
    """Load the brain a run should use.

    Args:
        path: A file written by ``build``, or None for the synthetic brain.

    Returns:
        The Connectome to simulate.
    """
    if path is None:
        print('No --connectome given; using a small synthetic brain.')
        return Connectome.synthetic()

    print(f'Loading {path} ...')
    return Connectome.load(path)


def build(args):
    """Fetch the real connectome from neuPrint and cache it."""
    print('Fetching from neuPrint (this takes a while) ...')
    connectome = Connectome.from_neuprint(token=args.token)
    print(connectome)

    connectome.save(args.output)
    print(f'Saved to {args.output}')


def play(args):
    """Play one episode and report how it went."""
    connectome = load_connectome(args.connectome)
    print(connectome)

    agent = Agent(connectome)
    game = TetrisGame(seed=args.seed)

    def show(current, action):
        if args.render:
            print(current.render())
            print(f'action: {action.name}\n')

    lines, frames = agent.play_episode(game, max_frames=args.frames, on_frame=show)

    print(game.render())
    print(f'Finished after {frames} frames: {lines} lines, topped out {game.over}')


def train(args):
    """Optimise the encoder and decoder with evolutionary strategies."""
    connectome = load_connectome(args.connectome)
    print(connectome)

    agent = Agent(connectome)
    trainer = Trainer(agent, pairs=args.pairs, max_frames=args.frames, seed=args.seed)
    print(f'Optimising {agent.get_parameters().numel():,} parameters '
          f'over {args.generations} generations.')

    def report(generation, fitness, lines, frames):
        print(f'generation {generation:3d} | fitness {fitness:8.1f} | '
              f'lines {lines:3d} | frames {frames:4d}')

    trainer.train(args.generations, on_generation=report)


def main(argv=None):
    """Parse arguments and run the chosen subcommand.

    Args:
        argv: Argument list, defaulting to sys.argv[1:].

    Returns:
        A process exit code.
    """
    parser = argparse.ArgumentParser(description='Run a fly connectome as a Tetris agent.')
    commands = parser.add_subparsers(dest='command', required=True)

    build_parser = commands.add_parser('build', help='Fetch the connectome and cache it.')
    build_parser.add_argument('--token', help='neuPrint token (default: $NEUPRINT_TOKEN).')
    build_parser.add_argument('--output', default='data/connectome.pt')
    build_parser.set_defaults(run=build)

    play_parser = commands.add_parser('play', help='Play one episode.')
    play_parser.add_argument('--render', action='store_true', help='Print every frame.')
    play_parser.set_defaults(run=play)

    train_parser = commands.add_parser('train', help='Train the encoder and decoder.')
    train_parser.add_argument('--generations', type=int, default=10)
    train_parser.add_argument('--pairs', type=int, default=Trainer.PAIRS,
                              help='Perturbation pairs per generation.')
    train_parser.set_defaults(run=train)

    for subparser in (play_parser, train_parser):
        subparser.add_argument('--connectome', help='Cached brain (default: synthetic).')
        subparser.add_argument('--frames', type=int, default=Trainer.MAX_FRAMES)
        subparser.add_argument('--seed', type=int, default=0)

    args = parser.parse_args(argv)
    args.run(args)
    return 0


if __name__ == '__main__':
    sys.exit(main())

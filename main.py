"""Command line entry point for TetriFLY.

Three subcommands::

    python main.py build                   # fetch and cache the real brain
    python main.py serve                   # watch it in a browser
    python main.py train --generations 20  # optimise the agent

``build`` must be run once before the others: it downloads the connectome from
neuPrint and caches it at ``CONNECTOME_PATH``, which ``serve`` and ``train``
then load. All three work on the real fly brain; there is no synthetic mode.
"""

import argparse
import sys
from pathlib import Path

from tetrifly import Agent, Connectome, Trainer, web

# Where build writes the cached brain, and where serve and train look for it.
CONNECTOME_PATH = 'data/connectome.pt'


def load_connectome(path):
    """Load the cached brain a run should use.

    Args:
        path: A file previously written by ``build``.

    Returns:
        The Connectome to simulate.

    Raises:
        SystemExit: If no cached connectome exists at ``path``.
    """
    if not Path(path).exists():
        raise SystemExit(f'No connectome at {path}. Fetch it first with: '
                         f'python main.py build --output {path}')

    print(f'Loading {path} ...')
    return Connectome.load(path)


def build(args):
    """Fetch the real connectome from neuPrint and cache it."""
    print('Fetching from neuPrint (this takes a while) ...')
    connectome = Connectome.from_neuprint(token=args.token)
    print(connectome)

    connectome.save(args.output)
    print(f'Saved to {args.output}')


def serve(args):
    """Play continuously and stream the board to a browser."""
    connectome = load_connectome(args.connectome)
    print(connectome)

    web.serve(Agent(connectome), port=args.port, delay=args.delay,
              max_frames=args.frames, seed=args.seed)


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
    build_parser.add_argument('--token', help='neuPrint token (default: '
                              '$NEUPRINT_TOKEN, then local_config.TOKEN).')
    build_parser.add_argument('--output', default=CONNECTOME_PATH)
    build_parser.set_defaults(run=build)

    serve_parser = commands.add_parser('serve', help='Watch it play in a browser.')
    serve_parser.add_argument('--port', type=int, default=web.PORT)
    serve_parser.add_argument('--delay', type=float, default=web.GameRunner.DELAY,
                              help='Seconds to pause after each frame.')
    serve_parser.set_defaults(run=serve)

    train_parser = commands.add_parser('train', help='Train the encoder and decoder.')
    train_parser.add_argument('--generations', type=int, default=10)
    train_parser.add_argument('--pairs', type=int, default=Trainer.PAIRS,
                              help='Perturbation pairs per generation.')
    train_parser.set_defaults(run=train)

    for subparser in (serve_parser, train_parser):
        subparser.add_argument('--connectome', default=CONNECTOME_PATH,
                               help='Cached brain from build.')
        subparser.add_argument('--frames', type=int, default=Trainer.MAX_FRAMES,
                               help='Frame budget per episode.')
        subparser.add_argument('--seed', type=int, default=0)

    args = parser.parse_args(argv)
    args.run(args)
    return 0


if __name__ == '__main__':
    sys.exit(main())

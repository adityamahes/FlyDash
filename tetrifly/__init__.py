"""TetriFLY: a real fly connectome playing Tetris.

Three modules:

* :mod:`tetrifly.brain` - the frozen connectome and the spiking simulation.
* :mod:`tetrifly.tetris` - the game.
* :mod:`tetrifly.agent` - the encoder, decoder, agent loop and trainer.
"""

from .agent import Agent, Decoder, Encoder, Reward, Trainer
from .brain import Connectome, SpikingNetwork
from .tetris import Action, Piece, TetrisGame

__all__ = [
    'Action',
    'Agent',
    'Connectome',
    'Decoder',
    'Encoder',
    'Piece',
    'Reward',
    'SpikingNetwork',
    'TetrisGame',
    'Trainer',
]

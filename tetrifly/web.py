"""Watch the agent play in a browser.

A background thread plays episodes back to back, pausing for a moment after each
frame so the game is watchable, and publishes a snapshot of the board. The page
polls that snapshot and redraws.

The board is drawn as a grid of square divs whose opacity marks filled cells,
following the approach in robertrypula/simple-tetris:
https://github.com/robertrypula/simple-tetris

Only the standard library is used, so there is no web framework to install.
"""

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .tetris import Action, TetrisGame

#: Loopback only - the viewer is for this machine, not the network.
HOST = '127.0.0.1'

#: Default port for the viewer.
PORT = 8000

PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>TetriFLY</title>
<style>
  :root {
    --ink: #071007;
    --dim: rgba(126, 231, 135, 0.07);
    --locked: #7ee787;
    --falling: #ffc857;
    --text: #9fb8a3;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    background: var(--ink);
    color: var(--text);
    font: 14px/1.5 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    display: flex;
    justify-content: center;
    padding: 24px 16px;
  }
  main { display: flex; gap: 28px; flex-wrap: wrap; max-width: 760px; }
  h1 { margin: 0 0 2px; font-size: 20px; letter-spacing: 3px; color: var(--locked); }
  .sub { margin: 0 0 18px; font-size: 12px; opacity: 0.6; }

  #board {
    display: grid;
    gap: 2px;
    width: 260px;
    padding: 8px;
    border: 1px solid rgba(126, 231, 135, 0.25);
    border-radius: 4px;
    background: rgba(0, 0, 0, 0.35);
  }
  #board div {
    aspect-ratio: 1;
    border-radius: 2px;
    background: var(--dim);
    transition: background-color 90ms linear;
  }
  #board div.locked { background: var(--locked); }
  #board div.falling { background: var(--falling); }

  .panel { min-width: 260px; flex: 1; }
  dl { display: grid; grid-template-columns: auto 1fr; gap: 4px 14px; margin: 0 0 20px; }
  dt { opacity: 0.55; }
  dd { margin: 0; color: var(--locked); }

  h2 { font-size: 12px; letter-spacing: 2px; opacity: 0.55; margin: 0 0 8px;
       font-weight: normal; text-transform: uppercase; }
  .bar { display: grid; grid-template-columns: 110px 1fr 46px;
         align-items: center; gap: 8px; margin-bottom: 5px; }
  .bar span:first-child { opacity: 0.6; }
  .track { height: 8px; background: var(--dim); border-radius: 4px; overflow: hidden; }
  .fill { height: 100%; background: rgba(126, 231, 135, 0.45); width: 0; }
  .bar.chosen span:first-child, .bar.chosen span:last-child { color: var(--falling); }
  .bar.chosen .fill { background: var(--falling); }
  .value { text-align: right; opacity: 0.8; }
  footer { margin-top: 20px; font-size: 11px; opacity: 0.4; }
  #over { color: var(--falling); }
</style>
</head>
<body>
<main>
  <div>
    <h1>TETRIFLY</h1>
    <p class="sub">a fly connectome playing tetris</p>
    <div id="board"></div>
  </div>

  <div class="panel">
    <h2>Game</h2>
    <dl>
      <dt>episode</dt><dd id="episode">-</dd>
      <dt>frame</dt><dd id="frame">-</dd>
      <dt>lines cleared</dt><dd id="lines">-</dd>
      <dt>action</dt><dd id="action">-</dd>
      <dt>reward</dt><dd id="reward">-</dd>
      <dt>dopamine</dt><dd id="dopamine">-</dd>
      <dt>holes</dt><dd id="holes">-</dd>
      <dt>bumpiness</dt><dd id="bumpiness">-</dd>
      <dt>status</dt><dd id="over">-</dd>
    </dl>

    <h2>Brain</h2>
    <dl>
      <dt>neurons</dt><dd id="neurons">-</dd>
      <dt>connections</dt><dd id="connections">-</dd>
      <dt>visual</dt><dd id="visual">-</dd>
      <dt>motor</dt><dd id="motor">-</dd>
      <dt>dopamine</dt><dd id="reward">-</dd>
      <dt>motor firing</dt><dd id="rate">-</dd>
    </dl>

    <h2>Decoder output</h2>
    <div id="logits"></div>

    <footer>
      Board UI after robertrypula/simple-tetris.<br>
      Amber = the piece the brain still controls.
    </footer>
  </div>
</main>

<script>
let cells = [];

function buildBoard(width, height) {
  const board = document.getElementById('board');
  board.style.gridTemplateColumns = `repeat(${width}, 1fr)`;
  board.innerHTML = '';
  cells = [];
  for (let i = 0; i < width * height; i++) {
    cells.push(board.appendChild(document.createElement('div')));
  }
}

function buildLogits(names) {
  document.getElementById('logits').innerHTML = names.map((name, i) => `
    <div class="bar" id="bar${i}">
      <span>${name}</span>
      <span class="track"><span class="fill"></span></span>
      <span class="value"></span>
    </div>`).join('');
}

function drawLogits(logits, chosen) {
  const low = Math.min(...logits), high = Math.max(...logits);
  const span = high - low || 1;
  logits.forEach((value, i) => {
    const bar = document.getElementById('bar' + i);
    bar.classList.toggle('chosen', i === chosen);
    bar.querySelector('.fill').style.width = ((value - low) / span * 100) + '%';
    bar.querySelector('.value').textContent = value.toFixed(2);
  });
}

function draw(state) {
  if (cells.length !== state.width * state.height) {
    buildBoard(state.width, state.height);
    buildLogits(state.action_names);
  }

  cells.forEach(cell => { cell.className = ''; });
  state.locked.forEach((row, y) => row.forEach((filled, x) => {
    if (filled) cells[y * state.width + x].className = 'locked';
  }));
  state.falling.forEach(([y, x]) => { cells[y * state.width + x].className = 'falling'; });

  document.getElementById('episode').textContent = state.episode;
  document.getElementById('frame').textContent = state.frame;
  document.getElementById('lines').textContent = state.lines;
  document.getElementById('action').textContent = state.action || '-';
  document.getElementById('reward').textContent = state.reward_signal;
  document.getElementById('dopamine').textContent = state.dopamine;
  document.getElementById('holes').textContent = state.board.holes;
  document.getElementById('bumpiness').textContent = state.board.bumpiness;
  document.getElementById('over').textContent = state.over ? 'topped out' : 'playing';

  document.getElementById('neurons').textContent = state.neurons.toLocaleString();
  document.getElementById('connections').textContent = state.connections.toLocaleString();
  document.getElementById('visual').textContent = state.visual.toLocaleString();
  document.getElementById('motor').textContent = state.motor.toLocaleString();
  document.getElementById('reward').textContent = state.reward.toLocaleString();
  document.getElementById('rate').textContent = (state.motor_rate * 100).toFixed(1) + '%';

  drawLogits(state.logits, state.action_index);
}

async function poll() {
  try {
    draw(await (await fetch('/state')).json());
  } catch (error) {
    document.getElementById('over').textContent = 'server gone';
  }
  setTimeout(poll, 60);
}
poll();
</script>
</body>
</html>
"""


class GameRunner:
    """Plays episodes in a background thread so a browser can watch.

    Only one snapshot is kept: the most recent frame. The page polls for it, so
    a slow or absent browser never blocks the game.

    Attributes:
        agent: The agent doing the playing.
        delay: Seconds to pause after each frame.
    """

    #: Seconds to pause after each frame, so the game is watchable.
    DELAY = 0.12

    #: Seconds to hold the final board after a game over, before restarting.
    GAME_OVER_PAUSE = 2.0

    def __init__(self, agent, delay=DELAY, max_frames=500, seed=0):
        """Set up a runner.

        Args:
            agent: The agent to watch.
            delay: Seconds to pause after each frame, so play is watchable.
            max_frames: Frame budget per episode.
            seed: Seed for the first episode; later episodes count up from it.
        """
        self.agent = agent
        self.delay = delay
        self.max_frames = max_frames
        self.seed = seed

        self._lock = threading.Lock()
        self._state = self._describe(TetrisGame(seed=seed), action=None, frame=0,
                                     episode=0)

    def start(self):
        """Begin playing in a daemon thread."""
        threading.Thread(target=self._play_forever, daemon=True).start()

    def snapshot(self):
        """Return the most recent frame, safe to serialise.

        Returns:
            A dict describing the board, the brain and the last decision.
        """
        with self._lock:
            return self._state

    def _play_forever(self):
        """Play episode after episode, publishing every frame."""
        episode = 0
        while True:
            episode += 1
            game = TetrisGame(seed=self.seed + episode - 1)
            self.agent.reset()

            for frame in range(1, self.max_frames + 1):
                action = self.agent.play_frame(game)
                self._publish(game, action, frame, episode)
                time.sleep(self.delay)
                if game.over:
                    break

            time.sleep(self.GAME_OVER_PAUSE)

    def _publish(self, game, action, frame, episode):
        """Store a snapshot of the current frame.

        Args:
            game: The game in its current state.
            action: The action just taken.
            frame: Frame number within the episode.
            episode: Episode number since the server started.
        """
        state = self._describe(game, action, frame, episode)
        with self._lock:
            self._state = state

    def _describe(self, game, action, frame, episode):
        """Build the JSON-serialisable description of one frame.

        Args:
            game: The game to describe.
            action: The action just taken, or None before the first frame.
            frame: Frame number within the episode.
            episode: Episode number.

        Returns:
            A plain dict of board, brain and decoder state.
        """
        connectome = self.agent.connectome
        motor_rates = self.agent.rates[connectome.motor]

        return {
            'width': game.WIDTH,
            'height': game.HEIGHT,
            'locked': game.board,
            'falling': [] if game.over else game.piece.cells(),
            'lines': game.lines,
            'over': game.over,
            'frame': frame,
            'episode': episode,
            # Action.LEFT is 0, so these must test against None rather than
            # truthiness - a plain `if action` silently drops every LEFT.
            'action': action.name if action is not None else None,
            'action_index': int(action) if action is not None else -1,
            'action_names': [a.name for a in Action],
            'logits': [round(v, 4) for v in self.agent.logits.tolist()],
            'motor_rate': round(float(motor_rates.mean()), 4),
            # The reward rule's verdict on the last frame, and the current it
            # became at the dopamine neurons.
            'reward_signal': round(float(self.agent.reward), 2),
            'dopamine': round(float(self.agent.dopamine), 4),
            'board': game.features(),
            'neurons': len(connectome),
            'connections': connectome.matrix._nnz(),
            'visual': len(connectome.visual),
            'motor': len(connectome.motor),
            'reward': len(connectome.reward),
        }


class Handler(BaseHTTPRequestHandler):
    """Serves the page and the current game state.

    The runner is attached to the class by :func:`serve` before the server
    starts, which is how http.server hands dependencies to a handler.
    """

    runner = None

    def do_GET(self):
        """Answer a GET for the page or the state."""
        if self.path == '/':
            self._respond('text/html; charset=utf-8', PAGE.encode())
        elif self.path == '/state':
            body = json.dumps(self.runner.snapshot()).encode()
            self._respond('application/json', body)
        else:
            self.send_error(404)

    def _respond(self, content_type, body):
        """Send one response.

        Args:
            content_type: Value for the Content-Type header.
            body: Response body as bytes.
        """
        self.send_response(200)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        """Silence the per-request logging; the page polls constantly."""


def serve(agent, port=PORT, delay=GameRunner.DELAY, max_frames=500, seed=0):
    """Play in a background thread and serve the viewer until interrupted.

    Args:
        agent: The agent to watch.
        port: Port to listen on.
        delay: Seconds to pause after each frame.
        max_frames: Frame budget per episode.
        seed: Seed for the first episode.
    """
    runner = GameRunner(agent, delay=delay, max_frames=max_frames, seed=seed)
    runner.start()

    Handler.runner = runner
    server = ThreadingHTTPServer((HOST, port), Handler)

    print(f'Watch at http://{HOST}:{port}  (Ctrl+C to stop)')
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('\nStopping.')
    finally:
        server.server_close()

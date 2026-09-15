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

import torch

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
    --dopa: #ff79c6;
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
  main { display: flex; gap: 26px; flex-wrap: wrap; max-width: 1120px; }
  h1 { margin: 0 0 2px; font-size: 20px; letter-spacing: 3px; color: var(--locked); }
  .sub { margin: 0 0 6px; font-size: 12px; opacity: 0.6; }
  .census { margin: 0 0 16px; font-size: 11px; opacity: 0.45; max-width: 260px; }

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

  .brain { width: 320px; }
  #fly {
    width: 320px; height: 320px; display: block;
    border: 1px solid rgba(126, 231, 135, 0.25);
    border-radius: 4px;
    background: #040804;
  }
  .legend { display: flex; gap: 14px; flex-wrap: wrap; font-size: 11px;
            margin-top: 8px; opacity: 0.75; }
  .legend i { display: inline-block; width: 8px; height: 8px; border-radius: 2px;
              margin-right: 5px; vertical-align: middle; }
  .hint { font-size: 11px; opacity: 0.4; margin-top: 8px; }

  .panel { min-width: 240px; flex: 1; }
  dl { display: grid; grid-template-columns: auto 1fr; gap: 4px 14px; margin: 0 0 18px; }
  dt { opacity: 0.55; }
  dd { margin: 0; color: var(--locked); }

  h2 { font-size: 12px; letter-spacing: 2px; opacity: 0.55; margin: 0 0 8px;
       font-weight: normal; text-transform: uppercase; }
  .bar { display: grid; grid-template-columns: 110px 1fr 56px;
         align-items: center; gap: 8px; margin-bottom: 5px; }
  .bar span:first-child { opacity: 0.6; }
  .track { height: 8px; background: var(--dim); border-radius: 4px; overflow: hidden; }
  .fill { height: 100%; background: rgba(126, 231, 135, 0.45); width: 0; }
  .bar.chosen span:first-child, .bar.chosen span:last-child { color: var(--falling); }
  .bar.chosen .fill { background: var(--falling); }
  .value { text-align: right; opacity: 0.8; }
  footer { margin-top: 18px; font-size: 11px; opacity: 0.4; }
  #over { color: var(--falling); }
  #reward.good { color: var(--locked); }
  #reward.bad { color: var(--falling); }
</style>
</head>
<body>
<main>
  <div>
    <h1>TETRIFLY</h1>
    <p class="sub">a fly connectome playing tetris</p>
    <p class="census" id="census">&nbsp;</p>
    <div id="board"></div>
  </div>

  <div class="brain">
    <h2>Brain activity</h2>
    <canvas id="fly"></canvas>
    <div class="legend">
      <span><i style="background:#7ee787"></i>visual</span>
      <span><i style="background:#ffc857"></i>motor</span>
      <span><i style="background:#ff79c6"></i>dopamine</span>
      <span><i style="background:#2c4430"></i>other</span>
    </div>
    <p class="hint" id="sampled">&nbsp;</p>
  </div>

  <div class="panel">
    <h2>Game</h2>
    <dl>
      <dt>episode</dt><dd id="episode">-</dd>
      <dt>frame</dt><dd id="frame">-</dd>
      <dt>lines cleared</dt><dd id="lines">-</dd>
      <dt>action</dt><dd id="action">-</dd>
      <dt>status</dt><dd id="over">-</dd>
    </dl>

    <h2>Teaching signal</h2>
    <dl>
      <dt>reward this frame</dt><dd id="reward">-</dd>
      <dt>episode total</dt><dd id="total">-</dd>
      <dt>dopamine current</dt><dd id="dopamine">-</dd>
    </dl>

    <h2>Board quality</h2>
    <dl>
      <dt>holes</dt><dd id="holes">-</dd>
      <dt>bumpiness</dt><dd id="bumpiness">-</dd>
      <dt>stack height</dt><dd id="height">-</dd>
      <dt>wells</dt><dd id="wells">-</dd>
    </dl>

    <h2>Firing</h2>
    <dl>
      <dt>motor neurons</dt><dd id="rate">-</dd>
      <dt>brain-wide</dt><dd id="active">-</dd>
    </dl>

    <h2>Decoder output</h2>
    <div id="logits"></div>

    <footer>
      Board UI after robertrypula/simple-tetris.<br>
      Amber = the piece the brain still controls.<br>
      Neurons drawn at their real soma coordinates from neuPrint.
    </footer>
  </div>
</main>

<script>
const boardEl = document.getElementById('board');
const canvas = document.getElementById('fly');
const ctx = canvas.getContext('2d');
let cells = [], names = null, layout = null, size = 320;

function buildBoard(width, height) {
  boardEl.style.gridTemplateColumns = `repeat(${width}, 1fr)`;
  boardEl.innerHTML = '';
  cells = [];
  for (let i = 0; i < width * height; i++) {
    const cell = document.createElement('div');
    boardEl.appendChild(cell);
    cells.push(cell);
  }
}

function buildLogits(actionNames) {
  names = actionNames;
  document.getElementById('logits').innerHTML = actionNames.map((name, i) => `
    <div class="bar" id="bar${i}">
      <span>${name.toLowerCase().replace('_', ' ')}</span>
      <div class="track"><div class="fill"></div></div>
      <span class="value"></span>
    </div>`).join('');
}

function drawBoard(state) {
  const falling = new Set(state.falling.map(([r, c]) => r * state.width + c));
  state.locked.forEach((row, r) => row.forEach((filled, c) => {
    const i = r * state.width + c;
    cells[i].className = falling.has(i) ? 'falling' : (filled ? 'locked' : '');
  }));
}

function drawLogits(logits, chosen) {
  const low = Math.min(...logits), high = Math.max(...logits);
  const span = (high - low) || 1;
  logits.forEach((value, i) => {
    const bar = document.getElementById('bar' + i);
    bar.classList.toggle('chosen', i === chosen);
    bar.querySelector('.fill').style.width = (8 + 92 * (value - low) / span) + '%';
    bar.querySelector('.value').textContent = value.toFixed(3);
  });
}

// Neuron colours by population: other, visual, motor, dopamine.
const POP = [[44, 68, 48], [126, 231, 135], [255, 200, 87], [255, 121, 198]];

function sizeCanvas() {
  const ratio = window.devicePixelRatio || 1;
  canvas.width = size * ratio;
  canvas.height = size * ratio;
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
}

function drawBrain(activity) {
  if (!layout) return;
  ctx.clearRect(0, 0, size, size);
  ctx.fillStyle = '#040804';
  ctx.fillRect(0, 0, size, size);

  const pad = 10, span = size - 2 * pad;
  for (let i = 0; i < layout.x.length; i++) {
    const pop = layout.pop[i];
    const a = activity ? activity[i] / 255 : 0;
    const [r, g, b] = POP[pop];
    // Inactive neurons stay faintly visible so the anatomy is always legible;
    // activity brightens and enlarges them.
    const alpha = pop === 0 ? 0.10 + 0.85 * a : 0.30 + 0.70 * a;
    const px = pad + layout.x[i] * span;
    const py = pad + layout.y[i] * span;
    const d = (pop === 0 ? 1.4 : 2.2) + 2.2 * a;
    ctx.fillStyle = `rgba(${r},${g},${b},${alpha.toFixed(3)})`;
    ctx.fillRect(px - d / 2, py - d / 2, d, d);
  }
}

function render(state) {
  if (!cells.length) buildBoard(state.width, state.height);
  if (!names) buildLogits(state.action_names);

  drawBoard(state);
  document.getElementById('episode').textContent = state.episode;
  document.getElementById('frame').textContent = state.frame;
  document.getElementById('lines').textContent = state.lines;
  document.getElementById('action').textContent = state.action || '-';
  document.getElementById('over').textContent = state.over ? 'topped out' : 'playing';

  const reward = document.getElementById('reward');
  reward.textContent = (state.reward > 0 ? '+' : '') + state.reward.toFixed(1);
  reward.className = state.reward > 0 ? 'good' : (state.reward < 0 ? 'bad' : '');
  document.getElementById('total').textContent = state.reward_total.toFixed(1);
  document.getElementById('dopamine').textContent = state.dopamine.toFixed(4);

  document.getElementById('holes').textContent = state.board.holes;
  document.getElementById('bumpiness').textContent = state.board.bumpiness;
  document.getElementById('height').textContent = state.board.aggregate_height;
  document.getElementById('wells').textContent = state.board.wells;

  document.getElementById('rate').textContent = state.motor_rate.toFixed(4);
  document.getElementById('active').textContent =
      (100 * state.active_fraction).toFixed(1) + '% firing';

  drawLogits(state.logits, state.action_index);
  drawBrain(state.activity);

}

async function boot() {
  sizeCanvas();
  const res = await fetch('/brain');
  layout = await res.json();
  document.getElementById('census').textContent = layout.census;
  document.getElementById('sampled').textContent =
      layout.sampled.toLocaleString() + ' of ' + layout.total.toLocaleString() +
      ' neurons drawn, sampled to keep the frame small';
  drawBrain(null);
  poll();
}

async function poll() {
  try {
    const res = await fetch('/state');
    render(await res.json());
  } catch (err) { /* server going away; keep trying */ }
  setTimeout(poll, 120);
}

boot();
</script>
</body>
</html>
"""


def brain_layout(connectome, sample=4200, seed=0):
    """Choose neurons to draw and project them onto the page.

    All 176,422 neurons cannot go down the wire every frame, so this picks a
    representative sample once: every motor and dopamine neuron, because those
    are the populations worth watching, plus a slice of the visual neurons and
    of everything else for context.

    The projection is the frontal view: neuPrint's x (left-right) and y
    (dorsal-ventral) soma coordinates, normalised to the unit square, with y
    flipped so dorsal is up. Neurons whose cell type has no soma location in
    neuPrint are skipped.

    Args:
        connectome: The brain to draw. Must carry positions.
        sample: Roughly how many neurons to draw.
        seed: Seed for choosing which neurons to include.

    Returns:
        A dict of parallel lists the page renders directly: x, y in [0, 1] and
        pop in 0..3 (other, visual, motor, dopamine), plus a census string.

    Raises:
        ValueError: If the connectome has no positions.
    """
    if connectome.positions is None:
        raise ValueError('This connectome has no soma positions. Rebuild it, '
                         'or serve without the brain panel.')

    positions = connectome.positions
    placed = ~torch.isnan(positions[:, 0])

    # Population code per neuron; later assignments win, so the small
    # populations are never masked by the large ones.
    pop = torch.zeros(len(connectome), dtype=torch.long)
    pop[connectome.visual] = 1
    pop[connectome.motor] = 2
    pop[connectome.reward] = 3

    generator = torch.Generator().manual_seed(seed)
    keep = torch.zeros(len(connectome), dtype=torch.bool)
    for code, share in ((3, 1.0), (2, 1.0), (1, 0.09), (0, 0.02)):
        members = ((pop == code) & placed).nonzero().squeeze(1)
        if share >= 1.0:
            chosen = members
        else:
            take = max(1, int(len(members) * share))
            chosen = members[torch.randperm(len(members), generator=generator)[:take]]
        keep[chosen] = True

    index = keep.nonzero().squeeze(1)
    xy = positions[index][:, :2]
    low, high = xy.min(dim=0).values, xy.max(dim=0).values
    unit = (xy - low) / (high - low).clamp(min=1e-6)

    return {
        'index': index,
        'x': [round(v, 4) for v in unit[:, 0].tolist()],
        # Flip y so dorsal is at the top of the canvas.
        'y': [round(1.0 - v, 4) for v in unit[:, 1].tolist()],
        'pop': pop[index].tolist(),
        'sampled': len(index),
        'total': len(connectome),
        'census': (f'{len(connectome):,} neurons - '
                   f'{connectome.matrix._nnz() / 1e6:.1f}M synapses - '
                   f'{len(connectome.visual):,} visual - '
                   f'{len(connectome.motor):,} motor - '
                   f'{len(connectome.reward):,} dopamine'),
    }


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

        self.layout = brain_layout(agent.connectome)
        self._sample = self.layout['index']
        self._total = 0.0
        self._frame = 0

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
            self._total = 0.0

            for frame in range(1, self.max_frames + 1):
                action = self.agent.play_frame(game)
                self._total += self.agent.reward
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
            'active_fraction': round(float((self.agent.rates > 0).float().mean()), 4),
            # The reward rule's verdict on the last frame, the episode so far,
            # and the current the reward became at the dopamine neurons.
            'reward': round(float(self.agent.reward), 2),
            'reward_total': round(float(self._total), 1),
            'dopamine': round(float(self.agent.dopamine), 4),
            'board': game.features(),
            # Firing rate of each drawn neuron, quantised to a byte to keep the
            # frame small. The page holds the matching layout from /brain.
            'activity': (self.agent.rates[self._sample]
                         .clamp(0, 1).mul(255).to(torch.uint8).tolist()),
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
        elif self.path == '/brain':
            # The neuron layout never changes, so the page fetches it once and
            # only the per-frame activity rides along with /state.
            layout = {k: v for k, v in self.runner.layout.items() if k != 'index'}
            self._respond('application/json', json.dumps(layout).encode())
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

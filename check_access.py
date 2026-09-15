"""Verify that the real neuPrint connectome is reachable and usable.

Run this once after putting your token in ``tetrifly/local_config.py`` and
before the long ``python main.py build``::

    python check_access.py

Every check is cheap. The queries here return per-type aggregates, never the
~125M-synapse adjacency table, so the whole run takes seconds. The point is to
fail fast on a bad token or a renamed cell type rather than deep into a build.

Exit code is 0 if every check passed, 1 otherwise.
"""

import os
import sys
from pathlib import Path

import pandas as pd

from tetrifly.brain import Connectome

# Populations the pipeline locates by regex, as (label, attribute) pairs.
POPULATIONS = [('visual', 'VISUAL_TYPES'),
               ('motor', 'MOTOR_TYPES'),
               ('reward', 'REWARD_TYPES')]

# Columns _signed_strengths and _build_matrix read off the neuron table.
REQUIRED_COLUMNS = ['bodyId', 'type', 'predictedNt', 'predictedNtProb']

# Neurons to pull per transmitter when checking that signs still work.
SIGN_SAMPLE = 500


def report(ok, label, detail=''):
    """Print one check result and return its pass/fail flag."""
    print(f'  [{"PASS" if ok else "FAIL"}] {label}' + (f' -- {detail}' if detail else ''))
    return ok


def check_token():
    """Confirm a token is configured, without printing it."""
    print('token')
    token, source = Connectome.TOKEN, 'local_config.TOKEN'
    if not token:
        token = os.environ.get(Connectome.TOKEN_ENV_VAR, '')
        source = f'${Connectome.TOKEN_ENV_VAR}'

    if not token:
        return report(False, 'configured', 'set TOKEN in tetrifly/local_config.py')
    # neuPrint hands out long tokens; a short value is usually a truncated paste.
    return report(len(token) > 20, f'configured from {source}', f'{len(token)} chars')


def check_client():
    """Connect and confirm the server accepts the token.

    Returns:
        A live Client, or None if the connection failed.
    """
    print()
    print('connection')
    from neuprint import Client
    try:
        client = Client(Connectome.SERVER, dataset=Connectome.DATASET,
                        token=Connectome.TOKEN or None)
        version = client.fetch_version()
    except Exception as exc:
        report(False, f'connect to {Connectome.SERVER}',
               f'{type(exc).__name__}: {exc}')
        return None

    report(True, f'connect to {Connectome.SERVER}', f'server v{version}')
    report(True, f'dataset {Connectome.DATASET!r}', 'accepted')
    return client


def check_schema(client):
    """Confirm the neuron table still has the columns the build reads."""
    print()
    print('neuron table schema')
    projection = ', '.join(f'n.{c} AS {c}' for c in REQUIRED_COLUMNS)
    rows = client.fetch_custom(f'MATCH (n:Neuron) WHERE n.type IS NOT NULL '
                               f'RETURN {projection} LIMIT 3')

    missing = [c for c in REQUIRED_COLUMNS if c not in rows.columns]
    ok = report(not missing, f'columns present ({", ".join(REQUIRED_COLUMNS)})',
                f'missing {missing}' if missing else f'{len(rows)} sample rows')
    if ok:
        print(rows.to_string(index=False, max_colwidth=22))
    return ok


def check_populations(client):
    """Confirm each regex the agent depends on still matches real cell types.

    This is the check most likely to catch a real problem: if neuPrint renames
    a cell type, the regex matches nothing and ``_find_population`` raises --
    but only after the entire download has finished.

    The census covers every cell type in the dataset, with no LIMIT. Sampling
    here produces false failures, because the types being looked for are a
    handful out of many thousands. Matching uses the same pandas call as
    ``Connectome._find_population``, so a pass here means that code passes.
    """
    print()
    print('populations (the regexes in Connectome)')
    census = client.fetch_custom('MATCH (n:Neuron) WHERE n.type IS NOT NULL '
                                 'RETURN n.type AS type, count(*) AS neurons')
    types = census['type']
    print(f'  {len(types):,} distinct cell types, '
          f'{census["neurons"].sum():,} typed neurons')

    all_ok = True
    for label, attribute in POPULATIONS:
        pattern = getattr(Connectome, attribute)
        matched = types.str.contains(pattern, na=False, regex=True)
        names = sorted(types[matched])
        neurons = int(census.loc[matched, 'neurons'].sum())

        detail = f'{pattern} -> {len(names)} types, {neurons:,} neurons'
        if names:
            detail += f' e.g. {", ".join(names[:4])}'
            detail += ' ...' if len(names) > 4 else ''
        all_ok &= report(bool(names), f'{label:6} {attribute}', detail)
    return all_ok


def check_signs(client):
    """Confirm real transmitters still produce both excitation and inhibition.

    ``_signed_strengths`` can fail silently: if the transmitter lookup yields
    nothing usable, every synapse falls back to ``DEFAULT_SIGN`` and the brain
    becomes purely excitatory. Nothing raises -- the matrix is simply wrong.
    So run the real classmethod over a transmitter-spanning sample and insist
    that inhibition survives.

    The sample is drawn per transmitter on purpose. An arbitrary ``LIMIT`` over
    the neuron table returns acetylcholine rows almost exclusively, which would
    pass this check while proving nothing.
    """
    print()
    print('synapse signs')
    inhibitory_nts = [nt for nt, sign in Connectome.TRANSMITTER_SIGNS.items()
                      if sign < 0]

    frames = []
    for transmitter in list(Connectome.TRANSMITTER_SIGNS) + ['dopamine', 'unclear']:
        frames.append(client.fetch_custom(
            f"MATCH (n:Neuron) WHERE n.predictedNt = '{transmitter}' "
            f"RETURN n.bodyId AS bodyId, n.type AS type, "
            f"n.predictedNt AS predictedNt, "
            f"n.predictedNtProb AS predictedNtProb LIMIT {SIGN_SAMPLE}"))
    neurons = pd.concat(frames, ignore_index=True)

    graded = int(neurons['predictedNtProb'].notna().sum())
    report(True, 'confidence data', f'predictedNtProb set on {graded}/{len(neurons)} '
                                    f'sampled neurons'
                                    + ('' if graded else ' (gate skipped)'))

    # weight 1 everywhere, so the sign is the only thing the result reflects.
    pairs = pd.DataFrame({'bodyId_pre': neurons['bodyId'],
                          'bodyId_post': neurons['bodyId'], 'weight': 1})
    strengths = Connectome._signed_strengths(neurons, pairs)

    inhibitory = int((strengths < 0).sum())
    expected = int(neurons['predictedNt'].str.lower().isin(inhibitory_nts).sum())
    return report(inhibitory > 0 and inhibitory == expected,
                  f'inhibition survives ({"+".join(inhibitory_nts)})',
                  f'{int((strengths > 0).sum()):,} excitatory, '
                  f'{inhibitory:,} inhibitory, expected {expected:,}')


def check_cache():
    """Report on the cached connectome, if the build has already been run."""
    print()
    print('cached connectome')
    from main import CONNECTOME_PATH

    if not Path(CONNECTOME_PATH).exists():
        return report(True, f'{CONNECTOME_PATH} absent',
                      'expected until you run: python main.py build')

    connectome = Connectome.load(CONNECTOME_PATH)
    counts = {label: len(getattr(connectome, label)) for label, _ in POPULATIONS}
    empty = [label for label, count in counts.items() if count == 0]
    summary = ', '.join(f'{label}={count:,}' for label, count in counts.items())
    return report(not empty, f'loaded {CONNECTOME_PATH}',
                  f'{len(connectome):,} neurons, {summary}')


def main():
    """Run every check and summarise."""
    print(f'Checking access to {Connectome.DATASET} on {Connectome.SERVER}')
    print()

    results = [check_token()]
    client = check_client() if results[0] else None
    if client is not None:
        results += [check_schema(client), check_populations(client),
                    check_signs(client)]
    results.append(check_cache())

    passed = all(results)
    print()
    print(f'{"all checks passed" if passed else "SOME CHECKS FAILED"} '
          f'({sum(bool(r) for r in results)}/{len(results)})')
    if passed and not Path('data/connectome.pt').exists():
        print()
        print('Next: python main.py build   (large download, runs once)')
    return 0 if passed else 1


if __name__ == '__main__':
    sys.exit(main())

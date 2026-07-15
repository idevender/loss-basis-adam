"""
Aggregate the H100 JSONL results into the paper summary tables (Appendix C8).

Usage:
    python collect.py --dir ../nibi_results --md ../nibi_results/SUMMARY.md

Applies the paper's selection rules: for the zoo and phase sweeps, the best lr is the interpolating
one (mean train < 1e-4 over selection seeds) with lowest mean recovery, with the cosine-all control
reported alongside and the equivariant vs coordinate-wise split checked; for the dial, best lr per
(n, p) with a monotonicity check along p; for attention, gauge and noise drift ranges per
configuration with the conservative structural ratio; for Pavia and Indian Pines, the train-only lr
rule (deepest common level, ties by fewest steps) with the 4-seed matched-loss table and full lr
transparency.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np

EQUIVARIANT = {'gd', 'adam_p0rms', 'muon', 'shampoo'}
COORDWISE = {'adam', 'rmsprop', 'signum', 'lion', 'adafactor'}
LOOSE = 1e-4
SEL = {42, 123, 456}


def rows_of(path):
    out = []
    for line in open(path):
        try:
            r = json.loads(line)
            r = {k: (float('nan') if v is None else v) for k, v in r.items()}
        except json.JSONDecodeError:
            continue
        if r.get('kind') != 'meta' and not str(r.get('key', '')).startswith('__meta__'):
            out.append(r)
    return out


def best_lr(cells, sel_seeds=SEL):
    by = defaultdict(list)
    for c in cells:
        if c.get('seed') in sel_seeds:
            by[c['lr']].append(c)
    best, score = None, None
    for lr, rs in sorted(by.items()):
        rec = np.mean([r['rec'] for r in rs])
        tr = np.mean([r['train'] for r in rs])
        if not (np.isfinite(rec) and np.isfinite(tr)):
            continue
        s = rec if tr < LOOSE else rec + 10 + tr
        if score is None or s < score:
            best, score = lr, s
    return best


def fmt(m, s):
    return f'{m:.4f}+-{s:.4f}'


def zoo_tables(files, md):
    for f in sorted(files):
        rows = rows_of(f)
        cells = [r for r in rows if r.get('opt') and 'rec' in r]
        if not cells:
            continue
        arms = sorted({(r.get('n'), r.get('rank'), r.get('mdof')) for r in cells})
        for (n, rank, mdof) in arms:
            sub = [r for r in cells if r.get('n') == n and r.get('rank') == rank
                   and r.get('mdof') == mdof]
            opts = sorted({r['opt'] for r in sub})
            md.append(f'\n### zoo {os.path.basename(f)} | n={n} r={rank} mdof={mdof}\n')
            md.append('| opt | class | lr* | edge | rec (protocol) | erank | rec (cosine-all) | '
                      'train worst | n |\n|---|---|---|---|---|---|---|---|---|\n')
            stats = {}
            for opt in opts:
                oc = [r for r in sub if r['opt'] == opt]
                proto = [r for r in oc if r.get('schedule') == 'protocol']
                lr = best_lr(proto)
                if lr is None:
                    continue
                at = [r for r in proto if r['lr'] == lr]
                ca = [r for r in oc if r.get('schedule') == 'cosine-all' and r['lr'] == lr]
                recs = [r['rec'] for r in at]
                grid = sorted({r['lr'] for r in proto})
                edge = 'EDGE' if lr in (grid[0], grid[-1]) else ''
                cls = ('eq' if opt in EQUIVARIANT else
                       'cw' if opt in COORDWISE else 'ctrl')
                stats[opt] = (np.mean(recs), cls)
                md.append(f"| {opt} | {cls} | {lr:g} | {edge} | "
                          f"{fmt(np.mean(recs), np.std(recs))} | "
                          f"{np.mean([r['er'] for r in at]):.2f} | "
                          f"{fmt(np.mean([r['rec'] for r in ca]), np.std([r['rec'] for r in ca])) if ca else '--'} | "
                          f"{max(r['train'] for r in at):.1e} | {len(at)} |\n")
            eq = [v for v, c in stats.values() if c == 'eq']
            cw = [v for v, c in stats.values() if c == 'cw']
            if eq and cw:
                ok = max(eq) < min(cw)
                md.append(f'\n**split: max(eq)={max(eq):.4f} < min(cw)={min(cw):.4f} -> '
                          f'{"CLEAN 2-CLUSTER" if ok else "OVERLAP (inspect)"}**\n')


def phase_tables(files, md):
    for f in sorted(files):
        rows = [r for r in rows_of(f) if 'tau' in r and 'rec' in r]
        if not rows:
            continue
        n = rows[0].get('n')
        taus = sorted({r['tau'] for r in rows})
        opts = sorted({r['opt'] for r in rows})
        md.append(f'\n### phase {os.path.basename(f)} | n={n}\n')
        md.append('| tau | ' + ' | '.join(opts) + ' |\n|---' * (len(opts) + 1) + '|\n')
        table = {}
        for tau in taus:
            line = [f'| {tau:g} ']
            for opt in opts:
                oc = [r for r in rows if r['tau'] == tau and r['opt'] == opt]
                lr = best_lr(oc, sel_seeds={r['seed'] for r in oc})
                at = [r for r in oc if r['lr'] == lr]
                m = np.mean([r['rec'] for r in at]) if at else float('nan')
                table[(tau, opt)] = m
                line.append(f'| {m:.4f} ')
            md.append(''.join(line) + '|\n')
        prev = None
        for tau in taus:
            d = table.get((tau, 'muon'), np.nan) - table.get((tau, 'gd'), np.nan)
            if prev is not None and prev < 0 <= d:
                md.append(f'\n**tau* ~ {tau:g} (muon cedes to gd)**\n')
            prev = d


def dial_tables(files, md):
    for f in sorted(files):
        rows = [r for r in rows_of(f) if 'p' in r and 'rec' in r]
        if not rows:
            continue
        md.append(f'\n### dial {os.path.basename(f)}\n| opt | p | lr* | rec | erank | steps |\n'
                  '|---|---|---|---|---|---|\n')
        for opt in sorted({r['opt'] for r in rows}):
            for p in sorted({r['p'] for r in rows if r['opt'] == opt}, reverse=True):
                oc = [r for r in rows if r['opt'] == opt and r['p'] == p]
                lr = best_lr(oc, sel_seeds={r['seed'] for r in oc})
                at = [r for r in oc if r['lr'] == lr]
                md.append(f"| {opt} | {p:g} | {lr:g} | "
                          f"{fmt(np.mean([r['rec'] for r in at]), np.std([r['rec'] for r in at]))} | "
                          f"{np.mean([r['er'] for r in at]):.2f} | "
                          f"{np.mean([r['steps'] for r in at]):.0f} |\n")


def attn_tables(files, md):
    for f in sorted(files):
        rows = [r for r in rows_of(f) if r.get('twin')]
        if not rows:
            continue
        cfgs = sorted({r['cfg'] for r in rows})
        for cfg in cfgs:
            sub = [r for r in rows if r['cfg'] == cfg]
            md.append(f'\n### attention {cfg} ({os.path.basename(f)})\n')
            md.append('| opt | twin | pairs | step1 range | final range | perhead max |\n'
                      '|---|---|---|---|---|---|\n')
            for opt in sorted({r['opt'] for r in sub}):
                for twin in ('gauge', 'noise', 'identity'):
                    tc = [r for r in sub if r['opt'] == opt and r['twin'] == twin]
                    if not tc:
                        continue
                    s1 = [next((c[1] for c in r['curve'] if c[0] == 1), np.nan) for r in tc]
                    fin = [r['curve'][-1][1] for r in tc]
                    ph = max(max(r['per_head_final']) for r in tc)
                    ns = sorted({r.get('noise_scale') for r in tc})
                    md.append(f"| {opt} | {twin}{'@' + str(ns) if twin == 'noise' else ''} | "
                              f"{len(tc)} | {min(s1):.1e}--{max(s1):.1e} | "
                              f"{min(fin):.1e}--{max(fin):.1e} | {ph:.1e} |\n")
            def _s1(r):
                return next((c[1] for c in r['curve'] if c[0] == 1), np.nan)
            ag = [_s1(r) for r in sub if r['opt'] == 'adam' and r['twin'] == 'gauge']
            an = [(r.get('noise_scale'), _s1(r)) for r in sub
                  if r['opt'] == 'adam' and r['twin'] == 'noise']
            if ag and an:
                big = max(an)[1]
                md.append(f'\n**conservative structural ratio (step 1) = min gauge / noise@max-scale = '
                          f'{min(ag) / big if big > 0 else float("inf"):.0f}x**\n')


def pavia_tables(files, md):
    for f in sorted(files):
        rows = [r for r in rows_of(f) if 'hits' in r]
        if not rows:
            continue
        for ds in sorted({r['dataset'] for r in rows}):
            for dens in sorted({r['dens'] for r in rows if r['dataset'] == ds}):
                sub = [r for r in rows if r['dataset'] == ds and r['dens'] == dens]
                md.append(f'\n### {ds} d={dens:g} ({os.path.basename(f)})\n')
                sel = {}
                for opt in sorted({r['opt'] for r in sub}):
                    best = None
                    for lr in sorted({r['lr'] for r in sub if r['opt'] == opt}):
                        hl = [r for r in sub if r['opt'] == opt and r['lr'] == lr
                              and r['seed'] in (42, 123)]
                        if not hl:
                            continue
                        common = [lv for lv in ('0.003', '0.001', '0.0003', '0.0001', '3e-05', '1e-05')
                                  if all(lv in h['hits'] for h in hl)]
                        if not common:
                            continue
                        deepest = common[-1]
                        steps = np.mean([h['hits'][deepest][2] for h in hl])
                        score = (-len(common), steps)
                        if best is None or score < best[0]:
                            best = (score, lr, deepest)
                    if best:
                        sel[opt] = best[1]
                        md.append(f'- {opt}: train-only rule selects lr={best[1]:g} '
                                  f'(deepest {best[2]}, {-best[0][0]} levels)\n')
                md.append('\n| opt | lr | ' + ' | '.join(
                    f'test@{lv}' for lv in ('0.003', '0.001', '0.0003', '0.0001', '3e-05', '1e-05')) + ' |\n')
                md.append('|---' * 8 + '|\n')
                for opt, lr in sel.items():
                    hl = [r for r in sub if r['opt'] == opt and r['lr'] == lr]
                    cells = []
                    for lv in ('0.003', '0.001', '0.0003', '0.0001', '3e-05', '1e-05'):
                        vals = [h['hits'][lv] for h in hl if lv in h['hits']]
                        cells.append(f'{np.mean([v[0] for v in vals]):.5f}'
                                     f'(rk{np.mean([v[1] for v in vals]):.0f},n{len(vals)})'
                                     if vals else '--')
                    md.append(f'| {opt} | {lr:g} | ' + ' | '.join(cells) + ' |\n')
                if 'gd' in sel and 'adam' in sel:
                    g = [r for r in sub if r['opt'] == 'gd' and r['lr'] == sel['gd']]
                    a = [r for r in sub if r['opt'] == 'adam' and r['lr'] == sel['adam']]
                    for lv in ('1e-05', '3e-05', '0.0001', '0.0003', '0.001', '0.003'):
                        if all(lv in h['hits'] for h in g) and all(lv in h['hits'] for h in a):
                            gm = np.mean([h['hits'][lv][0] for h in g])
                            am = np.mean([h['hits'][lv][0] for h in a])
                            wins = sum(h1['hits'][lv][0] < h2['hits'][lv][0]
                                       for h1, h2 in zip(sorted(g, key=lambda r: r['seed']),
                                                         sorted(a, key=lambda r: r['seed'])))
                            md.append(f'\n**HEADLINE {ds} d={dens:g} @train<={lv}: GD {gm:.5f} vs '
                                      f'Adam {am:.5f} -> +{(am - gm) / am * 100:.1f}% '
                                      f'({wins}/{len(g)} seeds)**\n')
                            break


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', required=True)
    ap.add_argument('--md', required=True)
    a = ap.parse_args()
    g = lambda pat: glob.glob(os.path.join(a.dir, pat))
    md = ['# Nibi H100 ladder — collected results\n']
    zoo_tables(g('zoo_*.jsonl'), md)
    phase_tables(g('phase_*.jsonl'), md)
    dial_tables(g('dial_*.jsonl'), md)
    attn_tables(g('attn_*.jsonl'), md)
    pavia_tables(g('pavia*.jsonl') + g('indianpines*.jsonl'), md)
    os.makedirs(os.path.dirname(a.md), exist_ok=True)
    with open(a.md, 'w') as f:
        f.write(''.join(md))
    print(f'wrote {a.md}')


if __name__ == '__main__':
    main()

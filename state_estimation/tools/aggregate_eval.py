"""Combine several evaluation runs into one thesis table (mean +- std across runs).

Reads every *_summary.txt that plot_eval.py wrote in a folder, groups the runs by
their run_name (the part before _YYYYMMDD_HHMMSS) and prints mean +- std per group.
Each run is counted once, even if it appears in several summary files.

Usage:
    python3 aggregate_eval.py ~/eval_logs
Writes ~/eval_logs/aggregate_table.md as well.
"""
import glob
import os
import re
import statistics
import sys

ROW = re.compile(r'^(EKF|AMCL|Wheel)\s+(position|heading) error \[(m|deg)\]\s+mean\s+([\d.]+)\s+rms\s+([\d.]+)\s+p95\s+([\d.]+)\s+max\s+([\d.]+)')
INSIDE = re.compile(r'inside its \+-2 sigma: (\d+) %')
WIN = re.compile(r'^EKF position (first|after) 20 m \[m\]\s+mean\s+([\d.]+)\s+rms\s+[\d.]+\s+p95\s+[\d.]+\s+max\s+([\d.]+)')
KSTART = re.compile(r'Speed scale k start ([\d.]+)')
KFIN = re.compile(r'Speed scale k final ([\d.]+) \(true ([\d.]+)\)')

folder = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else '~/eval_logs')
runs = {}
for path in sorted(glob.glob(os.path.join(folder, '*_summary.txt'))):
    name = None
    for line in open(path):
        line = line.strip()
        m = re.match(r'^== (\S+) ==(.*)', line)
        if m:
            # only blocks made by the fixed plot_eval (driving-time statistics);
            # older summaries counted the stopped time and are skipped
            name = m.group(1) if 'driving time only' in m.group(2) else None
            if name:
                runs.setdefault(name, {})
            continue
        if name is None:
            continue
        r = runs[name]
        m = ROW.match(line)
        if m:
            src, kind, unit = m.group(1), m.group(2), m.group(3)
            scale = 100.0 if unit == 'm' else 1.0          # positions in cm
            for stat, val in zip(('mean', 'p95', 'max'), (m.group(4), m.group(6), m.group(7))):
                r[f'{src} {kind} {stat}'] = float(val) * scale
        m = WIN.match(line)
        if m:
            r[f'EKF position {m.group(1)} 20 m mean'] = float(m.group(2)) * 100.0
            r[f'EKF position {m.group(1)} 20 m max'] = float(m.group(3)) * 100.0
        m = KSTART.search(line)
        if m:
            r['speed scale k at start'] = float(m.group(1))
        m = INSIDE.search(line)
        if m:
            r['EKF inside 2sigma %'] = float(m.group(1))
        m = KFIN.search(line)
        if m:
            r['speed scale error abs(k - true)'] = abs(float(m.group(1)) - float(m.group(2)))

groups = {}
for name, r in runs.items():
    g = re.sub(r'_\d{8}_\d{6}$', '', name)
    groups.setdefault(g, []).append(r)

rows = ['EKF position mean', 'EKF position p95', 'EKF position max',
        'EKF heading mean', 'EKF heading max',
        'AMCL position mean', 'AMCL position p95', 'AMCL position max',
        'EKF position first 20 m mean', 'EKF position first 20 m max',
        'EKF position after 20 m mean', 'EKF position after 20 m max',
        'EKF inside 2sigma %', 'speed scale k at start', 'speed scale error abs(k - true)']
units = {'position': 'cm', 'heading': 'deg'}

def cell(vals, d=2):
    if not vals:
        return '-'
    if len(vals) == 1:
        return f'{vals[0]:.{d}f} (1 run)'
    return f'{statistics.mean(vals):.{d}f} ± {statistics.stdev(vals):.{d}f}'

names = sorted(groups)
out = ['| | ' + ' | '.join(f'{g} (n={len(groups[g])})' for g in names) + ' |',
       '|---|' + '---|' * len(names)]
for row in rows:
    unit = next((u for k, u in units.items() if k in row), '')
    label = f'{row} [{unit}]' if unit else row
    out.append(f'| {label} | ' + ' | '.join(cell([r[row] for r in groups[g] if row in r], 4 if 'speed' in row else 2) for g in names) + ' |')
table = '\n'.join(out)
print(table)
with open(os.path.join(folder, 'aggregate_table.md'), 'w') as f:
    f.write(table + '\n')
print(f'\nWritten to {os.path.join(folder, "aggregate_table.md")}')

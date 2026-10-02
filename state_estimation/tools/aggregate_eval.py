"""Combine several evaluation runs into one thesis table (mean +- std across runs).

Reads every *_summary.txt that plot_eval.py wrote in a folder, groups the runs by
their run_name (the part before _YYYYMMDD_HHMMSS) and prints mean +- std per group.
Each run is counted once, even if it appears in several summary files.
Summaries whose FILE name starts with INVALID (runs renamed INVALID-<reason>_...)
are skipped -- the run name inside the file is unchanged, so the file name decides.

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
# anchored to the EKF line: the SE line also contains 'inside its +-2 sigma'
# (unanchored, the SE e_y value overwrote the EKF value -- fixed 2026-10-02)
INSIDE = re.compile(r'^EKF error inside its \+-2 sigma: (\d+) %')
WIN = re.compile(r'^EKF position (first|after) 20 m \[m\]\s+mean\s+([\d.]+)\s+rms\s+[\d.]+\s+p95\s+[\d.]+\s+max\s+([\d.]+)')
KSTART = re.compile(r'Speed scale k start ([\d.]+)')
KFIN = re.compile(r'Speed scale k final ([\d.]+) \(true ([\d.]+)\)')
# StateEstimate block (2026-10-02)
SE_ROW = re.compile(r'^SE\s+(.+?) \[([^\]]+)\]\s+mean\s+([\d.]+)\s+rms\s+[\d.]+\s+p95\s+([\d.]+)\s+max\s+([\d.]+)')
SE_IN = re.compile(r'e_y inside its \+-2 sigma: (\d+) %\s+e_psi inside its \+-2 sigma: (\d+) %')
SE_RATE = re.compile(r'^SE\s+output rate ([\d.]+) Hz')
SE_OK = re.compile(r'^SE\s+modes while driving: .*?OK ([\d.]+) %')

folder = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else '~/eval_logs')
runs = {}
for path in sorted(glob.glob(os.path.join(folder, '*_summary.txt'))):
    if os.path.basename(path).startswith('INVALID'):
        continue        # runs renamed INVALID-<reason>_... are kept on disk but never counted
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
        m = INSIDE.match(line)
        if m:
            r['EKF inside 2sigma %'] = float(m.group(1))
        m = KFIN.search(line)
        if m:
            r['speed scale error abs(k - true)'] = abs(float(m.group(1)) - float(m.group(2)))
        m = SE_ROW.match(line)
        if m:
            what, unit = m.group(1), m.group(2)
            scale = 100.0 if unit == 'm' else 1.0                # s, e_y in cm
            unit = 'cm' if unit == 'm' else unit
            for stat, val in zip(('mean', 'p95', 'max'), (m.group(3), m.group(4), m.group(5))):
                r[f'SE {what} {stat} [{unit}]'] = float(val) * scale
        m = SE_IN.search(line)
        if m:
            r['SE e_y inside 2sigma %'] = float(m.group(1))
            r['SE e_psi inside 2sigma %'] = float(m.group(2))
        m = SE_RATE.match(line)
        if m:
            r['SE output rate [Hz]'] = float(m.group(1))
        m = SE_OK.match(line)
        if m:
            r['SE mode OK while driving %'] = float(m.group(1))

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
# StateEstimate rows: only shown if at least one run has them
se_rows = [f'SE {q} error {st} [{u}]' for q, u in (('s', 'cm'), ('e_y', 'cm'), ('e_psi', 'deg'))
           for st in ('mean', 'p95', 'max')]
se_rows += ['SE e_psi error straights max [deg]', 'SE e_psi error corners max [deg]',
            'SE vx error p95 [m/s]', 'SE vy error p95 [m/s]', 'SE r error p95 [rad/s]',
            'SE e_y inside 2sigma %', 'SE e_psi inside 2sigma %',
            'SE latency stamp->publish mean [ms]', 'SE latency stamp->publish p95 [ms]',
            'SE latency stamp->publish max [ms]', 'SE age at subscriber p95 [ms]',
            'SE output rate [Hz]', 'SE mode OK while driving %']
rows += [r for r in se_rows if any(r in run for run in runs.values())]
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
    if row.startswith('SE '):
        label = row                                    # unit already in the name
    out.append(f'| {label} | ' + ' | '.join(cell([r[row] for r in groups[g] if row in r], 4 if 'speed' in row else 2) for g in names) + ' |')
table = '\n'.join(out)
print(table)
with open(os.path.join(folder, 'aggregate_table.md'), 'w') as f:
    f.write(table + '\n')
print(f'\nWritten to {os.path.join(folder, "aggregate_table.md")}')

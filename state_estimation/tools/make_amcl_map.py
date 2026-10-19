#!/usr/bin/env python3
"""make_amcl_map.py <map.yaml> <out.yaml> -- AMCL's THIN-WALL copy of a map, as a map FILE (2026-10-19).

The live system does this on the fly (node amcl_map_filter: /map -> /map_amcl, see amcl_map.py for WHY).
This tool writes the same map as a file: for the offline AMCL replays (runner/tools/replay_lib.sh), and to
LOOK at it once on the real car (after saving the slam_toolbox map: every wall must stay a closed line).

Pixels are classified exactly as nav2 map_server does (trinary): p = (255 - pixel) / 255 (negate 0),
occupied if p > occupied_thresh, free if p < free_thresh, else unknown.  A wall pixel stays occupied only
if it touches a free pixel (8 neighbours); the others become unknown (205).  Origin / resolution unchanged.
Writes <out>.png (0 / 205 / 254) + <out>.yaml.
"""
import argparse
import os
import sys

import numpy as np
from PIL import Image

try:
    from state_estimation.amcl_map import surface
except ImportError:                                  # from the source tree without the workspace
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'state_estimation'))
    from amcl_map import surface


def read_yaml(path):
    meta = {}
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.split('#', 1)[0]
            if ':' in line:
                k, v = line.split(':', 1)
                meta[k.strip()] = v.strip()
    return meta


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('map_yaml')
    ap.add_argument('out_yaml')
    a = ap.parse_args(argv)
    meta = read_yaml(a.map_yaml)
    img_path = meta['image'].strip('"\'')
    if not os.path.isabs(img_path):
        img_path = os.path.join(os.path.dirname(os.path.abspath(a.map_yaml)), img_path)
    img = np.array(Image.open(img_path).convert('L')).astype(np.float64)
    negate = int(float(meta.get('negate', 0)))
    occ_th, free_th = float(meta.get('occupied_thresh', 0.65)), float(meta.get('free_thresh', 0.196))
    p = img / 255.0 if negate else (255.0 - img) / 255.0
    occ, free = p > occ_th, p < free_th
    keep = surface(occ, free)
    out = np.full(img.shape, 205, dtype=np.uint8)
    out[free] = 254
    out[keep] = 0
    png = os.path.splitext(os.path.abspath(a.out_yaml))[0] + '.png'
    Image.fromarray(out).save(png)
    with open(a.out_yaml, 'w', encoding='utf-8') as f:
        f.write(f'image: {os.path.basename(png)}\nresolution: {meta["resolution"]}\norigin: {meta["origin"]}\n'
                'negate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.196\n')
    print(f'{a.map_yaml}: {int(occ.sum())} wall cells -> {int(keep.sum())} kept (the surface), '
          f'{int(occ.sum() - keep.sum())} now unknown; free {int(free.sum())}; wrote {a.out_yaml} + {os.path.basename(png)}')
    return 0


if __name__ == '__main__':
    sys.exit(main())

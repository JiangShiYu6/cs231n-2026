"""Run the unchanged notebook UNet/CFG references on a recorded CPU backend."""
import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cs231n.unet import Unet


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--diagnostic', action='store_true',
                        help='Compare CPU backends without treating collection as a passing test.')
    parser.add_argument('--output', type=Path, default=ROOT / 'ddpm_reference_report.json')
    args = parser.parse_args()
    notebook = json.loads((ROOT / 'DDPM.ipynb').read_text(encoding='utf-8'))
    results = []
    for threads in ((1, 2) if args.diagnostic else (2,)):
        torch.set_num_threads(threads)
        for mkldnn in ((True, False) if args.diagnostic else (True,)):
            torch.backends.mkldnn.enabled = mkldnn
            for index in (18, 31):
                def rel_error(actual, expected):
                    error = np.abs(actual - expected)
                    relative = float(np.max(error / np.maximum(1e-10, np.abs(actual) + np.abs(expected))))
                    results.append(dict(cell=index, threads=threads, mkldnn=mkldnn,
                                        relative_error=relative, absolute_error=float(error.max()),
                                        threshold=1e-6, passed=bool(relative < 1e-6),
                                        source_sha256=hashlib.sha256(source.encode('utf-8')).hexdigest()))
                    return relative

                namespace = dict(np=np, torch=torch, Unet=Unet, rel_error=rel_error)
                source = ''.join(notebook['cells'][index]['source'])
                exec(source, namespace)
    report = dict(platform=platform.platform(), python=platform.python_version(),
                  torch=torch.__version__, numpy=np.__version__, results=results)
    args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))
    failures = [result for result in results if not result['passed']]
    if args.diagnostic:
        print(f'DIAGNOSTIC ONLY: {len(failures)}/{len(results)} reference comparisons failed.')
    elif failures:
        raise SystemExit(f'FAILED: {len(failures)}/{len(results)} reference comparisons exceed 1e-6.')


if __name__ == '__main__':
    main()

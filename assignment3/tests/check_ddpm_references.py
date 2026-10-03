"""Run the unchanged notebook UNet/CFG references on a recorded CPU backend."""
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
    notebook = json.loads((ROOT / 'DDPM.ipynb').read_text(encoding='utf-8'))
    results = []
    for threads in (1, 2, 4, 8):
        torch.set_num_threads(threads)
        for mkldnn in (True, False):
            torch.backends.mkldnn.enabled = mkldnn
            for index in (18, 31):
                def rel_error(actual, expected):
                    error = np.abs(actual - expected)
                    relative = float(np.max(error / np.maximum(1e-10, np.abs(actual) + np.abs(expected))))
                    results.append(dict(cell=index, threads=threads, mkldnn=mkldnn,
                                        relative_error=relative, absolute_error=float(error.max())))
                    return relative

                namespace = dict(np=np, torch=torch, Unet=Unet, rel_error=rel_error)
                exec(''.join(notebook['cells'][index]['source']), namespace)
                if threads == 1 and mkldnn:
                    arrays = {name: value.detach().numpy() for name, value in namespace['unet'].state_dict().items()}
                    arrays.update({name: namespace[name].numpy() for name in ('inp_x', 'inp_text_emb', 'inp_t')})
                    np.savez(ROOT / f'ddpm_fixture_{index}.npz', **arrays)
    report = dict(platform=platform.platform(), torch=torch.__version__, results=results)
    (ROOT / 'ddpm_reference_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

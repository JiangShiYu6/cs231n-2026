"""Execute the convolution notebook locally and save outputs after each cell."""
import os
from pathlib import Path
import sys
import time

# Avoid excessive BLAS threading for the many small gradient-check operations.
for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "2"

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager

root = Path(__file__).resolve().parent
os.chdir(root)
os.environ["JUPYTER_RUNTIME_DIR"] = str(root / ".notebook-runtime")
Path(os.environ["JUPYTER_RUNTIME_DIR"]).mkdir(exist_ok=True)
path = root / "ConvolutionalNetworks.ipynb"
notebook = nbformat.read(path, as_version=4)
for cell in notebook.cells:
    if cell.cell_type == "code":
        cell.outputs = []
        cell.execution_count = None
manager = KernelManager(kernel_name="python3")
manager.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
client = NotebookClient(notebook, km=manager, timeout=7200, resources={"metadata": {"path": str(root)}})
try:
    with client.setup_kernel():
        for index, cell in enumerate(notebook.cells):
            if cell.cell_type != "code" or not cell.source.strip():
                continue
            print(f"START cell {index}: {cell.source.splitlines()[0]}", flush=True)
            started = time.perf_counter()
            client.execute_cell(cell, index)
            nbformat.write(notebook, path)
            for output in cell.get("outputs", []):
                if output.output_type == "stream":
                    print(output.text, end="", flush=True)
            print(f"DONE cell {index}: {time.perf_counter() - started:.1f}s", flush=True)
finally:
    nbformat.write(notebook, path)
    if manager.has_kernel:
        manager.shutdown_kernel(now=True)
    manager.cleanup_resources()
print("COMPLETE: all notebook cells executed and saved", flush=True)

"""Run Part V and its prerequisites, preserving notebook outputs."""
import os
from pathlib import Path
import sys

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager

root = Path(__file__).resolve().parent
os.chdir(root)
for name, folder in (("JUPYTER_RUNTIME_DIR", ".notebook-runtime"),
                     ("IPYTHONDIR", ".ipython-local")):
    os.environ[name] = str(root / folder)
    (root / folder).mkdir(exist_ok=True)
path = root / "PyTorch.ipynb"
nb = nbformat.read(path, as_version=4)

class LoggingClient(NotebookClient):
    def process_message(self, msg, cell, cell_index):
        if msg["msg_type"] == "stream":
            print(msg["content"]["text"], end="", flush=True)
        return super().process_message(msg, cell, cell_index)

manager = KernelManager(kernel_name="python3")
manager.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
client = LoggingClient(nb, km=manager, timeout=14400)
challenge = next(i for i, c in enumerate(nb.cells) if c.cell_type == "code" and "class CIFAR10ConvNet" in c.source)
accuracy = next(i for i, c in enumerate(nb.cells) if c.cell_type == "code" and "def check_accuracy_part34" in c.source)
try:
    with client.setup_kernel():
        for index in (1, 6, 9, accuracy, challenge):
            print(f"START cell {index}", flush=True)
            client.execute_cell(nb.cells[index], index)
            nbformat.write(nb, path)
            print(f"DONE cell {index}", flush=True)
        output = "".join(o.get("text", "") for o in nb.cells[challenge].outputs)
        if "70% target reached: True" in output:
            index = next(i for i in range(challenge + 1, len(nb.cells))
                         if nb.cells[i].cell_type == "code" and "check_accuracy_part34(loader_test" in nb.cells[i].source)
            print("Final test evaluation (once)", flush=True)
            client.execute_cell(nb.cells[index], index)
            nbformat.write(nb, path)
finally:
    nbformat.write(nb, path)
    if manager.has_kernel:
        manager.shutdown_kernel(now=True)
    manager.cleanup_resources()
print("COMPLETE", flush=True)

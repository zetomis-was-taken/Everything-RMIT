"""Environment probe: confirm GPU, attached inputs and model paths on Kaggle."""

from __future__ import annotations

import glob
import os
import subprocess

import torch
import transformers


def main() -> None:
    print("cwd:", os.getcwd())

    print("--- /kaggle/input ---")
    for path in sorted(glob.glob("/kaggle/input/*")):
        print(path)

    print("--- /kaggle/input/models ---")
    for path in sorted(glob.glob("/kaggle/input/models/*/*")):
        print(path)

    print("--- nvidia-smi ---")
    result = subprocess.run(["nvidia-smi"], capture_output=True, text=True, check=False)
    print(result.stdout[:1500])

    print("torch:", torch.__version__, "cuda:", torch.cuda.is_available())
    print("device_count:", torch.cuda.device_count())
    print("devices:", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
    print("transformers:", transformers.__version__)


if __name__ == "__main__":
    main()

"""The module provides an automated way to install FAISS, choosing between GPU and CPU."""

import logging
import subprocess
import sys

import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def install_faiss() -> None:
    """Install the appropriate version of FAISS depending on CUDA availability.

    Uses torch to detect CUDA and installs 'faiss-gpu' if available, otherwise installs 'faiss-cpu'.
    It uses subprocess.run to ensure that the installation commands are checked for success.
    """
    try:
        pip_executable: str = sys.executable.replace(
            "python",
            "pip",
        )  # More secure way to reference pip
        if torch.cuda.is_available():
            subprocess.run(args=[pip_executable, "install", "faiss-gpu>=1.7.1"], check=True)
            logging.info("Installed FAISS with GPU support.")
        else:
            subprocess.run(args=[pip_executable, "install", "faiss-cpu>=1.7.1"], check=True)
            logging.info("Installed FAISS for CPU.")
    except subprocess.CalledProcessError:
        logging.exception("Failed to install FAISS.")
        sys.exit(1)


if __name__ == "__main__":
    install_faiss()

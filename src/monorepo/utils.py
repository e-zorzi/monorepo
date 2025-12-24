import subprocess
from typing import Optional
import os
from colorama import Fore, init as colorama_init

colorama_init(autoreset=True)


# From https://modal.com/docs/examples/diffusers_lora_finetune
def exec_subprocess(cmd: list[str]):
    """Executes subprocess and prints log to terminal while subprocess is running."""
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    with process.stdout as pipe:
        for line in iter(pipe.readline, b""):
            line_str = line.decode()
            print(f"{line_str}", end="")

    exitcode = process.wait()
    if exitcode != 0:
        raise subprocess.CalledProcessError(exitcode, "\n".join(cmd))


def no_risky_api_key_is_being_used() -> bool | Optional[str]:
    environment_variables = os.environ
    for var in environment_variables:
        if (
            var.find("OPENAI") != -1
            or var.find("GEMINI") != -1
            or var.find("CEREBRAS") != -1
            or var.find("GROQ") != -1
            or var.find("BACKBLAZE") != -1
            or var.find("AWS") != -1
            or var.find("S3") != -1
        ):
            return (False, var)
    return (True, None)


def download_bare_repo_hf(repo_id, local_dir):
    from huggingface_hub import snapshot_download

    # Download and cache files
    # snapshot_download(repo_id)

    # Download and cache files + add symlinks in "my-folder"
    # snapshot_download(repo_id, local_dir="my-folder")

    # Duplicate files already existing in cache
    # and/or
    # Download missing files directly to "my-folder"
    #     => if ran multiple times, files are re-downloaded
    try:
        snapshot_download(
            repo_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            repo_type="dataset",
        )
    except Exception as e:
        print(
            Fore.YELLOW
            + "[WARN] I did encounter an error trying to download the repo. Trying again"
            + " with a different configuration."
            + Fore.WHITE
        )
        snapshot_download(
            repo_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            repo_type="model",
        )

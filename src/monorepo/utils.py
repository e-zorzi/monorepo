import subprocess
from typing import Optional
import os


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

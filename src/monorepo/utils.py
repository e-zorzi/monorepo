import subprocess


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

"""Standard-library hardware/runtime gate; importable without MLX installed."""

import platform
import subprocess
import sys


class UnsupportedEnvironment(RuntimeError):
    pass


def _machine_is_apple_silicon():
    if platform.system() != "Darwin":
        return False
    if platform.machine() == "arm64":
        return True
    try:
        translated = subprocess.check_output(
            ["sysctl", "-n", "sysctl.proc_translated"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        translated = ""
    return translated == "1"


def _chip_brand():
    try:
        return subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def reasons_unsupported():
    problems = []
    if platform.system() != "Darwin":
        problems.append(f"OS is {platform.system()}, not macOS.")
    if not _machine_is_apple_silicon():
        problems.append(
            f"CPU is '{_chip_brand()}' (arch {platform.machine()}); MLX requires "
            "Apple Silicon (M-series) with a Metal GPU. No Intel-Mac backend exists."
        )
    if sys.version_info < (3, 10):
        problems.append(
            f"Python is {sys.version.split()[0]}; MLX wheels require Python >= 3.10."
        )
    return problems


def preflight():
    problems = reasons_unsupported()
    if not problems:
        return
    sys.stderr.write("LongCache preflight failed; this machine cannot run MLX:\n")
    for p in problems:
        sys.stderr.write(f"  - {p}\n")
    sys.stderr.write(
        "\nRun on an Apple Silicon Mac (M1/M2/M3/M4), macOS 14+, Python 3.10+.\n"
    )
    raise SystemExit(2)

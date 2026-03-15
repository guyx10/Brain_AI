import subprocess
import json
import os

HUNTER_PATH = "../hunter_v27.1/hunter.py"


def run_hunter_recon(domain):

    cmd = [
        "python",
        HUNTER_PATH,
        "--domain",
        domain,
        "--mode",
        "recon"
    ]

    try:

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600
        )

        return result.stdout

    except Exception as e:

        return str(e)


def run_hunter_scan(domain):

    cmd = [
        "python",
        HUNTER_PATH,
        "--domain",
        domain,
        "--mode",
        "scan"
    ]

    try:

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=1200
        )

        return result.stdout

    except Exception as e:

        return str(e)


def run_hunter_endpoints(domain):

    cmd = [
        "python",
        HUNTER_PATH,
        "--domain",
        domain,
        "--mode",
        "endpoints"
    ]

    try:

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600
        )

        return result.stdout

    except Exception as e:

        return str(e)

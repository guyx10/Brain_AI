import subprocess

HUNTER_PATH = "...../hunter_v27.1.py"

def hunter_run(domain):

    cmd = [
        "python",
        HUNTER_PATH,
        "--domain",
        domain,
        "--mode",
        "auto"
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600
        )

        return result.stdout

    except Exception as e:
        return str(e)

def hunter_results(domain):

    target_dir = os.path.join(REPORT_DIR, domain)

    results = {}

    files = [
        "endpoints.json",
        "vulnerabilities.json",
        "technologies.json"
    ]

    for f in files:

        path = os.path.join(target_dir, f)

        if os.path.exists(path):

            try:
                with open(path) as fp:
                    results[f] = json.load(fp)
            except:
                results[f] = "error parsing"

    return json.dumps(results, indent=2)

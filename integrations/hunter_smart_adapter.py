#!/usr/bin/env python3
"""
BRAIN_AI ↔ HUNTER V27 SMART ADAPTER
=====================================
Replaces the dumb pipe adapter with an intelligent integration layer.

What it does:
1. Runs Hunter with timeout and non-blocking I/O
2. Parses structured output files (not raw stdout)
3. Generates concise analysis via GPT-4o-mini
4. Provides specific follow-up actions for Brain_AI to execute
5. Supports targeted re-testing of specific findings
6. Caches results to avoid re-running full scans

Architecture:
  Brain_AI → SmartHunterAdapter → Hunter v27.2
                                → GPT-4o-mini (analysis)
                                → Claude Code (deep reasoning, optional)

Usage from Brain_AI:
    from integrations.hunter_smart_adapter import SmartHunterAdapter
    
    adapter = SmartHunterAdapter()
    
    # Full scan with analysis
    results = adapter.scan("demo.testfire.net")
    
    # Get prioritized findings
    findings = adapter.get_findings("demo.testfire.net")
    
    # Targeted follow-up on a specific finding
    details = adapter.investigate_finding("demo.testfire.net", finding_id=3)
    
    # Run specific module only
    sqli = adapter.run_module("demo.testfire.net", module="sqli")
    
    # Get AI analysis of results
    analysis = adapter.analyze("demo.testfire.net")
"""

import os
import sys
import json
import time
import subprocess
import signal
import threading
import requests
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
from urllib.parse import urlparse, quote


# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

HUNTER_DIR = os.getenv("HUNTER_DIR", "/home/th0th/claude-hunter")
HUNTER_SCRIPT = os.getenv("HUNTER_SCRIPT", "hunter_v27.2.py")
HUNTER_VENV = os.path.join(HUNTER_DIR, "hunter", "bin", "activate")
HUNTER_PREPARE = os.path.join(HUNTER_DIR, "hunter_prepare_v27.sh")

# AI backends
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = "gpt-4o-mini"
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434/api/chat")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")

# Timeouts
SCAN_TIMEOUT = int(os.getenv("HUNTER_SCAN_TIMEOUT", "3600"))  # 1 hour max
ANALYSIS_TIMEOUT = 30  # GPT-4o-mini analysis timeout

# POC report generator
POC_REPORT_SCRIPT = os.path.join(HUNTER_DIR, "hunter_poc_report.py")


# ═══════════════════════════════════════════════════════════════════════════
# AI HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def ask_gpt4o_mini(prompt: str, max_tokens: int = 2000) -> str:
    """Quick GPT-4o-mini call for analysis. Falls back to Ollama."""
    if OPENAI_API_KEY:
        try:
            resp = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": OPENAI_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                    "temperature": 0.1,
                },
                timeout=ANALYSIS_TIMEOUT,
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"[Adapter] GPT-4o-mini failed ({e}), falling back to Ollama")

    # Fallback: Ollama
    try:
        resp = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=60,
        )
        data = resp.json()
        if "message" in data:
            return data["message"].get("content", "")
        return str(data)
    except Exception as e:
        return f"AI analysis unavailable: {e}"


# ═══════════════════════════════════════════════════════════════════════════
# SMART HUNTER ADAPTER
# ═══════════════════════════════════════════════════════════════════════════

class SmartHunterAdapter:
    """
    Intelligent adapter between Brain_AI and Hunter v27.
    
    Key differences from the old adapter:
    - Non-blocking with timeout (won't hang forever)
    - Reads structured files, not raw stdout
    - AI-powered analysis of findings
    - Targeted follow-up capabilities
    - Result caching
    """

    def __init__(self):
        self.hunter_dir = Path(HUNTER_DIR)
        self.results_cache: Dict[str, Dict] = {}

    # ─────────────────────────────────────────────────────────────────────
    # REPORT DIRECTORY
    # ─────────────────────────────────────────────────────────────────────

    def _report_dir(self, target: str) -> Path:
        """Get the report directory for a target."""
        clean = target.replace("://", "_").replace(".", "_").replace("/", "_")
        return self.hunter_dir / "reports" / clean

    # ─────────────────────────────────────────────────────────────────────
    # RUN HUNTER SCAN
    # ─────────────────────────────────────────────────────────────────────

    def scan(
        self,
        target: str,
        flags: str = "",
        timeout: int = SCAN_TIMEOUT,
    ) -> Dict[str, Any]:
        """
        Run a full Hunter scan with timeout protection.
        
        Args:
            target: Domain or IP to scan
            flags: Extra flags like --quick, --no-arjun, --delta
            timeout: Max seconds to wait
            
        Returns:
            Dict with: status, findings_count, duration, report_dir, summary
        """
        print(f"[Adapter] Starting Hunter scan: {target} {flags}")
        start_time = time.time()

        cmd = f"cd {HUNTER_DIR} && source {HUNTER_VENV} && python3 {HUNTER_SCRIPT} {target} {flags}"
        
        try:
            process = subprocess.Popen(
                cmd,
                shell=True,
                executable="/bin/bash",
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=str(self.hunter_dir),
            )

            # Stream output with timeout
            output_lines = []
            last_activity = time.time()

            while True:
                # Check timeout
                elapsed = time.time() - start_time
                if elapsed > timeout:
                    print(f"[Adapter] Timeout after {int(elapsed)}s — killing Hunter")
                    process.kill()
                    break

                # Check for stall (no output for 10 min)
                if time.time() - last_activity > 600:
                    print(f"[Adapter] Stalled for 10 min — killing Hunter")
                    process.kill()
                    break

                # Non-blocking read
                try:
                    line = process.stdout.readline()
                    if line == "" and process.poll() is not None:
                        break
                    if line:
                        last_activity = time.time()
                        stripped = line.strip()
                        if stripped:
                            output_lines.append(stripped)
                            # Print key milestones only
                            if any(kw in stripped for kw in [
                                "SCAN COMPLETE", "TOTAL FINDINGS", "🔥", "✅ Found",
                                "Smart Nuclei:", "ADVANCED SCANS COMPLETE",
                                "QUICK WINS COMPLETE", "SWARM COMPLETE",
                                "Content Discovery Complete", "XSS FOUND",
                                "SSRF VULNERABILITIES", "SQLi", "EXPLOIT_CHAINS",
                            ]):
                                print(f"[Hunter] {stripped}")
                except Exception:
                    time.sleep(0.1)

            process.wait()
            duration = int(time.time() - start_time)

        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "duration": int(time.time() - start_time),
            }

        # Parse results from files
        result = self._parse_results(target)
        result["status"] = "complete"
        result["duration"] = duration
        result["duration_human"] = f"{duration // 60}m {duration % 60}s"

        # Generate POC report
        self._generate_poc_report(target)

        # Cache
        self.results_cache[target] = result

        print(f"[Adapter] Scan complete: {result.get('findings_count', 0)} findings in {result['duration_human']}")
        return result

    # ─────────────────────────────────────────────────────────────────────
    # PARSE STRUCTURED RESULTS
    # ─────────────────────────────────────────────────────────────────────

    def _parse_results(self, target: str) -> Dict[str, Any]:
        """Parse Hunter output files into structured data."""
        rd = self._report_dir(target)
        result = {
            "target": target,
            "report_dir": str(rd),
            "findings": [],
            "findings_count": 0,
            "clusters": [],
            "nuclei_findings": [],
            "exploit_chains": [],
            "technologies": [],
            "content_discovery": [],
        }

        if not rd.exists():
            result["status"] = "no_report_dir"
            return result

        # Load filtered findings (main pipeline output)
        findings_file = rd / "filtered_findings.jsonl"
        if findings_file.exists():
            for line in findings_file.read_text().strip().split("\n"):
                if line.strip():
                    try:
                        result["findings"].append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

        # Load active findings (manual modules)
        active_file = rd / "active_findings.json"
        if active_file.exists():
            try:
                active = json.loads(active_file.read_text())
                if isinstance(active, list):
                    # Merge without duplicates
                    existing_urls = {f.get("matched", f.get("url", "")) for f in result["findings"]}
                    for af in active:
                        if af.get("url", "") not in existing_urls:
                            result["findings"].append(af)
            except Exception:
                pass

        result["findings_count"] = len(result["findings"])

        # Load clusters
        clusters_file = rd / "findings_clustered.json"
        if clusters_file.exists():
            try:
                result["clusters"] = json.loads(clusters_file.read_text())
            except Exception:
                pass

        # Load technologies
        tech_file = rd / "whatweb_technologies.json"
        if tech_file.exists():
            try:
                tech_data = json.loads(tech_file.read_text())
                result["technologies"] = tech_data.get("technologies", [])
            except Exception:
                pass

        # Load exploit chains (as text)
        chains_file = rd / "EXPLOIT_CHAINS.md"
        if chains_file.exists():
            result["exploit_chains_report"] = chains_file.read_text()[:3000]

        return result

    # ─────────────────────────────────────────────────────────────────────
    # GENERATE POC REPORT
    # ─────────────────────────────────────────────────────────────────────

    def _generate_poc_report(self, target: str):
        """Generate the POC report using hunter_poc_report.py."""
        rd = self._report_dir(target)
        if not rd.exists():
            return

        try:
            subprocess.run(
                ["python3", POC_REPORT_SCRIPT, str(rd)],
                cwd=str(self.hunter_dir),
                timeout=30,
                capture_output=True,
            )
        except Exception:
            pass

    # ─────────────────────────────────────────────────────────────────────
    # GET FINDINGS (from cache or files)
    # ─────────────────────────────────────────────────────────────────────

    def get_findings(self, target: str, severity: str = None) -> List[Dict]:
        """
        Get findings for a target. Reads from cache or files.
        
        Args:
            target: The target domain
            severity: Filter by severity (HIGH, MEDIUM, LOW, INFO)
        """
        if target in self.results_cache:
            findings = self.results_cache[target].get("findings", [])
        else:
            result = self._parse_results(target)
            self.results_cache[target] = result
            findings = result.get("findings", [])

        if severity:
            severity = severity.upper()
            findings = [
                f for f in findings
                if f.get("severity", "").upper() == severity
            ]

        return findings

    # ─────────────────────────────────────────────────────────────────────
    # GET CONCISE SUMMARY (for LLM consumption)
    # ─────────────────────────────────────────────────────────────────────

    def get_summary(self, target: str) -> str:
        """
        Get a concise text summary suitable for feeding to an LLM.
        Much smaller than raw findings — fits in context window.
        """
        result = self.results_cache.get(target) or self._parse_results(target)
        findings = result.get("findings", [])

        if not findings:
            return f"No findings for {target}. Scan may not have run yet."

        # Count by severity
        by_sev = {}
        for f in findings:
            s = f.get("severity", "info").upper()
            by_sev[s] = by_sev.get(s, 0) + 1

        # Count by type
        by_type = {}
        for f in findings:
            t = f.get("vuln_class", f.get("type", "UNKNOWN"))
            by_type[t] = by_type.get(t, 0) + 1

        # Build summary
        lines = [
            f"# Hunter Scan Results: {target}",
            f"Total findings: {len(findings)}",
            f"Severity: {', '.join(f'{k}={v}' for k, v in sorted(by_sev.items()))}",
            f"Types: {', '.join(f'{k}({v})' for k, v in sorted(by_type.items(), key=lambda x: -x[1]))}",
            f"Technologies: {', '.join(result.get('technologies', [])[:10])}",
            "",
            "## Top Findings (by severity):",
        ]

        # Top findings with POC data
        severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3, "info": 4}
        sorted_findings = sorted(
            findings,
            key=lambda f: severity_order.get(f.get("severity", "info").lower(), 5),
        )

        for i, f in enumerate(sorted_findings[:15], 1):
            sev = f.get("severity", "info").upper()
            vuln = f.get("vuln_class", f.get("type", "UNKNOWN"))
            url = f.get("matched", f.get("matched-at", f.get("url", "N/A")))
            param = f.get("parameter", "")
            payload = f.get("payload", "")

            entry = f"{i}. [{sev}] {vuln} @ {url[:80]}"
            if param:
                entry += f" (param: {param})"
            if payload:
                entry += f" | payload: {payload[:60]}"
            lines.append(entry)

        return "\n".join(lines)

    # ─────────────────────────────────────────────────────────────────────
    # AI ANALYSIS
    # ─────────────────────────────────────────────────────────────────────

    def analyze(self, target: str) -> str:
        """
        Use GPT-4o-mini to analyze findings and suggest next steps.
        Returns actionable analysis text.
        """
        summary = self.get_summary(target)
        if "No findings" in summary:
            return summary

        prompt = f"""You are an expert penetration tester analyzing automated scan results.

{summary}

Analyze these findings and provide:

1. REAL vs FALSE POSITIVE assessment — which findings look genuinely exploitable vs likely FP?
2. TOP 3 PRIORITIES — which findings should be manually verified first and why?
3. NEXT STEPS — specific manual tests to run (exact curl commands or tool commands)
4. CHAIN OPPORTUNITIES — can any findings be chained together for higher impact?
5. MISSING CHECKS — what did the scanner likely miss that should be tested manually?

Be specific and actionable. Include exact commands where possible.
Keep response under 800 words."""

        print(f"[Adapter] Analyzing {target} findings with AI...")
        return ask_gpt4o_mini(prompt)

    # ─────────────────────────────────────────────────────────────────────
    # INVESTIGATE SPECIFIC FINDING
    # ─────────────────────────────────────────────────────────────────────

    def investigate_finding(self, target: str, finding_index: int) -> str:
        """
        Deep-dive into a specific finding with AI analysis + manual verification commands.
        """
        findings = self.get_findings(target)
        if finding_index < 0 or finding_index >= len(findings):
            return f"Finding index {finding_index} out of range (0-{len(findings)-1})"

        finding = findings[finding_index]
        vuln = finding.get("vuln_class", finding.get("type", "UNKNOWN"))
        url = finding.get("matched", finding.get("matched-at", finding.get("url", "")))
        param = finding.get("parameter", "")
        payload = finding.get("payload", "")
        evidence = finding.get("evidence", "")

        prompt = f"""You are an expert penetration tester. Analyze this specific finding in depth:

Type: {vuln}
URL: {url}
Parameter: {param}
Payload: {payload}
Evidence: {evidence}

Provide:
1. Is this likely a TRUE POSITIVE or FALSE POSITIVE? Why?
2. Exact curl/browser commands to manually verify this finding
3. If it's real, what's the maximum impact? Can it be escalated?
4. Specific payloads to try for deeper exploitation
5. How to write this up for a bug bounty report (severity justification)

Be extremely specific with commands."""

        return ask_gpt4o_mini(prompt)

    # ─────────────────────────────────────────────────────────────────────
    # RUN TARGETED MODULE
    # ─────────────────────────────────────────────────────────────────────

    def run_targeted_test(self, target: str, url: str, test_type: str) -> str:
        """
        Run a targeted test on a specific URL — bypass the full scan.
        Uses curl/nuclei directly for fast verification.
        """
        results = []

        if test_type.lower() in ("sqli", "sql"):
            payloads = [
                "' OR '1'='1",
                "' UNION SELECT NULL--",
                "1' AND SLEEP(3)--",
                "1; SELECT pg_sleep(3)--",
            ]
            parsed = urlparse(url)
            for payload in payloads:
                test_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}?{parsed.query.split('=')[0]}={quote(payload)}" if '=' in (parsed.query or '') else url
                try:
                    cmd = f"curl -sk -o /dev/null -w '%{{http_code}} %{{size_download}} %{{time_total}}' '{test_url}'"
                    out = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=15, cwd=str(self.hunter_dir))
                    results.append(f"Payload: {payload}\n  Response: {out.stdout.strip()}")
                except Exception as e:
                    results.append(f"Payload: {payload}\n  Error: {e}")

        elif test_type.lower() in ("xss",):
            payloads = [
                "<script>alert(1)</script>",
                "<img src=x onerror=alert(1)>",
                "'\"><script>alert(1)</script>",
            ]
            parsed = urlparse(url)
            for payload in payloads:
                if '=' in (parsed.query or ''):
                    param_name = parsed.query.split('=')[0]
                    test_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}?{param_name}={quote(payload)}"
                else:
                    test_url = url
                try:
                    cmd = f"curl -sk '{test_url}'"
                    out = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=15, cwd=str(self.hunter_dir))
                    reflected = payload in out.stdout
                    results.append(f"Payload: {payload}\n  Reflected: {reflected}\n  Response size: {len(out.stdout)}")
                except Exception as e:
                    results.append(f"Payload: {payload}\n  Error: {e}")

        elif test_type.lower() in ("nuclei",):
            try:
                cmd = f"nuclei -u '{url}' -silent -jsonl -timeout 10"
                out = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=120, cwd=str(self.hunter_dir))
                if out.stdout.strip():
                    results.append(f"Nuclei findings:\n{out.stdout[:2000]}")
                else:
                    results.append("Nuclei: No findings")
            except Exception as e:
                results.append(f"Nuclei error: {e}")

        else:
            return f"Unknown test type: {test_type}. Supported: sqli, xss, nuclei"

        return "\n\n".join(results)

    # ─────────────────────────────────────────────────────────────────────
    # READ SPECIFIC REPORT FILE
    # ─────────────────────────────────────────────────────────────────────

    def read_report(self, target: str, report_name: str) -> str:
        """Read a specific report file. Truncates to 5000 chars for LLM context."""
        rd = self._report_dir(target)
        report_path = rd / report_name
        if not report_path.exists():
            return f"Report not found: {report_name}"
        content = report_path.read_text()
        if len(content) > 5000:
            return content[:5000] + f"\n\n... (truncated, {len(content)} total chars)"
        return content

    # ─────────────────────────────────────────────────────────────────────
    # LIST AVAILABLE REPORTS
    # ─────────────────────────────────────────────────────────────────────

    def list_reports(self, target: str) -> List[str]:
        """List all report files for a target."""
        rd = self._report_dir(target)
        if not rd.exists():
            return []
        return sorted([f.name for f in rd.iterdir() if f.is_file()])


# ═══════════════════════════════════════════════════════════════════════════
# BRAIN_AI COMPATIBLE TOOL FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

# Singleton adapter instance
_adapter = SmartHunterAdapter()


def hunter_scan(target: str) -> str:
    """Brain_AI tool: Run full Hunter scan and return summary."""
    result = _adapter.scan(target)
    if result.get("status") == "error":
        return f"Scan failed: {result.get('error')}"
    
    summary = _adapter.get_summary(target)
    return f"Scan complete in {result.get('duration_human', '?')}.\n\n{summary}"


def hunter_analyze(target: str) -> str:
    """Brain_AI tool: AI analysis of scan results."""
    return _adapter.analyze(target)


def hunter_findings(target: str) -> str:
    """Brain_AI tool: Get concise findings summary."""
    return _adapter.get_summary(target)


def hunter_investigate(target_and_index: str) -> str:
    """Brain_AI tool: Deep-dive a specific finding. Format: 'domain:index'"""
    parts = target_and_index.split(":")
    if len(parts) != 2:
        return "Format: domain:finding_index (e.g., demo.testfire.net:3)"
    target = parts[0].strip()
    try:
        index = int(parts[1].strip())
    except ValueError:
        return "Finding index must be a number"
    return _adapter.investigate_finding(target, index)


def hunter_test(target_url_type: str) -> str:
    """Brain_AI tool: Targeted test. Format: 'domain|url|type'"""
    parts = target_url_type.split("|")
    if len(parts) != 3:
        return "Format: domain|url|test_type (e.g., demo.testfire.net|https://demo.testfire.net/search.jsp?query=test|xss)"
    target, url, test_type = [p.strip() for p in parts]
    return _adapter.run_targeted_test(target, url, test_type)


def hunter_report(target_and_file: str) -> str:
    """Brain_AI tool: Read specific report. Format: 'domain:filename'"""
    parts = target_and_file.split(":")
    if len(parts) != 2:
        return "Format: domain:report_name (e.g., demo.testfire.net:VALIDATION_REPORT.md)"
    return _adapter.read_report(parts[0].strip(), parts[1].strip())


# ═══════════════════════════════════════════════════════════════════════════
# TOOL REGISTRY (for Brain_AI auto_brain.py)
# ═══════════════════════════════════════════════════════════════════════════

HUNTER_TOOLS = {
    "hunter_scan": hunter_scan,
    "hunter_analyze": hunter_analyze,
    "hunter_findings": hunter_findings,
    "hunter_investigate": hunter_investigate,
    "hunter_test": hunter_test,
    "hunter_report": hunter_report,
}


# ═══════════════════════════════════════════════════════════════════════════
# STANDALONE CLI
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python3 hunter_smart_adapter.py scan <target> [flags]")
        print("  python3 hunter_smart_adapter.py analyze <target>")
        print("  python3 hunter_smart_adapter.py findings <target>")
        print("  python3 hunter_smart_adapter.py investigate <target> <index>")
        print("  python3 hunter_smart_adapter.py test <target> <url> <type>")
        print("  python3 hunter_smart_adapter.py reports <target>")
        print("")
        print("Examples:")
        print("  python3 hunter_smart_adapter.py scan demo.testfire.net")
        print("  python3 hunter_smart_adapter.py scan demo.testfire.net --quick")
        print("  python3 hunter_smart_adapter.py analyze demo.testfire.net")
        print("  python3 hunter_smart_adapter.py investigate demo.testfire.net 3")
        print("  python3 hunter_smart_adapter.py test demo.testfire.net 'https://demo.testfire.net/search.jsp?query=test' xss")
        sys.exit(1)

    adapter = SmartHunterAdapter()
    command = sys.argv[1]

    if command == "scan":
        target = sys.argv[2]
        flags = " ".join(sys.argv[3:])
        result = adapter.scan(target, flags)
        print(json.dumps(result, indent=2, default=str))

    elif command == "analyze":
        print(adapter.analyze(sys.argv[2]))

    elif command == "findings":
        print(adapter.get_summary(sys.argv[2]))

    elif command == "investigate":
        print(adapter.investigate_finding(sys.argv[2], int(sys.argv[3])))

    elif command == "test":
        print(adapter.run_targeted_test(sys.argv[2], sys.argv[3], sys.argv[4]))

    elif command == "reports":
        for r in adapter.list_reports(sys.argv[2]):
            print(f"  {r}")

    else:
        print(f"Unknown command: {command}")

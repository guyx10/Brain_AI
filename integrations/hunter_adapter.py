# integrations/hunter_adapter.py
import subprocess
import os
import sys
import time
from pathlib import Path

# Fix the path - it should be the directory, not the file
HUNTER_PATH = "/home/th0th/claude-hunter"  # Directory containing hunter_v27.1.py
HUNTER_SCRIPT = os.path.join(HUNTER_PATH, "hunter_v27.1.py")

def run_hunter_scan(target):
    """Run full Hunter v27 scan on target and return results"""
    print(f"[Hunter] Starting full scan on {target}")
    print(f"[Hunter] This may take several minutes...")
    
    try:
        # Make sure the file exists
        if not os.path.exists(HUNTER_SCRIPT):
            return f"Error: Hunter script not found at {HUNTER_SCRIPT}"
        
        # Run the process and stream output in real-time
        process = subprocess.Popen(
            ['python3', HUNTER_SCRIPT, target],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=HUNTER_PATH,
            bufsize=1,  # Line buffered
            universal_newlines=True
        )
        
        # Collect output
        stdout_lines = []
        stderr_lines = []
        
        # Read output in real-time (optional - you can see progress)
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(f"[Hunter] {output.strip()}")  # Optional: show progress
                stdout_lines.append(output)
        
        # Get any remaining output
        stdout, stderr = process.communicate()
        if stdout:
            stdout_lines.append(stdout)
        if stderr:
            stderr_lines.append(stderr)
            print(f"[Hunter] Error: {stderr}")
        
        # Wait for process to complete
        return_code = process.wait()
        
        full_stdout = ''.join(stdout_lines)
        full_stderr = ''.join(stderr_lines)
        
        if return_code == 0:
            print(f"[Hunter] Full scan completed successfully")
            
            # Wait a moment for files to be written
            time.sleep(2)
            
            # Check for generated reports in the Hunter directory
            reports = []
            report_files = [
                'FINAL_REPORT.md',
                'NUCLEI_SMART_REPORT.md',
                'CVE_MATCH_REPORT.md',
                'JS_ANALYSIS_REPORT.md',
                'SKEPTIC_REPORT.md',
                'DELTA_REPORT.md',
                'VALIDATION_REPORT.md',
                'H1_REPORT.md'
            ]
            
            # Also check in reports subdirectory
            reports_dir = os.path.join(HUNTER_PATH, 'reports', target.replace('.', '_'))
            if os.path.exists(reports_dir):
                for report in report_files:
                    report_path = os.path.join(reports_dir, report)
                    if os.path.exists(report_path):
                        try:
                            with open(report_path, 'r') as f:
                                reports.append(f"--- {report} ---\n{f.read()}")
                        except Exception as e:
                            reports.append(f"--- {report} ---\nError reading: {e}")
            
            # Check root directory too
            for report in report_files:
                report_path = os.path.join(HUNTER_PATH, report)
                if os.path.exists(report_path):
                    try:
                        with open(report_path, 'r') as f:
                            reports.append(f"--- {report} (root) ---\n{f.read()}")
                    except Exception as e:
                        pass
            
            if reports:
                return "\n\n".join(reports)
            else:
                # If no reports found, return the stdout
                return f"Scan completed but no reports found. Output:\n{full_stdout}"
        else:
            return f"Error (code {return_code}): {full_stderr or full_stdout}"
            
    except Exception as e:
        return f"[Hunter] Error running scan: {str(e)}"

def run_hunter_recon(target):
    """Alias for run_hunter_scan for now"""
    print(f"[Hunter] Running recon on {target} (using full scan)")
    return run_hunter_scan(target)

def run_hunter_endpoints(target):
    """Alias for run_hunter_scan for now"""
    print(f"[Hunter] Discovering endpoints for {target} (using full scan)")
    return run_hunter_scan(target)
    
# Temporary test function - add this to hunter_adapter.py temporarily
def test_hunter_scan(target):
    """Test if Hunter scan starts properly"""
    print(f"[Test] Testing Hunter scan on {target}")
    
    process = subprocess.Popen(
        ['python3', HUNTER_SCRIPT, target],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=HUNTER_PATH
    )
    
    # Wait 10 seconds and check if it's still running
    time.sleep(10)
    if process.poll() is None:
        print("[Test] Process is still running after 10 seconds - good!")
        process.terminate()
        return "Process started successfully"
    else:
        stdout, stderr = process.communicate()
        return f"Process ended early: {stderr or stdout}"

#!/usr/bin/env python3
"""Development Runner Script for Agentic RAG Voice Assistant.

1. Checks virtual environment status (.venv).
2. Verifies LiveKit Signaling Server is actively listening on port 7880.
3. Runs pre-flight integration tests (pytest).
4. Concurrently boots FastAPI backend, LiveKit Agent Worker, and Next.js frontend.
5. Handles graceful shutdown on Ctrl+C.
"""
import os
import sys
import time
import socket
import subprocess
import signal

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Terminal ANSI color codes for readable output
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BLUE = "\033[94m"
RESET = "\033[0m"


def check_virtual_env():
    """Verifies that the script is running inside a Python virtual environment."""
    is_venv = (
        sys.prefix != sys.base_prefix
        or "VIRTUAL_ENV" in os.environ
    )
    
    if not is_venv:
        print(f"{RED}[ERROR] Virtual environment (.venv) is not active!{RESET}")
        print(f"{YELLOW}Please activate your virtual environment before running this script:{RESET}")
        print("  source .venv/bin/activate      # Linux/macOS")
        print("  .venv\\Scripts\\activate       # Windows")
        sys.exit(1)
        
    venv_path = os.environ.get("VIRTUAL_ENV", sys.prefix)
    print(f"{GREEN}[OK] Virtual Environment Active:{RESET} {venv_path}")


def check_livekit_port(host="127.0.0.1", port=7880):
    """Verifies if the LiveKit server binary is actively listening on port 7880."""
    print(f"\n{BLUE}[Checking] Verifying LiveKit Signaling Server at {host}:{port}...{RESET}")
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(2.0)
    try:
        result = sock.connect_ex((host, port))
        if result == 0:
            print(f"{GREEN}[OK] LiveKit Server Detected on port {port}!{RESET}\n")
            return True
        else:
            print(f"\n{RED}*******************************************************************{RESET}")
            print(f"{RED} [FATAL ERROR] LiveKit Server is NOT running on {host}:{port}!{RESET}")
            print(f"{RED} WebRTC voice streaming requires the local LiveKit server binary.  {RESET}")
            print(f"{YELLOW} Please start LiveKit server in a separate terminal window first:   {RESET}")
            print(f"{GREEN}   ./livekit-server --dev  (or ./bin/livekit-server.exe --dev)      {RESET}")
            print(f"{RED}*******************************************************************{RESET}\n")
            sys.exit(1)
    except Exception as e:
        print(f"{RED}[ERROR] Socket connection check failed: {e}{RESET}")
        sys.exit(1)
    finally:
        sock.close()


def run_preflight_tests():
    """Executes integration tests. Halts execution if any test fails."""
    print(f"\n{BLUE}=========================================={RESET}")
    print(f"{BLUE} Running Pre-Flight Integration Tests     {RESET}")
    print(f"{BLUE}=========================================={RESET}")
    
    # Check for test file location (tests/test_phase5.py or test_phase5.py)
    test_path = "tests/test_phase5.py" if os.path.exists("tests/test_phase5.py") else "test_phase5.py"
    
    if not os.path.exists(test_path):
        print(f"{YELLOW}[WARN] Test file '{test_path}' not found. Skipping test execution.{RESET}")
        return
        
    cmd = [sys.executable, "-m", "pytest", test_path, "-v"]
    result = subprocess.run(cmd)
    
    if result.returncode != 0:
        print(f"\n{RED}***************************************************{RESET}")
        print(f"{RED} [GUARDRAIL TRIGGERED] Pre-Flight Tests Failed!     {RESET}")
        print(f"{RED} Server boot aborted to prevent running bad code.   {RESET}")
        print(f"{RED} Fix failing tests and run start_dev.py again.      {RESET}")
        print(f"{RED}***************************************************{RESET}")
        sys.exit(result.returncode)
        
    print(f"\n{GREEN}[OK] All Phase 5 Integration Tests Passed!{RESET}\n")


def boot_servers():
    """Concurrently boots FastAPI backend, LiveKit Agent Worker, and Next.js frontend."""
    processes = []
    is_windows = sys.platform == "win32"
    project_root = os.path.abspath(os.path.dirname(__file__))
    env = os.environ.copy()
    env["PYTHONPATH"] = project_root + (os.pathsep + env["PYTHONPATH"] if "PYTHONPATH" in env else "")
    
    try:
        # 1. FastAPI Backend (Port 8000)
        print(f"{BLUE}[1/3] Starting FastAPI Backend (uvicorn)...{RESET}")
        backend_cmd = [sys.executable, "-m", "uvicorn", "app.main:app", "--reload", "--port", "8000"]
        processes.append(("FastAPI Backend", subprocess.Popen(backend_cmd, env=env)))

        # 2. LiveKit Python Agent Worker
        print(f"{BLUE}[2/3] Starting LiveKit Agent Worker...{RESET}")
        worker_cmd = [sys.executable, "app/worker/livekit_worker.py", "dev"]
        processes.append(("LiveKit Agent Worker", subprocess.Popen(worker_cmd, env=env)))

        # 3. Next.js Frontend (Port 3000)
        frontend_dir = os.path.abspath("./frontend")
        if os.path.exists(frontend_dir):
            print(f"{BLUE}[3/3] Starting Next.js Frontend (npm run dev)...{RESET}")
            npm_bin = "npm.cmd" if is_windows else "npm"
            processes.append(("Next.js Frontend", subprocess.Popen(
                [npm_bin, "run", "dev"],
                cwd=frontend_dir,
                shell=is_windows
            )))

        print(f"\n{GREEN}=============================================={RESET}")
        print(f"{GREEN} All 3 Voice Services Live & Running!          {RESET}")
        print(f"   • LiveKit Server: {BLUE}ws://127.0.0.1:7880{RESET}")
        print(f"   • FastAPI Backend:{BLUE}http://localhost:8000{RESET}")
        print(f"   • Next.js UI:     {BLUE}http://localhost:3000{RESET}")
        print(f"{GREEN} Press Ctrl+C to gracefully stop all services. {RESET}")
        print(f"{GREEN}=============================================={RESET}\n")

        while True:
            time.sleep(1)
            for name, proc in processes:
                if proc.poll() is not None:
                    print(f"{RED}[ALERT] {name} exited unexpectedly with code {proc.returncode}!{RESET}")
                    raise KeyboardInterrupt

    except KeyboardInterrupt:
        print(f"\n\n{YELLOW}[Shutdown] Stopping all child processes...{RESET}")
        for name, proc in processes:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
        print(f"{GREEN}[OK] All services stopped successfully.{RESET}")
        sys.exit(0)


if __name__ == "__main__":
    check_virtual_env()
    check_livekit_port()  # Halts early if 7880 is not listening
    run_preflight_tests()
    boot_servers()

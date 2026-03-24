"""Start the MIST.AI development stack.

Usage:
    python scripts/start_dev.py              # Start full stack (Neo4j, Ollama, backend, frontend)
    python scripts/start_dev.py --deps-only  # Start dependencies only (Neo4j, Ollama)
    python scripts/start_dev.py --stop       # Stop services

Prerequisites:
    - Docker Desktop (for Neo4j)
    - Ollama installed
    - Python venv with dependencies
    - Flutter SDK on PATH
"""

import argparse
import socket
import subprocess
import sys
import time


def check_port(host: str, port: int, timeout: float = 2.0) -> bool:
    """Check if a TCP port is reachable."""
    try:
        sock = socket.create_connection((host, port), timeout=timeout)
        sock.close()
        return True
    except (OSError, ConnectionRefusedError):
        return False


def start_ollama() -> bool:
    """Start Ollama serve in the background."""
    if check_port("localhost", 11434):
        print("  [OK] Ollama already running on :11434")
        return True

    print("  Starting Ollama...")
    try:
        subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
        )
    except FileNotFoundError:
        print("  [FAIL] Ollama not found. Install from https://ollama.com")
        return False

    # Wait for it to become ready
    for i in range(15):
        time.sleep(1)
        if check_port("localhost", 11434):
            print("  [OK] Ollama started on :11434")
            return True
        if i % 5 == 4:
            print(f"  Waiting... ({i + 1}s)")

    print("  [FAIL] Ollama did not start in 15s")
    return False


def check_model(model: str = "qwen2.5:7b-instruct") -> bool:
    """Check if the required model is available, pull if not."""
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if model in result.stdout:
            print(f"  [OK] Model {model} available")
            return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False

    print(f"  Pulling {model} (first time only, may take a few minutes)...")
    try:
        subprocess.run(["ollama", "pull", model], timeout=600)
        print(f"  [OK] Model {model} pulled")
        return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        print(f"  [FAIL] Could not pull {model}")
        return False


def start_neo4j() -> bool:
    """Start Neo4j via Docker if not already running."""
    if check_port("localhost", 7687):
        print("  [OK] Neo4j already running on bolt://localhost:7687")
        return True

    print("  Starting Neo4j via Docker...")
    try:
        result = subprocess.run(
            ["docker", "compose", "up", "-d", "neo4j"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            print(f"  [FAIL] docker compose failed: {result.stderr.strip()}")
            print("         Is Docker Desktop running?")
            return False
    except FileNotFoundError:
        print("  [FAIL] Docker not found. Install Docker Desktop.")
        return False
    except subprocess.TimeoutExpired:
        print("  [FAIL] Docker compose timed out.")
        return False

    # Wait for Neo4j to become ready
    for i in range(30):
        time.sleep(1)
        if check_port("localhost", 7687):
            print("  [OK] Neo4j started on bolt://localhost:7687")
            return True
        if i % 10 == 9:
            print(f"  Waiting for Neo4j... ({i + 1}s)")

    print("  [FAIL] Neo4j did not start in 30s")
    return False


def start_backend() -> subprocess.Popen | None:
    """Start the backend server."""
    if check_port("localhost", 8001):
        print("  [OK] Backend already running on :8001")
        return None

    print("  Starting backend server...")
    # Resolve venv python relative to project root (parent of scripts/)
    from pathlib import Path

    project_root = Path(__file__).resolve().parent.parent
    if sys.platform == "win32":
        venv_python = str(project_root / "venv" / "Scripts" / "python.exe")
    else:
        venv_python = str(project_root / "venv" / "bin" / "python")
    print(f"  Using venv: {venv_python}")
    try:
        proc = subprocess.Popen(
            [venv_python, "backend/server.py"],
            cwd=str(project_root),
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0,
        )
    except FileNotFoundError:
        print(f"  [FAIL] {venv_python} not found. Create venv first.")
        return None

    for i in range(90):
        time.sleep(1)
        if check_port("localhost", 8001):
            print("  [OK] Backend started on ws://localhost:8001")
            return proc
        if i % 10 == 9:
            print(f"  Waiting for backend (loading TTS models)... ({i + 1}s)")

    print("  [FAIL] Backend did not start in 90s")
    return None


def start_frontend() -> subprocess.Popen | None:
    """Start the Flutter frontend."""
    import shutil
    from pathlib import Path

    print("  Launching Flutter frontend...")
    flutter_cmd = shutil.which("flutter") or r"C:\Users\rajga\flutter\bin\flutter"
    project_root = Path(__file__).resolve().parent.parent
    try:
        proc = subprocess.Popen(
            [flutter_cmd, "run", "-d", "windows"],
            cwd=str(project_root / "mist_desktop"),
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0,
        )
        print("  [OK] Flutter frontend launching (mist_desktop)")
        return proc
    except FileNotFoundError:
        print("  [FAIL] flutter not found. Is Flutter SDK on PATH?")
        return None


def stop() -> None:
    """Stop Neo4j (Docker) and Ollama."""
    print("Stopping stack...")
    subprocess.run(["docker", "compose", "down"], capture_output=True)
    print("  Neo4j stopped.")
    if sys.platform == "win32":
        subprocess.run(
            ["taskkill", "/f", "/im", "ollama.exe"],
            capture_output=True,
        )
        subprocess.run(
            ["taskkill", "/f", "/im", "ollama_llama_server.exe"],
            capture_output=True,
        )
    else:
        subprocess.run(["pkill", "-f", "ollama serve"], capture_output=True)
    print("  Ollama stopped.")
    print("Stack stopped.")


def main() -> None:
    parser = argparse.ArgumentParser(description="MIST.AI dev stack manager")
    parser.add_argument("--stop", action="store_true", help="Stop services")
    parser.add_argument(
        "--deps-only",
        action="store_true",
        help="Only start dependencies (Neo4j, Ollama), not backend/frontend",
    )
    args = parser.parse_args()

    if args.stop:
        stop()
        return

    print("=" * 50)
    print("  MIST.AI Development Stack")
    print("=" * 50)
    print()

    # 1. Neo4j
    print("[1/5] Starting Neo4j...")
    neo4j_ok = start_neo4j()

    # 2. Ollama
    print("[2/5] Starting Ollama...")
    ollama_ok = start_ollama()

    # 3. Model
    if ollama_ok:
        print("[3/5] Checking model...")
        model_ok = check_model()
    else:
        model_ok = False

    if not (neo4j_ok and ollama_ok and model_ok):
        print()
        print("=" * 50)
        print("  Stack NOT ready. Fix the issues above.")
        print("=" * 50)
        sys.exit(1)

    if args.deps_only:
        print()
        print("=" * 50)
        print("  Dependencies ready.")
        print()
        print("  Neo4j:   bolt://localhost:7687")
        print("  Ollama:  http://localhost:11434")
        print()
        print("  Start backend:  venv\\Scripts\\python backend\\server.py")
        print("  Start frontend: cd mist_desktop && flutter run -d windows")
        print("=" * 50)
        return

    # 4. Backend
    print("[4/5] Starting backend...")
    backend_proc = start_backend()
    backend_ok = backend_proc is not None or check_port("localhost", 8001)

    # 5. Frontend
    print("[5/5] Starting frontend...")
    frontend_proc = start_frontend()
    frontend_ok = frontend_proc is not None

    print()
    print("=" * 50)
    print("  Stack running." if backend_ok and frontend_ok else "  Stack partially running.")
    print()
    print("  Neo4j:    bolt://localhost:7687")
    print("  Ollama:   http://localhost:11434")
    print("  Backend:  ws://localhost:8001" if backend_ok else "  Backend:  [NOT RUNNING]")
    print("  Frontend: Flutter desktop" if frontend_ok else "  Frontend: [NOT RUNNING]")
    print()
    print("  Stop all: python scripts/start_dev.py --stop")
    print("  Run tests: venv\\Scripts\\python -m pytest tests/ -v")
    print("=" * 50)

    # Keep script alive while subprocesses run, clean up everything on exit
    procs = [p for p in (backend_proc, frontend_proc) if p is not None]
    if procs:
        import atexit
        import contextlib

        def cleanup():
            print()
            print("Shutting down stack...")
            for p in procs:
                with contextlib.suppress(OSError):
                    p.terminate()
            for p in procs:
                with contextlib.suppress(Exception):
                    p.wait(timeout=5)
            stop()
            print("Stack stopped.")

        # Ensure cleanup runs on normal exit, Ctrl+C, and terminal close
        atexit.register(cleanup)
        if sys.platform == "win32":
            # CTRL_CLOSE_EVENT fires when terminal window is closed
            import ctypes

            kernel32 = ctypes.windll.kernel32

            def console_handler(event):
                if event in (0, 2):  # CTRL_C_EVENT, CTRL_CLOSE_EVENT
                    cleanup()
                    return True
                return False

            handler_func = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_ulong)(console_handler)
            kernel32.SetConsoleCtrlHandler(handler_func, True)

        try:
            while all(p.poll() is None for p in procs):
                time.sleep(1)
        except KeyboardInterrupt:
            pass  # cleanup runs via atexit


if __name__ == "__main__":
    main()

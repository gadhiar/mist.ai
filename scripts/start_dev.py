"""Start the MIST.AI development stack via Docker Compose.

Usage:
    python scripts/start_dev.py              # Start full stack + Flutter frontend
    python scripts/start_dev.py --deps-only  # Start backend stack only (no Flutter)
    python scripts/start_dev.py --stop       # Stop all services
    python scripts/start_dev.py --logs       # Tail backend logs
    python scripts/start_dev.py --restart    # Restart backend container (pick up code changes)

Prerequisites:
    - Docker Desktop with NVIDIA Container Toolkit
    - Flutter SDK on PATH (for frontend)
"""

import argparse
import contextlib
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


def check_docker() -> bool:
    """Verify Docker daemon is running."""
    try:
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            print("  [FAIL] Docker daemon not running. Start Docker Desktop.")
            return False
        print("  [OK] Docker daemon running")
        return True
    except FileNotFoundError:
        print("  [FAIL] Docker not found. Install Docker Desktop.")
        return False
    except subprocess.TimeoutExpired:
        print("  [FAIL] Docker info timed out.")
        return False


def start_stack(build: bool = False) -> bool:
    """Start the Docker Compose stack (backend + Neo4j + Ollama)."""
    print("  Starting Docker Compose stack...")
    cmd = ["docker", "compose", "up", "-d"]
    if build:
        cmd.append("--build")
    try:
        result = subprocess.run(
            cmd,
            timeout=600,
        )
        if result.returncode != 0:
            print("  [FAIL] docker compose up failed")
            return False
        print("  [OK] Compose stack started")
        return True
    except subprocess.TimeoutExpired:
        print("  [FAIL] docker compose up timed out (10min)")
        return False


def wait_for_service(name: str, host: str, port: int, max_wait: int = 120) -> bool:
    """Wait for a service to become reachable on a port."""
    for i in range(max_wait):
        if check_port(host, port):
            print(f"  [OK] {name} ready on :{port}")
            return True
        if i % 15 == 14:
            print(f"  Waiting for {name}... ({i + 1}s)")
        time.sleep(1)

    print(f"  [FAIL] {name} did not start in {max_wait}s")
    return False


def wait_for_container_healthy(
    container: str,
    max_wait: int = 180,
    quiet_interval: int = 15,
) -> bool:
    """Wait for a Docker container to report healthy status.

    Unlike port checks, this waits for the container's own healthcheck
    to pass, ensuring models are fully loaded before returning.
    """
    for i in range(max_wait):
        try:
            result = subprocess.run(
                ["docker", "inspect", container, "--format", "{{.State.Health.Status}}"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            status = result.stdout.strip()
            if status == "healthy":
                print(f"  [OK] {container} healthy")
                return True
            if status not in ("starting", "healthy"):
                print(f"  [FAIL] {container} status: {status}")
                return False
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        if i % quiet_interval == quiet_interval - 1:
            print(f"  Waiting for {container}... ({i + 1}s)")
        time.sleep(1)

    print(f"  [FAIL] {container} not healthy after {max_wait}s")
    return False


def pull_model(model: str = "qwen2.5:7b-instruct") -> bool:
    """Ensure the LLM model is available in Ollama."""
    print(f"  Checking model {model}...")
    try:
        result = subprocess.run(
            ["docker", "compose", "exec", "mist-ollama", "ollama", "list"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if model in result.stdout:
            print(f"  [OK] Model {model} available")
            return True
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    print(f"  Pulling {model} (first time only, may take several minutes)...")
    try:
        result = subprocess.run(
            ["docker", "compose", "exec", "mist-ollama", "ollama", "pull", model],
            timeout=600,
        )
        if result.returncode == 0:
            print(f"  [OK] Model {model} pulled")
            return True
        print(f"  [FAIL] Could not pull {model}")
        return False
    except subprocess.TimeoutExpired:
        print("  [FAIL] Model pull timed out")
        return False


def start_frontend() -> subprocess.Popen | None:
    """Start the Flutter frontend (native Windows)."""
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


def tail_logs() -> None:
    """Tail backend container logs."""
    with contextlib.suppress(KeyboardInterrupt):
        subprocess.run(["docker", "compose", "logs", "-f", "mist-backend"])


def stop() -> None:
    """Stop the Docker Compose stack."""
    print("Stopping stack...")
    subprocess.run(["docker", "compose", "down"], capture_output=True)
    print("  Stack stopped.")


def restart_backend() -> None:
    """Restart just the backend container (pick up code changes)."""
    print("Restarting backend...")
    subprocess.run(["docker", "compose", "restart", "mist-backend"])
    print("  Backend restarted.")


def main() -> None:
    parser = argparse.ArgumentParser(description="MIST.AI dev stack manager (Docker)")
    parser.add_argument("--stop", action="store_true", help="Stop all services")
    parser.add_argument("--logs", action="store_true", help="Tail backend logs")
    parser.add_argument("--restart", action="store_true", help="Restart backend container")
    parser.add_argument("--build", action="store_true", help="Rebuild images before starting")
    parser.add_argument(
        "--deps-only",
        action="store_true",
        help="Start backend stack only, no Flutter frontend",
    )
    args = parser.parse_args()

    if args.stop:
        stop()
        return

    if args.logs:
        tail_logs()
        return

    if args.restart:
        restart_backend()
        return

    print("=" * 50)
    print("  MIST.AI Development Stack (Docker)")
    print("=" * 50)
    print()

    # 1. Docker
    print("[1/5] Checking Docker...")
    if not check_docker():
        sys.exit(1)

    # 2. Start stack
    print("[2/5] Starting services...")
    if not start_stack(build=args.build):
        sys.exit(1)

    # 3. Wait for services
    #    Neo4j and Ollama: port checks (fast, no model loading).
    #    Backend: wait for Docker healthcheck (models take ~90s to load).
    print("[3/5] Waiting for services...")
    neo4j_ok = wait_for_service("Neo4j", "localhost", 7687, max_wait=60)
    ollama_ok = wait_for_service("Ollama", "localhost", 11434, max_wait=30)
    print("  Backend loading models (Whisper + Chatterbox + LLM)...")
    backend_ok = wait_for_container_healthy("mist-backend", max_wait=180)

    # 4. Model
    if ollama_ok:
        print("[4/5] Checking LLM model...")
        pull_model()

    if not (neo4j_ok and ollama_ok and backend_ok):
        print()
        print("=" * 50)
        print("  Stack NOT fully ready. Check logs:")
        print("  docker compose logs mist-backend")
        print("=" * 50)
        sys.exit(1)

    if args.deps_only:
        print()
        print("=" * 50)
        print("  Backend stack ready.")
        print()
        print("  Neo4j:    bolt://localhost:7687 (browser: http://localhost:7474)")
        print("  Ollama:   http://localhost:11434")
        print("  Backend:  ws://localhost:8001/ws")
        print()
        print("  Logs:     python scripts/start_dev.py --logs")
        print("  Restart:  python scripts/start_dev.py --restart")
        print("  Stop:     python scripts/start_dev.py --stop")
        print("=" * 50)
        return

    # 5. Frontend
    print("[5/5] Starting frontend...")
    frontend_proc = start_frontend()
    frontend_ok = frontend_proc is not None

    print()
    print("=" * 50)
    print("  Stack running." if frontend_ok else "  Backend running (frontend failed).")
    print()
    print("  Neo4j:    bolt://localhost:7687")
    print("  Ollama:   http://localhost:11434")
    print("  Backend:  ws://localhost:8001/ws")
    print("  Frontend: Flutter desktop" if frontend_ok else "  Frontend: [NOT RUNNING]")
    print()
    print("  Restart backend: python scripts/start_dev.py --restart")
    print("  View logs:       python scripts/start_dev.py --logs")
    print("  Stop all:        python scripts/start_dev.py --stop")
    print("  Run tests:       docker compose exec mist-backend pytest tests/unit/ -v")
    print("=" * 50)

    # Keep alive while Flutter runs
    if frontend_proc:
        import atexit
        import contextlib

        def cleanup():
            print()
            print("Shutting down...")
            with contextlib.suppress(OSError):
                frontend_proc.terminate()
            with contextlib.suppress(Exception):
                frontend_proc.wait(timeout=5)
            stop()

        atexit.register(cleanup)
        if sys.platform == "win32":
            import ctypes

            kernel32 = ctypes.windll.kernel32

            def console_handler(event):
                if event in (0, 2):
                    cleanup()
                    return True
                return False

            handler_func = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_ulong)(console_handler)
            kernel32.SetConsoleCtrlHandler(handler_func, True)

        try:
            while frontend_proc.poll() is None:
                time.sleep(1)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()

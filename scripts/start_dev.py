"""Start the MIST.AI backend development stack via Docker Compose.

Usage:
    python scripts/start_dev.py                 # Start backend stack
    python scripts/start_dev.py --with-frontend # Start backend + launch Tauri shell
    python scripts/start_dev.py --build         # Rebuild backend image then start
    python scripts/start_dev.py --full-restart  # docker compose down + up (respects --build)
    python scripts/start_dev.py --stop          # Stop all services
    python scripts/start_dev.py --logs          # Tail backend logs
    python scripts/start_dev.py --restart       # Restart backend container (pick up code changes)

Common workflows:
    # Full clean restart with frontend (rebuild backend image, restart stack,
    # launch Tauri shell). Use after pulling new BE deps like psutil/pynvml.
    python scripts/start_dev.py --full-restart --build --with-frontend

    # Quick verify after BE code change (no Dockerfile change):
    python scripts/start_dev.py --restart && python scripts/start_dev.py --with-frontend

The MIST Tauri frontend lives at ./mist-frontend/ as a nested git repo
(Tauri 2.x + React 19 + react-three-fiber). --with-frontend launches it
via scripts/start_frontend.py after the backend is healthy. Connection
to backend is over WebSocket per ADR-016 / ADR-017.

Prerequisites:
    - Docker Desktop with NVIDIA Container Toolkit
    - Node.js 18+ + Rust toolchain (cargo) for --with-frontend
"""

import argparse
import contextlib
import socket
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
FRONTEND_LAUNCHER = REPO_ROOT / "scripts" / "start_frontend.py"


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
    """Start the Docker Compose stack (backend + Neo4j + llama-server)."""
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


def launch_frontend() -> None:
    """Launch the Tauri frontend via scripts/start_frontend.py.

    Spawns the existing frontend launcher as a subprocess so we reuse its
    npm install + cargo check + `npm run tauri dev` orchestration. Stdio
    is inherited so the user sees Vite + Tauri build output in the same
    terminal. Blocks until the frontend exits (user Ctrl+C or window
    close); the backend stack keeps running independently.

    First run: Tauri compiles the Rust shell crate (2-10 min); subsequent
    runs are cached.
    """
    if not FRONTEND_LAUNCHER.exists():
        print(f"  [FAIL] Frontend launcher not found: {FRONTEND_LAUNCHER}")
        sys.exit(1)
    print()
    print("=" * 50)
    print("  Launching Tauri frontend (mist-frontend/)")
    print("=" * 50)
    print("  Handing off to scripts/start_frontend.py — Ctrl+C stops the")
    print("  Tauri shell only; the backend stack stays up. Stop backend")
    print("  later with: python scripts/start_dev.py --stop")
    print()
    try:
        subprocess.run([sys.executable, str(FRONTEND_LAUNCHER)], check=False)
    except KeyboardInterrupt:
        print()
        print("  Tauri shell stopped. Backend stays running.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MIST.AI dev stack manager: backend Docker stack + optional Tauri frontend.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--stop", action="store_true", help="Stop all services (docker compose down)"
    )
    parser.add_argument("--logs", action="store_true", help="Tail backend container logs")
    parser.add_argument(
        "--restart",
        action="store_true",
        help="Restart backend container only (pick up Python code changes; no image rebuild)",
    )
    parser.add_argument(
        "--build",
        action="store_true",
        help="Rebuild backend image before starting (use after Dockerfile / requirements.txt changes)",
    )
    parser.add_argument(
        "--full-restart",
        action="store_true",
        help="Full stack restart: docker compose down + up. Combine with --build for a clean rebuild + restart.",
    )
    parser.add_argument(
        "--with-frontend",
        action="store_true",
        help="After the backend is healthy, launch the Tauri frontend (scripts/start_frontend.py).",
    )
    args = parser.parse_args()

    # Terminal actions (no startup flow) ---------------------------------

    if args.stop:
        stop()
        return

    if args.logs:
        tail_logs()
        return

    if args.restart:
        restart_backend()
        if args.with_frontend:
            launch_frontend()
        return

    # Full-restart pre-stage: bring the stack down before normal startup.
    # Subsequent `docker compose up` either reuses cached images (no --build)
    # or rebuilds first (with --build) — same flow as a fresh start, just
    # gated behind an explicit teardown.
    if args.full_restart:
        print("Full-restart: bringing stack down before relaunch...")
        stop()
        print()

    # Normal startup flow ------------------------------------------------

    print("=" * 50)
    print("  MIST.AI Backend Development Stack (Docker)")
    print("=" * 50)
    print()

    # 1. Docker
    print("[1/4] Checking Docker...")
    if not check_docker():
        sys.exit(1)

    # 2. Start stack
    print("[2/4] Starting services...")
    if not start_stack(build=args.build):
        sys.exit(1)

    # 3. Wait for services
    #    Neo4j and Ollama: port checks (fast, no model loading).
    #    Backend: wait for Docker healthcheck (models take ~90s to load).
    print("[3/4] Waiting for services...")
    neo4j_ok = wait_for_service("Neo4j", "localhost", 7687, max_wait=60)
    ollama_ok = wait_for_service("Ollama", "localhost", 11434, max_wait=30)
    print("  Backend loading models (Whisper + Chatterbox + LLM)...")
    backend_ok = wait_for_container_healthy("mist-backend", max_wait=180)

    # 4. Model
    if ollama_ok:
        print("[4/4] Checking LLM model...")
        pull_model()

    if not (neo4j_ok and ollama_ok and backend_ok):
        print()
        print("=" * 50)
        print("  Stack NOT fully ready. Check logs:")
        print("  docker compose logs mist-backend")
        print("=" * 50)
        sys.exit(1)

    print()
    print("=" * 50)
    print("  Backend stack ready.")
    print()
    print("  Neo4j:    bolt://localhost:7687 (browser: http://localhost:7474)")
    print("  Ollama:   http://localhost:11434")
    print("  Backend:  ws://localhost:8001/ws")
    print()
    if args.with_frontend:
        print("  Frontend: launching Tauri shell via scripts/start_frontend.py...")
    else:
        print("  Frontend: separate nested repo at ./mist-frontend/")
        print("            Launch with: python scripts/start_frontend.py")
        print("            Or together: python scripts/start_dev.py --with-frontend")
    print()
    print("  Logs:     python scripts/start_dev.py --logs")
    print("  Restart:  python scripts/start_dev.py --restart")
    print("  Stop:     python scripts/start_dev.py --stop")
    print("  Tests:    docker compose exec mist-backend pytest tests/unit/ -v")
    print("=" * 50)

    if args.with_frontend:
        launch_frontend()


if __name__ == "__main__":
    main()

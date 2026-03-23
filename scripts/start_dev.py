"""Start the MIST.AI development stack.

Usage:
    python scripts/start_dev.py          # Start Ollama, verify Neo4j
    python scripts/start_dev.py --stop   # Stop Ollama

Prerequisites:
    - Neo4j Desktop running with a local database started
    - Ollama installed
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
    args = parser.parse_args()

    if args.stop:
        stop()
        return

    print("=" * 50)
    print("  MIST.AI Development Stack")
    print("=" * 50)
    print()

    # 1. Neo4j
    print("[1/3] Starting Neo4j...")
    neo4j_ok = start_neo4j()

    # 2. Ollama
    print("[2/3] Starting Ollama...")
    ollama_ok = start_ollama()

    # 3. Model
    if ollama_ok:
        print("[3/3] Checking model...")
        model_ok = check_model()
    else:
        model_ok = False

    print()
    print("=" * 50)
    if neo4j_ok and ollama_ok and model_ok:
        print("  Stack ready.")
        print()
        print("  Neo4j:   bolt://localhost:7687")
        print("  Ollama:  http://localhost:11434")
        print()
        print("  Start backend:  venv\\Scripts\\python backend\\server.py")
        print("  Start frontend: cd mist_desktop && flutter run -d windows")
        print("  Run tests:      venv\\Scripts\\python -m pytest tests/ -v")
    else:
        print("  Stack NOT ready. Fix the issues above.")
        sys.exit(1)
    print("=" * 50)


if __name__ == "__main__":
    main()

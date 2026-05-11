"""Start the MIST Tauri frontend dev server.

Usage:
    python scripts/start_frontend.py              # Run npm install (if needed) then npm run dev
    python scripts/start_frontend.py --build      # Run npm run build instead of dev
    python scripts/start_frontend.py --install    # Just run npm install and exit

The Tauri frontend lives at mist.ai/mist-frontend/ as a nested git repo
(Tauri 2.x + React 19 + react-three-fiber). It connects to the backend
at ws://localhost:8001/ws per ADR-016 + ADR-017.

Prerequisites:
    - Node.js 18+ and npm on PATH
    - Backend stack running (python scripts/start_dev.py)
    - Rust toolchain for Tauri (cargo, rustc) installed once via rustup

For a double-clickable shortcut see scripts/start_frontend.bat.
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
FRONTEND_DIR = REPO_ROOT / "mist-frontend"


def check_frontend_dir() -> bool:
    """Verify mist-frontend/ exists with a package.json."""
    if not FRONTEND_DIR.exists():
        print(f"  [FAIL] {FRONTEND_DIR} does not exist.")
        print("         Expected the Tauri frontend at this path.")
        return False
    if not (FRONTEND_DIR / "package.json").exists():
        print(f"  [FAIL] {FRONTEND_DIR} has no package.json.")
        return False
    print(f"  [OK] Frontend directory: {FRONTEND_DIR}")
    return True


def check_npm() -> bool:
    """Verify npm is on PATH."""
    npm = shutil.which("npm")
    if not npm:
        print("  [FAIL] npm not found on PATH. Install Node.js 18+.")
        return False
    print(f"  [OK] npm: {npm}")
    return True


def ensure_node_modules() -> bool:
    """Run npm install if node_modules/ is missing."""
    node_modules = FRONTEND_DIR / "node_modules"
    if node_modules.exists():
        print("  [OK] node_modules/ present (skipping npm install)")
        return True
    print("  Running npm install (first run; may take several minutes)...")
    try:
        result = subprocess.run(
            ["npm", "install"],
            cwd=str(FRONTEND_DIR),
            shell=(sys.platform == "win32"),
        )
        if result.returncode == 0:
            print("  [OK] npm install complete")
            return True
        print("  [FAIL] npm install failed")
        return False
    except FileNotFoundError:
        print("  [FAIL] npm not found")
        return False


def run_dev() -> None:
    """Run npm run dev (Vite + Tauri shell)."""
    print()
    print("Starting Tauri dev server...")
    print("  Vite: http://localhost:1420")
    print("  Tauri shell window will open when build is ready.")
    print("  Make sure the backend is running: python scripts/start_dev.py")
    print()
    try:
        subprocess.run(
            ["npm", "run", "dev"],
            cwd=str(FRONTEND_DIR),
            shell=(sys.platform == "win32"),
        )
    except KeyboardInterrupt:
        print("\n  Frontend stopped.")


def run_build() -> None:
    """Run npm run build (production Tauri bundle)."""
    print()
    print("Building Tauri production bundle...")
    try:
        result = subprocess.run(
            ["npm", "run", "build"],
            cwd=str(FRONTEND_DIR),
            shell=(sys.platform == "win32"),
        )
        if result.returncode == 0:
            print(
                "  [OK] Build complete. See mist-frontend/src-tauri/target/release/ for the binary."
            )
        else:
            print("  [FAIL] Build failed.")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n  Build cancelled.")


def main() -> None:
    parser = argparse.ArgumentParser(description="MIST Tauri frontend dev/build manager")
    parser.add_argument("--build", action="store_true", help="Run npm run build instead of dev")
    parser.add_argument("--install", action="store_true", help="Just run npm install and exit")
    args = parser.parse_args()

    print("=" * 50)
    print("  MIST Tauri Frontend")
    print("=" * 50)
    print()

    if not check_frontend_dir():
        sys.exit(1)
    if not check_npm():
        sys.exit(1)
    if not ensure_node_modules():
        sys.exit(1)

    if args.install:
        print()
        print("  [OK] npm install only. Done.")
        return

    if args.build:
        run_build()
        return

    run_dev()


if __name__ == "__main__":
    main()

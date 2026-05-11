"""Start the MIST Tauri frontend (native shell window).

Usage:
    python scripts/start_frontend.py              # Tauri shell + Vite (the native window)
    python scripts/start_frontend.py --web        # Vite only, browser-accessible at :1420
    python scripts/start_frontend.py --build      # Production Tauri bundle (npm run tauri build)
    python scripts/start_frontend.py --install    # Just run npm install and exit

The Tauri frontend lives at mist.ai/mist-frontend/ as a nested git repo
(Tauri 2.x + React 19 + react-three-fiber). It connects to the backend
at ws://localhost:8001/ws per ADR-016 + ADR-017.

Default mode (`npm run tauri dev`) launches the native Tauri shell window
together with the Vite dev server. Use --web for browser-only verification
(Vite at http://localhost:1420/) without spinning up the shell.

Prerequisites:
    - Node.js 18+ and npm on PATH
    - Rust toolchain (cargo, rustc) for Tauri dev/build; install via rustup
    - Backend stack running (python scripts/start_dev.py)

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


def check_cargo() -> bool:
    """Verify Rust toolchain (cargo) is on PATH; required for Tauri dev/build."""
    cargo = shutil.which("cargo")
    if not cargo:
        print("  [WARN] cargo (Rust) not found on PATH. Tauri dev/build requires it.")
        print("         Install via https://rustup.rs/ and reopen the shell.")
        return False
    print(f"  [OK] cargo: {cargo}")
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


def run_tauri_dev() -> None:
    """Run `npm run tauri dev` — starts Vite AND opens the native Tauri shell window.

    First-run cost: Tauri compiles the Rust shell crate (cargo build), which can
    take 2-10 minutes depending on dependencies. Subsequent runs are cached.
    """
    print()
    print("Starting Tauri shell + Vite dev server...")
    print("  Vite:  http://localhost:1420 (also accessible via browser)")
    print("  Shell: native window will open after cargo build completes.")
    print("  First run: cargo build can take several minutes.")
    print("  Make sure the backend is running: python scripts/start_dev.py")
    print()
    try:
        subprocess.run(
            ["npm", "run", "tauri", "dev"],
            cwd=str(FRONTEND_DIR),
            shell=(sys.platform == "win32"),
        )
    except KeyboardInterrupt:
        print("\n  Tauri shell stopped.")


def run_vite_only() -> None:
    """Run `npm run dev` — Vite only; no native Tauri shell. Browser-accessible at :1420."""
    print()
    print("Starting Vite dev server (browser-only mode, no Tauri shell)...")
    print("  Vite: http://localhost:1420")
    print("  Note: native Tauri features (file dialogs, OS integration) are unavailable.")
    print("  Make sure the backend is running: python scripts/start_dev.py")
    print()
    try:
        subprocess.run(
            ["npm", "run", "dev"],
            cwd=str(FRONTEND_DIR),
            shell=(sys.platform == "win32"),
        )
    except KeyboardInterrupt:
        print("\n  Vite stopped.")


def run_build() -> None:
    """Run `npm run tauri build` — production Tauri bundle (installer + binary)."""
    print()
    print("Building Tauri production bundle...")
    try:
        result = subprocess.run(
            ["npm", "run", "tauri", "build"],
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
    parser.add_argument(
        "--web",
        action="store_true",
        help="Run Vite only (browser-accessible at :1420); skip the Tauri shell.",
    )
    parser.add_argument(
        "--build",
        action="store_true",
        help="Build the production Tauri bundle (npm run tauri build).",
    )
    parser.add_argument(
        "--install",
        action="store_true",
        help="Just run npm install and exit.",
    )
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

    # Tauri (shell + build) requires cargo. Vite-only does not.
    needs_cargo = args.build or not args.web
    if needs_cargo and not check_cargo():
        print()
        print("  Cannot run Tauri shell or build without cargo.")
        print("  Use --web to run Vite in browser-only mode without the Tauri shell.")
        sys.exit(1)

    if args.build:
        run_build()
        return

    if args.web:
        run_vite_only()
        return

    run_tauri_dev()


if __name__ == "__main__":
    main()

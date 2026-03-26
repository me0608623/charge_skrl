"""Charge-SKRL Dashboard — Desktop Application Launcher

啟動 FastAPI server 並以 Chrome --app 模式開啟桌面應用視窗。
Ctrl+C 或關閉視窗即停止 server。

Usage:
    python scripts/reinforcement_learning/skrl/dashboard/launcher.py
    # 或
    ./launch_dashboard.sh
"""

from __future__ import annotations

import atexit
import os
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
HOST = "127.0.0.1"
PORT = 8050
URL = f"http://{HOST}:{PORT}"
DASHBOARD_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _find_free_port(start: int = 8050, end: int = 8100) -> int:
    """Find a free port in range."""
    for port in range(start, end):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    return start


def _wait_for_server(host: str, port: int, timeout: float = 15.0) -> bool:
    """Wait until the server is accepting connections."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.5):
                return True
        except (ConnectionRefusedError, OSError):
            time.sleep(0.2)
    return False


def _open_chrome_app(url: str) -> subprocess.Popen | None:
    """Open URL in Chrome --app mode (frameless window). Falls back to default browser.

    Uses a dedicated user-data-dir so that the --app window always opens as a
    new, independent process — even when Chrome is already running. This avoids
    the issue where Chrome delegates to the existing instance and exits immediately,
    preventing the dashboard window from appearing.

    Returns (proc, owns_process):
        proc: the Popen object (or None on fallback)
        owns_process: True if we launched an independent Chrome process that we
                      should wait on; False if we delegated to an existing instance.
    """
    chrome_candidates = [
        "google-chrome",
        "google-chrome-stable",
        "chromium-browser",
        "chromium",
    ]
    # Dedicated profile dir ensures a new independent Chrome process
    profile_dir = Path.home() / ".cache" / "charge-dashboard-chrome"
    profile_dir.mkdir(parents=True, exist_ok=True)

    # Clean up stale singleton locks from previous crashed sessions
    lock_file = profile_dir / "SingletonLock"
    if lock_file.is_symlink():
        lock_target = str(os.readlink(lock_file))
        # Lock format: "user-UID-PID" or "hostname-PID"
        parts = lock_target.rsplit("-", 1)
        if len(parts) == 2 and parts[1].isdigit():
            try:
                os.kill(int(parts[1]), 0)
            except ProcessLookupError:
                # Process is dead — remove stale locks
                for f in ("SingletonLock", "SingletonSocket", "SingletonCookie"):
                    (profile_dir / f).unlink(missing_ok=True)

    for name in chrome_candidates:
        chrome_path = shutil.which(name)
        if chrome_path:
            try:
                proc = subprocess.Popen(
                    [
                        chrome_path,
                        f"--app={url}",
                        f"--user-data-dir={profile_dir}",
                        f"--window-size=1400,900",
                        "--no-first-run",
                        "--no-default-browser-check",
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                # With --user-data-dir, Chrome always starts a new process
                return proc, True
            except OSError:
                continue

    # Fallback: default browser (will delegate to existing instance)
    webbrowser.open(url)
    return None, False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    global PORT, URL

    PORT = _find_free_port(PORT)
    URL = f"http://{HOST}:{PORT}"

    # Ensure dashboard dir is on sys.path (for config_parser imports)
    sys.path.insert(0, str(DASHBOARD_DIR))

    print(f"\033[1;34m{'=' * 60}\033[0m")
    print(f"\033[1;34m  Charge-SKRL Training Dashboard\033[0m")
    print(f"\033[1;34m{'=' * 60}\033[0m")
    print(f"  Starting server on \033[1;36m{URL}\033[0m ...")

    # Start uvicorn in a thread
    import uvicorn

    # Import app directly (sys.path already includes DASHBOARD_DIR)
    from app import app as fastapi_app  # noqa: E402

    server_config = uvicorn.Config(
        fastapi_app,
        host=HOST,
        port=PORT,
        log_level="warning",
    )
    server = uvicorn.Server(server_config)

    server_thread = threading.Thread(target=server.run, daemon=True)
    server_thread.start()

    # Wait for server to be ready
    if not _wait_for_server(HOST, PORT):
        print("\033[1;31m  ERROR: Server failed to start within 15s\033[0m")
        sys.exit(1)

    print(f"  Server ready! Opening desktop window...")
    print(f"  Press \033[1;33mCtrl+C\033[0m to quit.\n")

    # Open Chrome app window (uses dedicated profile → always a new process)
    chrome_proc, owns_process = _open_chrome_app(URL)

    # Graceful shutdown
    def _shutdown(*_):
        print(f"\n  Shutting down...")
        server.should_exit = True
        if chrome_proc and chrome_proc.poll() is None:
            chrome_proc.terminate()
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)
    atexit.register(lambda: setattr(server, "should_exit", True))

    if chrome_proc and owns_process:
        try:
            # Give Chrome time to initialize
            time.sleep(3)
            if chrome_proc.poll() is not None:
                # Chrome exited quickly without an existing session —
                # likely no display available. Keep server for manual access.
                print(f"  Browser window not available. Server still running at \033[1;36m{URL}\033[0m")
                print(f"  Open \033[1;36m{URL}\033[0m in your browser manually.")
                while True:
                    time.sleep(1)
            else:
                # Chrome is running as a new instance — wait for user to close it
                chrome_proc.wait()
                print("  Window closed. Shutting down server...")
                server.should_exit = True
                time.sleep(0.5)
        except KeyboardInterrupt:
            _shutdown()
    else:
        # Fallback: just wait for Ctrl+C
        print(f"  Open \033[1;36m{URL}\033[0m in your browser.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            _shutdown()


if __name__ == "__main__":
    main()

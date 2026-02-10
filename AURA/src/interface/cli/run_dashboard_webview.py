"""Launch the dashboard URL in a native pywebview window."""

from __future__ import annotations

import argparse
import sys
import time
import urllib.error
import urllib.request


def wait_for_url(url: str, timeout_sec: float) -> bool:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=2):
                return True
        except (urllib.error.URLError, TimeoutError, OSError):
            time.sleep(0.5)
    return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run dashboard in pywebview.")
    parser.add_argument("--url", required=True, help="Dashboard URL")
    parser.add_argument("--title", default="AURA Dashboard", help="Window title")
    parser.add_argument("--width", type=int, default=1400, help="Initial width")
    parser.add_argument("--height", type=int, default=900, help="Initial height")
    parser.add_argument(
        "--wait-timeout-sec",
        type=float,
        default=20.0,
        help="Seconds to wait for URL to become reachable",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable pywebview debug mode",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        import webview
    except Exception as exc:  # pragma: no cover
        print(f"[error] pywebview import failed: {exc}", file=sys.stderr)
        return 1

    ready = wait_for_url(args.url, args.wait_timeout_sec)
    if not ready:
        print(
            f"[warn] URL not reachable after {args.wait_timeout_sec:.1f}s. Opening anyway: {args.url}",
            file=sys.stderr,
        )

    class DashboardWebviewApi:
        def __init__(self) -> None:
            self._window = None

        def bind_window(self, window) -> None:
            self._window = window

        def close_window(self) -> bool:
            if self._window is None:
                return False
            try:
                self._window.destroy()
                return True
            except Exception:
                return False

    js_api = DashboardWebviewApi()
    window = webview.create_window(
        title=args.title,
        url=args.url,
        width=args.width,
        height=args.height,
        min_size=(960, 640),
        resizable=True,
        js_api=js_api,
    )
    js_api.bind_window(window)
    webview.start(debug=args.debug)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

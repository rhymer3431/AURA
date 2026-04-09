#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


AURA_ROOT = Path(__file__).resolve().parent
SRC_ROOT = AURA_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from g1_play.internvla_nav import InternVlaNavClient


DEFAULT_SERVER_URL = os.environ.get("INTERNVLA_SERVER_URL", "http://127.0.0.1:15801").strip() or "http://127.0.0.1:15801"
DEFAULT_MAX_TOKENS = int(os.environ.get("INTERNVLA_CHECK_MAX_TOKENS", "4") or 4)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Query the persistent InternVLA /check session.")
    parser.add_argument("message", nargs="*", help="Optional one-shot question. Omit for interactive mode.")
    parser.add_argument("--server-url", default=DEFAULT_SERVER_URL, help="InternVLA server base URL.")
    parser.add_argument("--session-id", default="", help="Optional check session id. Default uses the server default session.")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS, help="Maximum model tokens per check reply.")
    parser.add_argument("--json", action="store_true", help="Print full JSON responses instead of only the answer.")
    return parser.parse_args(argv)


def _print_response(response: dict[str, object], *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(response, ensure_ascii=False, indent=2))
        return
    answer = str(response.get("answer", "")).strip()
    if answer:
        print(answer)
        return
    print(json.dumps(response, ensure_ascii=False, indent=2))


def _open_session(client: InternVlaNavClient, args: argparse.Namespace) -> str:
    response = client.open_check_session(
        session_id=str(args.session_id).strip() or None,
    )
    session_id = str(response.get("session_id", "")).strip()
    if not session_id:
        raise RuntimeError(f"Server did not return a session_id: {response}")
    return session_id


def run_once(client: InternVlaNavClient, args: argparse.Namespace) -> int:
    session_id = _open_session(client, args)
    response = client.check_message(
        " ".join(args.message).strip(),
        session_id=session_id,
        max_tokens=int(args.max_tokens),
    )
    _print_response(response, as_json=bool(args.json))
    return 0


def run_interactive(client: InternVlaNavClient, args: argparse.Namespace) -> int:
    session_id = _open_session(client, args)
    print(f"[INFO] Server   : {args.server_url}")
    print(f"[INFO] Session  : {session_id}")
    print("[INFO] Commands : /exit, /quit, /close, /reopen")
    while True:
        try:
            user_input = input("check> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not user_input:
            continue
        if user_input in {"/exit", "/quit"}:
            return 0
        if user_input == "/close":
            response = client.close_check_session(session_id=session_id)
            if bool(response.get("reopened_default_session")):
                session_id = str(response.get("default_check_session_id", "")).strip() or session_id
            if args.json:
                print(json.dumps(response, ensure_ascii=False, indent=2))
            else:
                print(f"[INFO] Session close result: {json.dumps(response, ensure_ascii=False)}")
            continue
        if user_input == "/reopen":
            session_id = _open_session(client, args)
            print(f"[INFO] Reopened session: {session_id}")
            continue
        response = client.check_message(
            user_input,
            session_id=session_id,
            max_tokens=int(args.max_tokens),
        )
        _print_response(response, as_json=bool(args.json))


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    client = InternVlaNavClient(server_url=str(args.server_url), timeout_s=60.0)
    if args.message:
        return run_once(client, args)
    return run_interactive(client, args)


if __name__ == "__main__":
    raise SystemExit(main())

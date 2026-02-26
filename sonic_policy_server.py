#!/usr/bin/env python
from apps.sonic_policy_server.server import *  # noqa: F401,F403
from apps.sonic_policy_server import server as _server


if __name__ == "__main__":
    _args = _server._parse_args()
    _server.run_server(
        encoder_path=_args.encoder,
        decoder_path=_args.decoder,
        planner_path=_args.planner,
        host=_args.host,
        port=_args.port,
        action_scale_multiplier=_args.action_scale_multiplier,
    )

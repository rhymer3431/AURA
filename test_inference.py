import io
import json
from pathlib import Path

import msgpack
import numpy as np
import zmq


def _encode_custom(obj):
    if isinstance(obj, np.ndarray):
        buf = io.BytesIO()
        np.save(buf, obj, allow_pickle=False)
        return {"__ndarray_class__": True, "as_npy": buf.getvalue()}
    raise TypeError(type(obj))


def _decode_custom(obj):
    if isinstance(obj, dict) and "__ndarray_class__" in obj:
        return np.load(io.BytesIO(obj["as_npy"]), allow_pickle=False)
    return obj


def _build_observation() -> dict:
    proc = json.loads(
        Path("models/gr00t_n1_6_g1_pnp_apple_to_plate/processor_config.json").read_text(encoding="utf-8")
    )
    stats = json.loads(
        Path("models/gr00t_n1_6_g1_pnp_apple_to_plate/statistics.json").read_text(encoding="utf-8")
    )

    emb = "unitree_g1"
    emb_cfg = proc["processor_kwargs"]["modality_configs"][emb]
    emb_stats = stats[emb]

    video_keys = list(emb_cfg["video"]["modality_keys"])
    state_keys = list(emb_cfg["state"]["modality_keys"])
    language_key = emb_cfg["language"]["modality_keys"][0]
    video_h = max(1, len(emb_cfg["video"]["delta_indices"]))
    state_h = max(1, len(emb_cfg["state"]["delta_indices"]))

    obs = {}
    for key in video_keys:
        # Gr00tSimPolicyWrapper expects flat keys, uint8, BCHW? no -> (B, T, H, W, C).
        obs[f"video.{key}"] = np.zeros((1, video_h, 256, 256, 3), dtype=np.uint8)

    for key in state_keys:
        dim = len(emb_stats["state"][key]["mean"])
        obs[f"state.{key}"] = np.zeros((1, state_h, dim), dtype=np.float32)

    obs[language_key] = ["walk forward"]
    return obs


def main():
    ctx = zmq.Context()
    sock = ctx.socket(zmq.REQ)
    sock.setsockopt(zmq.RCVTIMEO, 60000)
    sock.setsockopt(zmq.SNDTIMEO, 5000)
    sock.setsockopt(zmq.LINGER, 0)
    sock.connect("tcp://127.0.0.1:5555")

    req = {
        "endpoint": "get_action",
        "data": {"observation": _build_observation(), "options": None},
    }

    sock.send(msgpack.packb(req, use_bin_type=True, default=_encode_custom))
    raw = sock.recv()
    response = msgpack.unpackb(raw, raw=False, object_hook=_decode_custom)

    if isinstance(response, dict) and "error" in response:
        print("ERROR:", response["error"])
        return

    if isinstance(response, (list, tuple)) and len(response) >= 1:
        actions = response[0]
        info = response[1] if len(response) > 1 else {}
        if isinstance(actions, dict):
            keys = list(actions.keys())
            print("Response keys:", ["actions", "info"])
            print("Action dict keys:", keys)
            first_key = keys[0] if keys else None
            if first_key is not None:
                arr = np.asarray(actions[first_key])
                print(f"Action shape ({first_key}):", arr.shape)
                flat = arr.reshape(-1)
                print("Action sample (first values):", flat[:6].tolist())
            print("Info keys:", list(info.keys()) if isinstance(info, dict) else type(info).__name__)
            return

    if isinstance(response, dict):
        print("Response keys:", list(response.keys()))
        actions = response.get("actions")
        if actions is not None:
            arr = np.asarray(actions)
            print("Action shape:", arr.shape)
            print("Action sample (first timestep):", arr.reshape(-1)[:6].tolist())
        else:
            print("ERROR: No actions in response:", response)
        return

    print("Unexpected response type:", type(response).__name__, response)


if __name__ == "__main__":
    main()

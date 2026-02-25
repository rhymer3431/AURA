import json
import time

import zmq

context = zmq.Context()

# Test REQ/REP
print("=== REQ/REP Test ===")
socket = context.socket(zmq.REQ)
socket.setsockopt(zmq.RCVTIMEO, 5000)
socket.setsockopt(zmq.SNDTIMEO, 5000)
socket.setsockopt(zmq.LINGER, 0)
socket.connect("tcp://127.0.0.1:5555")

try:
    test_payload = {"modality": "text", "text": "walk forward"}
    socket.send_json(test_payload)
    response = socket.recv_json()
    print(f"Response received: {response}")
except zmq.error.Again:
    print("REQ timeout - server not responding or socket type mismatch")
except Exception as exc:
    print(f"REQ error: {type(exc).__name__}: {exc}")
finally:
    socket.close()

# Test DEALER
print("\n=== DEALER/ROUTER Test ===")
socket2 = context.socket(zmq.DEALER)
socket2.setsockopt(zmq.RCVTIMEO, 5000)
socket2.setsockopt(zmq.SNDTIMEO, 5000)
socket2.setsockopt(zmq.LINGER, 0)
socket2.connect("tcp://127.0.0.1:5555")
try:
    socket2.send_json({"modality": "text", "text": "walk forward"})
    response = socket2.recv_json()
    print(f"Response received: {response}")
except zmq.error.Again:
    print("DEALER timeout")
except Exception as exc:
    print(f"DEALER error: {type(exc).__name__}: {exc}")
finally:
    socket2.close()

context.term()

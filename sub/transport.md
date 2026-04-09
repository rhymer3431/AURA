# Transport Subsystem

- Scope: runtime message contracts, bus abstraction, in-process and ZMQ transports, shared-memory frame transport, and transport health.
- Package root: `src/systems/transport`
- Components:
  - `messages.py`
  - `bus/base.py`, `bus/inproc_bus.py`, `bus/zmq_bus.py`
  - `codec.py`, `frame_codec.py`
  - `shm.py`
  - `health.py`
- Notes:
  - This subsystem is runtime substrate, not domain logic.
  - It is shared by runtime, server, dashboard, WebRTC, adapters, and bridge code.

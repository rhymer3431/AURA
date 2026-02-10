"""Online (server) entrypoint that exposes the streaming FastAPI app."""
import os
import sys
from pathlib import Path

import uvicorn

sys.path.append(str(Path(__file__).resolve().parents[3]))

from src.interface.api.app_factory import app, create_app


def main():
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app or create_app(), host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()

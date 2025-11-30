# Legacy HTTP VisClient removed in favor of DearPyGuiClient for local UI.
# Kept as a placeholder to avoid import errors if referenced elsewhere.

class VisClient:
    def __init__(self, *args, **kwargs):
        raise RuntimeError("VisClient HTTP mode removed. Use DearPyGuiClient instead.")

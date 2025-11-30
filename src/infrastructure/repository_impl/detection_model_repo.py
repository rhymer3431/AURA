from domain.detect.repository.detection_model_port import DetectionModelPort


class DetectionModelRepository(DetectionModelPort):
    """Simple pass-through repository for detection adapters."""

    def __init__(self, adapter: DetectionModelPort):
        self.adapter = adapter

    def track(self, frame):
        return self.adapter.track(frame)

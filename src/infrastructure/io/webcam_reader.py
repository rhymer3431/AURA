from infrastructure.io.video_reader_cv2 import VideoReaderCV2


class WebcamReader(VideoReaderCV2):
    def __init__(self, index: int = 0):
        super().__init__(index)

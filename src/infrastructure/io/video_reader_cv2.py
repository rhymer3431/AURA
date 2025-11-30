import cv2


class VideoReaderCV2:
    def __init__(self, source: str | int):
        self.cap = cv2.VideoCapture(source)

    def read(self):
        return self.cap.read()

    def release(self):
        self.cap.release()

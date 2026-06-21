import cv2


class Camera:
    def __init__(self, index=0, width=1280, height=720):
        self.index = index
        self.cap = self._open_capture(index)

        if not self.cap or not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera {index}.")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def _open_capture(self, index):
        backends = []
        if hasattr(cv2, 'CAP_DSHOW'):
            backends.append(cv2.CAP_DSHOW)
        if hasattr(cv2, 'CAP_MSMF'):
            backends.append(cv2.CAP_MSMF)
        backends.append(0)

        for backend in backends:
            cap = cv2.VideoCapture(index, backend)
            if cap.isOpened():
                return cap
            cap.release()
        return None

    def get_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame

    def release(self):
        self.cap.release()


class MultiCamera:
    def __init__(self, indexes=(1, 0), width=1280, height=720):
        self.cameras = [Camera(index=i, width=width, height=height) for i in indexes]

    def get_frames(self):
        frames = []
        for cam in self.cameras:
            frame = cam.get_frame()
            if frame is None:
                return None
            frames.append(frame)
        return frames

    def release(self):
        for cam in self.cameras:
            cam.release()
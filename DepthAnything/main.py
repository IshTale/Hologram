import cv2
import time

from camera import Camera
from depth_model import DepthEstimator
from depth_stack import DepthStack
from visualization import Visualizer


def main():
    camera = Camera(index=0, width=1280, height=720)

    # use_amp=False is more stable.
    # Later, try use_amp=True for more speed.
    depth_estimator = DepthEstimator(use_amp=False)

    depth_stack = DepthStack(num_layers=5)
    visualizer = Visualizer()

    print("Press Q to quit.")

    while True:
        frame = camera.get_frame()

        if frame is None:
            print("No frame received.")
            break

        start = time.time()

        depth_map = depth_estimator.estimate_depth(frame)
        layers = depth_stack.generate_layers(depth_map)

        end = time.time()
        fps = 1.0 / max(end - start, 1e-6)

        visualizer.show(frame, depth_map, layers, fps)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
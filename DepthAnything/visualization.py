import cv2


class Visualizer:
    def show(self, frame, depth_map, layers, fps):
        display_frame = frame.copy()
        depth_color = cv2.applyColorMap(depth_map, cv2.COLORMAP_JET)

        cv2.putText(
            display_frame,
            f"FPS: {fps:.2f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.imshow("Camera Feed", display_frame)
        cv2.imshow("Depth Map", depth_color)

        for i, layer in enumerate(layers):
            cv2.imshow(f"Depth Layer {i + 1}", layer)
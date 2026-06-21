import cv2
import time
import numpy as np

from camera import Camera, MultiCamera
from depth_model import DepthEstimator
from depth_stack import DepthStack
from visualization import Visualizer


class StereoCalibrator:
    def __init__(self, pattern_size=(8, 5), square_size=1.0, calib_file="stereo_calibration.npz"):
        self.pattern_size = pattern_size
        self.square_size = square_size
        self.calib_file = calib_file
        self.objpoints = []
        self.imgpoints_left = []
        self.imgpoints_right = []
        self.maps = None
        self.image_size = None

    def _object_point(self, pattern_size=None):
        if pattern_size is None:
            pattern_size = self.pattern_size
        objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
        return objp * self.square_size

    def find_corners(self, gray):
        # Try several candidate patterns and preprocessing variants to improve robustness.
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
        used_pattern = None
        method = None

        try:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray_clahe = clahe.apply(gray)
        except Exception:
            gray_clahe = gray

        gray_thresh = cv2.adaptiveThreshold(
            gray_clahe, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )

        # Candidate patterns: user-specified, user-1 (squares->internal), and swapped
        candidates = [self.pattern_size,
                      (max(1, self.pattern_size[0] - 1), max(1, self.pattern_size[1] - 1)),
                      (self.pattern_size[1], self.pattern_size[0])]

        # Try on original, CLAHE, then threshold
        for pattern in candidates:
            if used_pattern is not None:
                break
            for img in (gray, gray_clahe, gray_thresh):
                found, corners = cv2.findChessboardCorners(img, pattern, flags)
                if found:
                    used_pattern = pattern
                    method = 'chessboard'
                    break

        # Try SB detector as a last resort
        if used_pattern is None and hasattr(cv2, 'findChessboardCornersSB'):
            try:
                found, corners = cv2.findChessboardCornersSB(gray_clahe, self.pattern_size)
                if found:
                    used_pattern = self.pattern_size
                    method = 'chessboard_sb'
            except Exception:
                found = False

        # If chessboard wasn't found, try circles grid (symmetric) as a fallback
        if used_pattern is None:
            for pattern in candidates:
                if used_pattern is not None:
                    break
                for img in (gray, gray_clahe, gray_thresh):
                    try:
                        flags_circ = cv2.CALIB_CB_SYMMETRIC_GRID
                        found_c, corners_c = cv2.findCirclesGrid(img, pattern, flags=flags_circ)
                        if found_c:
                            used_pattern = pattern
                            corners = corners_c
                            method = 'circles'
                            found = True
                            break
                    except Exception:
                        found_c = False
                if used_pattern is not None:
                    break

        if used_pattern is None:
            return False, None, None

        # refine corners
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        try:
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        except Exception:
            # cornerSubPix may fail for circle grids; ignore refinement in that case
            pass
        print(f"Detection: method={method}, pattern={used_pattern}")
        return True, corners, used_pattern

    def add_pair(self, left_gray, right_gray):
        found_left, corners_left, used_left = self.find_corners(left_gray)
        found_right, corners_right, used_right = self.find_corners(right_gray)
        if found_left and found_right and used_left == used_right:
            self.objpoints.append(self._object_point(pattern_size=used_left))
            self.imgpoints_left.append(corners_left)
            self.imgpoints_right.append(corners_right)
            return True

        if not found_left and not found_right:
            print("Chessboard not found in either image.")
        elif not found_left:
            print("Chessboard not found in left image.")
        elif not found_right:
            print("Chessboard not found in right image.")
        else:
            print(f"Detected different patterns: left={used_left}, right={used_right}. Make sure both cameras see the same pattern.")
        return False

    def calibrate(self, image_size):
        if len(self.objpoints) < 6:
            raise RuntimeError("Need at least 6 calibration pairs.")

        self.image_size = image_size
        ret_left, mtx_left, dist_left, _, _ = cv2.calibrateCamera(
            self.objpoints, self.imgpoints_left, image_size, None, None
        )
        ret_right, mtx_right, dist_right, _, _ = cv2.calibrateCamera(
            self.objpoints, self.imgpoints_right, image_size, None, None
        )

        flags = cv2.CALIB_FIX_INTRINSIC
        criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 30, 1e-6)
        _, mtx_left, dist_left, mtx_right, dist_right, R, T, _, _ = cv2.stereoCalibrate(
            self.objpoints,
            self.imgpoints_left,
            self.imgpoints_right,
            mtx_left,
            dist_left,
            mtx_right,
            dist_right,
            image_size,
            criteria=criteria,
            flags=flags,
        )

        R1, R2, P1, P2, _, _, _ = cv2.stereoRectify(
            mtx_left,
            dist_left,
            mtx_right,
            dist_right,
            image_size,
            R,
            T,
            flags=cv2.CALIB_ZERO_DISPARITY,
            alpha=0,
        )

        map1_x, map1_y = cv2.initUndistortRectifyMap(
            mtx_left, dist_left, R1, P1, image_size, cv2.CV_32FC1
        )
        map2_x, map2_y = cv2.initUndistortRectifyMap(
            mtx_right, dist_right, R2, P2, image_size, cv2.CV_32FC1
        )

        self.maps = {
            "map1_x": map1_x,
            "map1_y": map1_y,
            "map2_x": map2_x,
            "map2_y": map2_y,
        }
        np.savez(
            self.calib_file,
            map1_x=map1_x,
            map1_y=map1_y,
            map2_x=map2_x,
            map2_y=map2_y,
        )

    def load(self):
        try:
            data = np.load(self.calib_file)
            self.maps = {
                "map1_x": data["map1_x"],
                "map1_y": data["map1_y"],
                "map2_x": data["map2_x"],
                "map2_y": data["map2_y"],
            }
            return True
        except Exception:
            return False

    def rectify_frames(self, frames):
        if self.maps is None:
            return frames
        rectified = []
        rectified.append(
            cv2.remap(frames[0], self.maps["map1_x"], self.maps["map1_y"], cv2.INTER_LINEAR)
        )
        rectified.append(
            cv2.remap(frames[1], self.maps["map2_x"], self.maps["map2_y"], cv2.INTER_LINEAR)
        )
        return rectified

    def run_interactive(self, cameras, min_pairs=6):
        print(
            "Interactive stereo calibration mode: press C to capture a pair, F to finish, Q to quit."
        )
        captured = 0
        while True:
            frames = cameras.get_frames()
            if frames is None:
                print("Missing frame during calibration.")
                break

            left = frames[0].copy()
            right = frames[1].copy()
            combined = np.hstack((left, right))
            cv2.putText(
                combined,
                f"Pairs: {captured}/{min_pairs} - C capture, F finish, Q quit",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )
            cv2.imshow("Stereo Calibration", combined)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("c"):
                left_gray = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
                right_gray = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)

                # Save debug images so you can inspect what the detector sees
                debug_id = len(self.objpoints) + captured + 1
                cv2.imwrite(f"debug_left_{debug_id}.png", left)
                cv2.imwrite(f"debug_right_{debug_id}.png", right)
                try:
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                    left_clahe = clahe.apply(left_gray)
                    right_clahe = clahe.apply(right_gray)
                except Exception:
                    left_clahe = left_gray
                    right_clahe = right_gray
                cv2.imwrite(f"debug_left_clahe_{debug_id}.png", left_clahe)
                cv2.imwrite(f"debug_right_clahe_{debug_id}.png", right_clahe)
                left_thresh = cv2.adaptiveThreshold(
                    left_clahe, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
                )
                right_thresh = cv2.adaptiveThreshold(
                    right_clahe, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
                )
                cv2.imwrite(f"debug_left_thresh_{debug_id}.png", left_thresh)
                cv2.imwrite(f"debug_right_thresh_{debug_id}.png", right_thresh)

                # Try to find corners and draw them for immediate feedback
                found_left, corners_left, used_left = self.find_corners(left_gray)
                found_right, corners_right, used_right = self.find_corners(right_gray)

                drawn_left = left.copy()
                drawn_right = right.copy()
                if used_left is not None:
                    cv2.drawChessboardCorners(drawn_left, used_left, corners_left if found_left else None, found_left)
                if used_right is not None:
                    cv2.drawChessboardCorners(drawn_right, used_right, corners_right if found_right else None, found_right)

                combined_draw = np.hstack((drawn_left, drawn_right))
                cv2.imshow("Stereo Calibration", combined_draw)
                cv2.waitKey(500)

                if found_left and found_right and used_left == used_right and self.add_pair(left_gray, right_gray):
                    captured += 1
                    print(f"Captured calibration pair {captured}.")
                else:
                    if not found_left and not found_right:
                        print("Chessboard not found in either image. Saved debug images for inspection.")
                    elif not found_left:
                        print("Chessboard not found in left image (see overlay and debug images).")
                    elif not found_right:
                        print("Chessboard not found in right image (see overlay and debug images).")
                    else:
                        print(f"Detected different patterns: left={used_left}, right={used_right}. Saved debug images.")
            elif key == ord("f"):
                if captured >= min_pairs:
                    image_size = (frames[0].shape[1], frames[0].shape[0])
                    self.calibrate(image_size)
                    cv2.destroyWindow("Stereo Calibration")
                    return True
                print(f"Need at least {min_pairs} successful pairs.")
            elif key == ord("q"):
                break

        cv2.destroyWindow("Stereo Calibration")
        return False


def warp_with_depth(frame, depth_map, direction=1, max_shift=10):
    h, w = depth_map.shape

    # Smooth depth to reduce high-frequency noise which causes streaking when warping
    try:
        depth_smooth = cv2.bilateralFilter(depth_map, 9, 75, 75)
    except Exception:
        depth_smooth = cv2.GaussianBlur(depth_map, (9, 9), 0)

    depth_norm = depth_smooth.astype(np.float32) / 255.0
    shift_map = ((1.0 - depth_norm) * float(max_shift) * direction).astype(np.float32)

    x_coords = np.arange(w, dtype=np.float32)
    base_x = np.tile(x_coords, (h, 1))
    map_x = base_x + shift_map
    map_y = np.tile(np.arange(h, dtype=np.float32)[:, None], (1, w))

    # Clamp coordinates to valid ranges to avoid extreme sampling and stretching
    np.clip(map_x, 0, w - 1, out=map_x)
    np.clip(map_y, 0, h - 1, out=map_y)

    return cv2.remap(frame, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def create_parallax_view(frame_left, frame_right, depth_left, depth_right, max_shift=10):
    warped_left = warp_with_depth(frame_left, depth_left, direction=1, max_shift=max_shift)
    warped_right = warp_with_depth(frame_right, depth_right, direction=-1, max_shift=max_shift)
    return cv2.addWeighted(warped_left, 0.5, warped_right, 0.5, 0)


def create_parallax_view_inverted(frame_left, frame_right, depth_left, depth_right, max_shift=10):
    # Inverted depth interpretation: near=0 -> larger shift
    warped_left = warp_with_depth(frame_left, (255 - depth_left).astype(np.uint8), direction=1, max_shift=max_shift)
    warped_right = warp_with_depth(frame_right, (255 - depth_right).astype(np.uint8), direction=-1, max_shift=max_shift)
    return cv2.addWeighted(warped_left, 0.5, warped_right, 0.5, 0)


def merge_depth_maps(depth0, depth1):
    return cv2.addWeighted(depth0, 0.5, depth1, 0.5, 0)


def save_snapshot(image, output_path):
    cv2.imwrite(output_path, image)


def main():
    camera_indexes = (1, 0)
    cameras = MultiCamera(indexes=camera_indexes, width=1280, height=720)
    print(f"Opened cameras: {camera_indexes}")

    # use_amp=False is more stable.
    # Later, try use_amp=True for more speed.
    depth_estimator = DepthEstimator(use_amp=False)

    calibrator = StereoCalibrator()
    if calibrator.load():
        print("Loaded stereo calibration.")
    else:
        print(
            "No stereo calibration found. Press C in the preview window to collect chessboard pairs, F to finish calibration, or Q to continue without calibration."
        )
        if calibrator.run_interactive(cameras):
            print("Stereo calibration complete and loaded.")
        else:
            print("Continuing without stereo calibration.")

    depth_stack = DepthStack(num_layers=5)
    visualizer = Visualizer()

    print("Press Q to quit, S to save snapshots. Keys: i invert depth, m toggle mapping, k/j change shift.")

    # Interactive controls
    max_shift = 10
    invert_depth = False
    mapping_mode = 0  # 0: use (1-depth), 1: use depth

    while True:
        frames = cameras.get_frames()
        if calibrator.maps is not None:
            frames = calibrator.rectify_frames(frames)

        if frames is None:
            print("Missing frame from one of the cameras.")
            break

        start = time.time()

        results = []
        for frame in frames:
            depth_map = depth_estimator.estimate_depth(frame)
            if depth_map is None:
                print("DepthEstimator returned None for a frame.")
            else:
                # ensure uint8 single-channel
                if depth_map.dtype != np.uint8:
                    try:
                        depth_map = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                    except Exception:
                        depth_map = depth_map.astype(np.uint8)
            
            layers = depth_stack.generate_layers(depth_map)
            results.append((frame, depth_map, layers))

        end = time.time()
        fps = 1.0 / max(end - start, 1e-6)

        # Visualize per-camera depth maps for debugging
        # Depth visualizations with safety checks; save debug images if empty
        try:
            for side, name in enumerate(("Left", "Right")):
                if len(results) <= side or results[side][1] is None:
                    print(f"Depth map missing for camera {side}.")
                    continue
                depth_vis = results[side][1]
                if np.count_nonzero(depth_vis) == 0:
                    print(f"Depth map for camera {side} appears empty; saving debug image.")
                    cv2.imwrite(f"debug_depth_empty_cam{side}.png", depth_vis)
                depth_norm = cv2.normalize(depth_vis, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                cv2.imshow(f"Depth {name} (vis)", cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET))
        except Exception as e:
            print("Error showing depth visualizations:", e)

        merged_depth = merge_depth_maps(results[0][1], results[1][1])
        merged_color = cv2.applyColorMap(merged_depth, cv2.COLORMAP_JET)
        cv2.imshow("Merged 2.5D", merged_color)

        # Create parallax views using current interactive settings
        if invert_depth:
            left_depth = (255 - results[0][1]).astype(np.uint8)
            right_depth = (255 - results[1][1]).astype(np.uint8)
        else:
            left_depth = results[0][1]
            right_depth = results[1][1]

        # mapping_mode selects whether depth is interpreted as near=bright or near=dark
        if mapping_mode == 1:
            # treat depth value directly
            pass
        else:
            # invert mapping so that darker -> smaller shift (use 1 - depth_norm internally)
            pass

        parallax_view = create_parallax_view(
            results[0][0], results[1][0], left_depth, right_depth, max_shift=max_shift
        )
        parallax_view_inv = create_parallax_view_inverted(
            results[0][0], results[1][0], left_depth, right_depth, max_shift=max_shift
        )
        cv2.imshow("Parallax View", parallax_view)
        cv2.imshow("Parallax View (inverted)", parallax_view_inv)

        for idx, (frame, depth_map, layers) in enumerate(results):
            visualizer.show(frame, depth_map, layers, fps, prefix=f"Cam{idx}")

        key = cv2.waitKey(1) & 0xFF
        if key == ord("s"):
            for idx, (_, depth_map, _) in enumerate(results):
                save_snapshot(depth_map, f"snapshot_depth_cam{camera_indexes[idx]}.png")
            save_snapshot(merged_depth, "snapshot_merged_2_5d.png")
            save_snapshot(parallax_view, "snapshot_parallax_view.png")
            save_snapshot(parallax_view_inv, "snapshot_parallax_view_inverted.png")
            print("Saved depth snapshots, merged 2.5D image, and parallax view.")
        elif key == ord("q"):
            break
        elif key == ord("i"):
            invert_depth = not invert_depth
            print(f"invert_depth={invert_depth}")
        elif key == ord("m"):
            mapping_mode = 1 - mapping_mode
            print(f"mapping_mode={mapping_mode}")
        elif key == ord("k"):
            max_shift = min(200, max_shift + 1)
            print(f"max_shift={max_shift}")
        elif key == ord("j"):
            max_shift = max(0, max_shift - 1)
            print(f"max_shift={max_shift}")

    cameras.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
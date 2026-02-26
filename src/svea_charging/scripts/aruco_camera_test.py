#!/usr/bin/env python3
"""
Simple ArUco test utility.

Features:
1. Generate an ArUco marker image.
2. Open the laptop camera and detect ArUco markers in real time.

Usage examples:
    python3 aruco_camera_test.py
    python3 aruco_camera_test.py --marker-id 7 --dictionary DICT_4X4_50
    python3 aruco_camera_test.py --camera-index 1 --no-generate
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def get_aruco_module():
    if not hasattr(cv2, "aruco"):
        raise RuntimeError(
            "OpenCV ArUco module saknas. Installera 'opencv-contrib-python'."
        )
    return cv2.aruco


def get_dictionary(aruco, dictionary_name: str):
    if not hasattr(aruco, dictionary_name):
        available = [name for name in dir(aruco) if name.startswith("DICT_")]
        raise ValueError(
            f"Okänd dictionary '{dictionary_name}'. Exempel: {', '.join(available[:10])}"
        )

    dictionary_id = getattr(aruco, dictionary_name)
    if hasattr(aruco, "getPredefinedDictionary"):
        return aruco.getPredefinedDictionary(dictionary_id)
    return aruco.Dictionary_get(dictionary_id)


def generate_marker(
    aruco,
    dictionary,
    marker_id: int,
    marker_size_px: int,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if hasattr(aruco, "generateImageMarker"):
        marker_img = aruco.generateImageMarker(dictionary, marker_id, marker_size_px)
    else:
        marker_img = aruco.drawMarker(dictionary, marker_id, marker_size_px)

    cv2.imwrite(str(output_path), marker_img)


def create_detector(aruco, dictionary):
    if hasattr(aruco, "DetectorParameters"):
        parameters = aruco.DetectorParameters()
    else:
        parameters = aruco.DetectorParameters_create()

    # Improve robustness for webcam images.
    if hasattr(parameters, "cornerRefinementMethod") and hasattr(aruco, "CORNER_REFINE_SUBPIX"):
        parameters.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX

    if hasattr(aruco, "ArucoDetector"):
        detector = aruco.ArucoDetector(dictionary, parameters)
        return detector, None

    return None, parameters


def detect_markers(aruco, detector, parameters, dictionary, frame):
    if detector is not None:
        return detector.detectMarkers(frame)
    return aruco.detectMarkers(frame, dictionary, parameters=parameters)


def load_calibration(calibration_file: Path | None):
    if calibration_file is None:
        return None, None

    data = np.load(str(calibration_file))
    if "camera_matrix" not in data or "dist_coeffs" not in data:
        raise ValueError(
            "Kalibreringsfilen måste innehålla 'camera_matrix' och 'dist_coeffs' (.npz)."
        )

    camera_matrix = data["camera_matrix"]
    dist_coeffs = data["dist_coeffs"]
    return camera_matrix, dist_coeffs


def get_fallback_camera_matrix(frame_shape, focal_length_px: float | None):
    height, width = frame_shape[:2]
    fx = focal_length_px if focal_length_px is not None else 0.9 * width
    fy = focal_length_px if focal_length_px is not None else 0.9 * width
    cx = width / 2.0
    cy = height / 2.0

    camera_matrix = np.array(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32
    )
    dist_coeffs = np.zeros((5, 1), dtype=np.float32)
    return camera_matrix, dist_coeffs


def rvec_to_euler_deg(rvec):
    rot_mat, _ = cv2.Rodrigues(rvec)
    sy = np.sqrt(rot_mat[0, 0] ** 2 + rot_mat[1, 0] ** 2)
    singular = sy < 1e-6

    if not singular:
        roll = np.arctan2(rot_mat[2, 1], rot_mat[2, 2])
        pitch = np.arctan2(-rot_mat[2, 0], sy)
        yaw = np.arctan2(rot_mat[1, 0], rot_mat[0, 0])
    else:
        roll = np.arctan2(-rot_mat[1, 2], rot_mat[1, 1])
        pitch = np.arctan2(-rot_mat[2, 0], sy)
        yaw = 0.0

    return np.degrees([roll, pitch, yaw])


def estimate_pose_for_markers(aruco, corners, marker_length_m, camera_matrix, dist_coeffs):
    if hasattr(aruco, "estimatePoseSingleMarkers"):
        rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(
            corners, marker_length_m, camera_matrix, dist_coeffs
        )
        return rvecs, tvecs

    raise RuntimeError("Din OpenCV-version saknar estimatePoseSingleMarkers för ArUco.")


def draw_axes(frame, camera_matrix, dist_coeffs, rvec, tvec, axis_length_m):
    if hasattr(cv2, "drawFrameAxes"):
        cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, axis_length_m)


def run_camera_detection(
    aruco,
    dictionary,
    camera_index: int,
    marker_length_m: float | None,
    calibration_file: Path | None,
    focal_length_px: float | None,
) -> None:
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(
            f"Kunde inte öppna kamera med index {camera_index}. Testa t.ex. --camera-index 1"
        )

    detector, parameters = create_detector(aruco, dictionary)
    calibrated_camera_matrix, calibrated_dist_coeffs = load_calibration(calibration_file)
    warned_fallback_intrinsics = False

    print("Kameran startad. Visa en ArUco-marker mot kameran. Tryck 'q' för att avsluta.")
    if marker_length_m is not None:
        if calibrated_camera_matrix is not None:
            print(f"Pose estimation aktiv (markerstorlek: {marker_length_m} m, kalibrerad kamera).")
        else:
            print(
                "Pose estimation aktiv utan kalibrering (approximerade intrinsics). "
                "Avstånd/rotation blir ungefärliga."
            )

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Kunde inte läsa frame från kameran.")
                break

            corners, ids, rejected = detect_markers(
                aruco, detector, parameters, dictionary, frame
            )

            if ids is not None and len(ids) > 0:
                aruco.drawDetectedMarkers(frame, corners, ids)
                ids_list = [int(i) for i in ids.flatten()]
                text = f"Detected IDs: {ids_list}"
                color = (0, 200, 0)

                if marker_length_m is not None:
                    if calibrated_camera_matrix is None:
                        if not warned_fallback_intrinsics:
                            print(
                                "Ingen kalibrering angiven. Använder approximerad kameramodell "
                                "för pose estimation."
                            )
                            warned_fallback_intrinsics = True
                        camera_matrix, dist_coeffs = get_fallback_camera_matrix(
                            frame.shape, focal_length_px
                        )
                    else:
                        camera_matrix, dist_coeffs = (
                            calibrated_camera_matrix,
                            calibrated_dist_coeffs,
                        )

                    rvecs, tvecs = estimate_pose_for_markers(
                        aruco, corners, marker_length_m, camera_matrix, dist_coeffs
                    )

                    for i, marker_id in enumerate(ids.flatten()):
                        rvec = rvecs[i]
                        tvec = tvecs[i]
                        distance_m = float(np.linalg.norm(tvec))
                        roll_deg, pitch_deg, yaw_deg = rvec_to_euler_deg(rvec)
                        draw_axes(
                            frame,
                            camera_matrix,
                            dist_coeffs,
                            rvec,
                            tvec,
                            axis_length_m=max(marker_length_m * 0.5, 0.02),
                        )

                        anchor = corners[i][0][0]
                        x_px, y_px = int(anchor[0]), int(anchor[1])
                        lines = [
                            f"ID {int(marker_id)}: {distance_m:.2f} m",
                            f"R/P/Y: {roll_deg:.0f}/{pitch_deg:.0f}/{yaw_deg:.0f} deg",
                        ]
                        for line_idx, line in enumerate(lines):
                            cv2.putText(
                                frame,
                                line,
                                (x_px, y_px - 12 - line_idx * 20),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (255, 220, 0),
                                2,
                                cv2.LINE_AA,
                            )
            else:
                text = f"No markers | rejected: {len(rejected)}"
                color = (0, 0, 255)

            cv2.putText(
                frame,
                text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2,
                cv2.LINE_AA,
            )

            cv2.imshow("ArUco Camera Test", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


def parse_args():
    parser = argparse.ArgumentParser(description="Generate and detect ArUco markers.")
    parser.add_argument(
        "--dictionary",
        default="DICT_4X4_50",
        help="ArUco dictionary, t.ex. DICT_4X4_50 eller DICT_6X6_250",
    )
    parser.add_argument(
        "--marker-id",
        type=int,
        default=0,
        help="ID på markern som genereras",
    )
    parser.add_argument(
        "--marker-size-px",
        type=int,
        default=400,
        help="Storlek på genererad marker i pixlar",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("aruco_marker.png"),
        help="Sökväg till genererad markerbild",
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="Kameraindex för laptopkamera (oftast 0)",
    )
    parser.add_argument(
        "--no-generate",
        action="store_true",
        help="Skippa generering av marker och starta bara kameradetektion",
    )
    parser.add_argument(
        "--marker-length-m",
        type=float,
        default=None,
        help="Fysisk markerstorlek i meter (t.ex. 0.05) för pose estimation",
    )
    parser.add_argument(
        "--calibration-file",
        type=Path,
        default=None,
        help="Valfri .npz med 'camera_matrix' och 'dist_coeffs' för bättre pose estimation",
    )
    parser.add_argument(
        "--focal-length-px",
        type=float,
        default=None,
        help="Valfri approximerad focal length i pixlar om kalibrering saknas",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    aruco = get_aruco_module()
    dictionary = get_dictionary(aruco, args.dictionary)

    if not args.no_generate:
        generate_marker(
            aruco=aruco,
            dictionary=dictionary,
            marker_id=args.marker_id,
            marker_size_px=args.marker_size_px,
            output_path=args.output,
        )
        print(f"Marker sparad: {args.output.resolve()}")
        print("Skriv ut eller visa bilden på en skärm för att testa detektion.")

    run_camera_detection(
        aruco=aruco,
        dictionary=dictionary,
        camera_index=args.camera_index,
        marker_length_m=args.marker_length_m,
        calibration_file=args.calibration_file,
        focal_length_px=args.focal_length_px,
    )


if __name__ == "__main__":
    main()

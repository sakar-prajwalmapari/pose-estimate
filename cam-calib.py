import cv2
import numpy as np
import math
import argparse

# ---------- Utility Functions ----------
def order_corners(pts):
    pts = np.array(pts, dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype=np.float32)

def find_largest_quad(gray, min_area=10000):
    blurred = cv2.GaussianBlur(gray, (5,5), 0)
    edged = cv2.Canny(blurred, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    edged = cv2.dilate(edged, kernel, iterations=1)
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    for c in contours:
        if cv2.contourArea(c) < min_area:
            break
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            return approx.reshape(4,2)
    return None

def pixel_dist(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

def load_camera_params(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    return data['camera_matrix'], data['dist_coeffs']

def estimate_metric_size(quad_ord, camera_matrix, dist_coeffs, real_w_m, real_h_m):
    objp = np.array([
        [0.0, 0.0, 0.0],
        [real_w_m, 0.0, 0.0],
        [real_w_m, real_h_m, 0.0],
        [0.0, real_h_m, 0.0]
    ], dtype=np.float32)
    imgp = np.array(quad_ord, dtype=np.float32)
    success, rvec, tvec = cv2.solvePnP(objp, imgp, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
    if not success:
        return None
    R, _ = cv2.Rodrigues(rvec)
    center_obj = np.array([[real_w_m/2, real_h_m/2, 0.0]]).T
    center_cam = R @ center_obj + tvec
    dist_center = float(np.linalg.norm(center_cam))
    return dist_center

# ---------- Live Measurement ----------
def main(args):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Camera not found.")
        return

    cam_matrix, dist_coeffs = (None, None)
    if args.cam:
        cam_matrix, dist_coeffs = load_camera_params(args.cam)
        print("[INFO] Camera parameters loaded.")

    real_w_m = args.real_width_mm/1000 if args.real_width_mm else None
    real_h_m = args.real_height_mm/1000 if args.real_height_mm else None

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        quad = find_largest_quad(gray)
        h, w = frame.shape[:2]
        text = "No paper detected"

        if quad is not None:
            quad_ord = order_corners(quad)
            top_w = pixel_dist(quad_ord[0], quad_ord[1])
            bot_w = pixel_dist(quad_ord[3], quad_ord[2])
            left_h = pixel_dist(quad_ord[0], quad_ord[3])
            right_h = pixel_dist(quad_ord[1], quad_ord[2])
            width_px = (top_w + bot_w) / 2
            height_px = (left_h + right_h) / 2

            # Draw
            cv2.polylines(frame, [quad_ord.astype(int)], True, (0,255,0), 2)
            for p in quad_ord.astype(int):
                cv2.circle(frame, tuple(p), 6, (0,0,255), -1)

            text = f"{width_px:.0f}px x {height_px:.0f}px"

            if cam_matrix is not None and real_w_m and real_h_m:
                dist_center = estimate_metric_size(quad_ord, cam_matrix, dist_coeffs, real_w_m, real_h_m)
                if dist_center:
                    text += f" | Dist: {dist_center*100:.1f} cm"

        cv2.putText(frame, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        cv2.imshow("Paper Measurement (Live)", frame)

        k = cv2.waitKey(1) & 0xFF
        if k == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", help="Path to camera calibration .npz")
    ap.add_argument("--real_width_mm", type=float, help="Known real paper width in mm (e.g., 210)")
    ap.add_argument("--real_height_mm", type=float, help="Known real paper height in mm (e.g., 297)")
    args = ap.parse_args()
    main(args)

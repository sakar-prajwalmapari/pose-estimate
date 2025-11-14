import cv2
import numpy as np
import math

def largest_contour(img_bin, min_area=500):
    contours, _ = cv2.findContours(img_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"[DEBUG] Found {len(contours)} total contours")
    if not contours:
        return None
    contours = [c for c in contours if cv2.contourArea(c) >= min_area]
    print(f"[DEBUG] {len(contours)} contours after area filter (>{min_area})")
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    print(f"[DEBUG] Largest contour area: {cv2.contourArea(c):.1f}")
    return c

def pca_orientation(contour):
    pts = contour.reshape(-1, 2).astype(np.float32)
    mean = pts.mean(axis=0)
    cov = np.cov(pts.T)
    eigvals, eigvecs = np.linalg.eig(cov)
    idx = np.argmax(eigvals)
    principal = eigvecs[:, idx]
    angle = math.degrees(math.atan2(principal[1], principal[0]))
    print(f"[DEBUG] PCA centroid=({mean[0]:.1f},{mean[1]:.1f}), angle={angle:.2f}°")
    return mean, principal, angle

def detect_bottle_pose(img_bgr, debug=False):
    img = img_bgr.copy()
    h, w = img.shape[:2]
    print(f"\n[DEBUG] Processing frame size: {w}x{h}")

    # --- Stage 1: Preprocess ---
    blur = cv2.GaussianBlur(img, (7,7), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    # --- Stage 2: Segmentation ---
    lower = np.array([0, 30, 30])
    upper = np.array([180, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    edges = cv2.Canny(cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY), 60, 180)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15,15), np.uint8))
    combined_mask = cv2.bitwise_or(mask, edges)

    # --- Stage 3: Largest contour ---
    cnt = largest_contour(combined_mask, min_area=1000)
    contour_view = img.copy()
    if cnt is not None:
        cv2.drawContours(contour_view, [cnt], -1, (0,255,255), 2)

    # --- Display all intermediate results ---
    window_w, window_h = 480, 360
    layout = {
        "Original":  (0,   0),
        "Blur":      (480, 0),
        "Edges":     (960, 0),
        "Mask":      (0,   480),
        "Combined Mask": (480, 480),
        "Contours":  (960, 480)
    }
    windows = {
        "Original": img,
        "Blur": blur,
        "Edges": edges,
        "Mask": mask,
        "Combined Mask": combined_mask,
        "Contours": contour_view
    }

    for name, (x, y) in layout.items():
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(name, window_w, window_h)
        cv2.moveWindow(name, x, y)
        cv2.imshow(name, windows[name])

    if cnt is None:
        print("[DEBUG] No valid contour found.")
        return None

    # --- Pose extraction ---
    result = {}
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    box = box.astype(int)
    result['minAreaRect'] = (rect, box)
    print(f"[DEBUG] MinAreaRect center=({rect[0][0]:.1f},{rect[0][1]:.1f}), angle={rect[2]:.1f}")

    if len(cnt) >= 5:
        ellipse = cv2.fitEllipse(cnt)
        result['ellipse'] = ellipse
        center, axes, ang = ellipse
        print(f"[DEBUG] Ellipse center=({center[0]:.1f},{center[1]:.1f}), axes=({axes[0]:.1f},{axes[1]:.1f}), angle={ang:.1f}")

    centroid, principal_vec, angle = pca_orientation(cnt)
    result['centroid'] = (float(centroid[0]), float(centroid[1]))
    result['angle_deg'] = angle
    result['principal_vec'] = (float(principal_vec[0]), float(principal_vec[1]))

    pts = cnt.reshape(-1, 2).astype(np.float32)
    centered = pts - centroid
    proj = centered.dot(principal_vec)
    min_idx = np.argmin(proj)
    max_idx = np.argmax(proj)
    top_pt = tuple(pts[max_idx].astype(int))
    bottom_pt = tuple(pts[min_idx].astype(int))
    result['top_pt'] = top_pt
    result['bottom_pt'] = bottom_pt
    print(f"[DEBUG] Top point={top_pt}, Bottom point={bottom_pt}")

    if debug:
        out = img.copy()
        cv2.drawContours(out, [box], 0, (0,255,0), 2)

        if 'ellipse' in result:
            center, axes, ang = result['ellipse']
            center = (int(round(center[0])), int(round(center[1])))
            axes = (int(round(axes[0]/2)), int(round(axes[1]/2)))
            cv2.ellipse(out, center, axes, float(ang), 0, 360, (255,0,0), 2)

        cx, cy = map(int, result['centroid'])
        cv2.circle(out, (cx,cy), 4, (0,0,255), -1)

        pv = np.array(result['principal_vec'])
        pt1 = (int(cx - pv[0]*150), int(cy - pv[1]*150))
        pt2 = (int(cx + pv[0]*150), int(cy + pv[1]*150))
        cv2.line(out, pt1, pt2, (255,0,255), 2)

        cv2.circle(out, top_pt, 6, (0,255,255), -1)
        cv2.circle(out, bottom_pt, 6, (0,255,255), -1)
        cv2.putText(out, f"Angle: {angle:.1f} deg", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
        return result, out
    else:
        return result

if __name__ == "__main__":
    import sys
    path = 0  # webcam by default
    if len(sys.argv) > 1:
        path = sys.argv[1]

    cap = cv2.VideoCapture(path)
    frame_num = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[DEBUG] No frame captured — end of video or camera not ready.")
            break
        frame_num += 1
        print(f"\n========== FRAME {frame_num} ==========")
        res = detect_bottle_pose(frame, debug=True)
        if res is None:
            cv2.putText(frame, "No bottle found", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
            cv2.imshow("Bottle Pose", frame)
        else:
            result, out = res
            cv2.imshow("Bottle Pose", out)
            #cv2.moveWindow("Bottle Pose", 0, 800)
            cv2.resizeWindow("Bottle Pose", 960, 540)

        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            print("[DEBUG] ESC pressed — exiting.")
            break

    cap.release()
    cv2.destroyAllWindows()

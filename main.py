import cv2
import torch
import numpy as np
from ultralytics import YOLO

# Load YOLO model
yolo = YOLO("yolov8n.pt")

# Load MiDaS model
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
midas.eval()

# Load MiDaS transforms
transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = transforms.small_transform

# Read image
img = cv2.imread("image.jpg")

if img is None:
    print("Image not found!")
    exit()

# ---------- MiDaS ----------
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
input_batch = transform(img_rgb)

with torch.no_grad():
    prediction = midas(input_batch)

    prediction = torch.nn.functional.interpolate(
        prediction.unsqueeze(1),
        size=img_rgb.shape[:2],
        mode="bicubic",
        align_corners=False
    ).squeeze()

depth_map = prediction.cpu().numpy()

# Normalize depth map
depth_normalized = cv2.normalize(
    depth_map,
    None,
    0,
    255,
    cv2.NORM_MINMAX
).astype(np.uint8)

depth_colored = cv2.applyColorMap(
    depth_normalized,
    cv2.COLORMAP_JET
)

# ---------- YOLO ----------
results = yolo(img)

for r in results:
    for box in r.boxes:

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        class_id = int(box.cls[0])
        class_name = r.names[class_id]

        roi = depth_normalized[y1:y2, x1:x2]

        if roi.size > 0:
            avg_depth = roi.mean()

            # Approximate distance
            distance = 3 - (avg_depth / 255) * 3

            label = f"{class_name}: {distance:.2f} m"

            cv2.rectangle(img, (x1, y1), (x2, y2),
                          (0, 255, 0), 2)

            cv2.putText(
                img,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

# Display outputs
cv2.imshow("YOLO + Distance", img)
cv2.imshow("Depth Map", depth_colored)

cv2.waitKey(0)
cv2.destroyAllWindows()
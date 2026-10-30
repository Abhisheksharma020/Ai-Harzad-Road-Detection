import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import time
import argparse
from pathlib import Path


# ============================================================
# 1. CONFIGURATION — matches your training notebook
# ============================================================

MODEL_PATH = "road_model.pth"

CLASSES = (
    "Broken Road Sign Issues",
    "Damaged Road issues",
    "Illegal Parking Issues",
    "Mixed Issues",
    "Pothole Issues"
)

MEAN = (0.4581, 0.4506, 0.4425)
STD = (0.2835, 0.2802, 0.2848)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ============================================================
# 2. CNN ARCHITECTURE — exactly the same as training
# ============================================================

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2, 2),

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2, 2),

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2, 2),
        )

        self.pool = nn.AdaptiveAvgPool2d((4, 4))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x


# ============================================================
# 3. LOAD MODEL
# ============================================================

model = SimpleCNN(num_classes=len(CLASSES))

state_dict = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=True
)

model.load_state_dict(state_dict)
model.to(device)
model.eval()

print("Model loaded successfully.")
print("Classes:", len(CLASSES))


# ============================================================
# 4. PREPROCESSING — same as validation/test
# ============================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD)
])


# ============================================================
# 5. PREDICTION FUNCTION
# ============================================================

def predict_pil(image):
    """
    Input:
        PIL RGB image

    Returns:
        predicted class
        confidence
        all class probabilities
    """

    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.inference_mode():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, dim=1)

    predicted_class = CLASSES[predicted.item()]
    confidence_value = confidence.item() * 100

    all_probs = {
        CLASSES[i]: probabilities[0, i].item() * 100
        for i in range(len(CLASSES))
    }

    return predicted_class, confidence_value, all_probs


# ============================================================
# 6. SINGLE IMAGE TEST
# ============================================================

def test_image(image_path):

    image_path = Path(image_path)

    if not image_path.exists():
        print("ERROR: Image not found:", image_path)
        return

    image = Image.open(image_path).convert("RGB")

    start = time.perf_counter()

    predicted_class, confidence, probabilities = predict_pil(image)

    elapsed = time.perf_counter() - start

    print("\n========== IMAGE RESULT ==========")
    print("Image:", image_path)
    print("Prediction:", predicted_class)
    print(f"Confidence: {confidence:.2f}%")
    print(f"Inference time: {elapsed * 1000:.2f} ms")

    print("\nAll class probabilities:")

    for cls, prob in sorted(
        probabilities.items(),
        key=lambda x: x[1],
        reverse=True
    ):
        print(f"{cls:30s}: {prob:6.2f}%")

    print("==================================\n")


# ============================================================
# 7. DRAW PREDICTION ON FRAME
# ============================================================

def process_frame(frame, confidence_threshold=60):

    # OpenCV BGR -> RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    pil_image = Image.fromarray(rgb)

    start = time.perf_counter()

    predicted_class, confidence, _ = predict_pil(pil_image)

    inference_time = (time.perf_counter() - start) * 1000

    if confidence < confidence_threshold:
        display_class = "UNCERTAIN"
    else:
        display_class = predicted_class

    return display_class, confidence, inference_time


# ============================================================
# 8. DISPLAY INFORMATION ON FRAME
# ============================================================

def draw_result(
    frame,
    predicted_class,
    confidence,
    inference_time,
    fps
):

    # Background rectangle
    cv2.rectangle(
        frame,
        (10, 10),
        (650, 115),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        frame,
        f"Prediction: {predicted_class}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.70,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"Confidence: {confidence:.1f}%",
        (20, 68),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Inference: {inference_time:.1f} ms | FPS: {fps:.1f}",
        (20, 96),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1
    )

    return frame


# ============================================================
# 9. REAL-TIME WEBCAM TEST
# ============================================================

def webcam_test(camera_index=0, confidence_threshold=60):

    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        return

    print("\n================================")
    print("WEBCAM TEST STARTED")
    print("Press Q to quit.")
    print("Press S to save the current frame.")
    print("================================\n")

    frame_count = 0
    fps_start = time.perf_counter()
    fps = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            print("ERROR: Could not read webcam frame.")
            break

        predicted_class, confidence, inference_time = process_frame(
            frame,
            confidence_threshold
        )

        # FPS calculation
        frame_count += 1

        elapsed = time.perf_counter() - fps_start

        if elapsed >= 1.0:
            fps = frame_count / elapsed
            frame_count = 0
            fps_start = time.perf_counter()

        frame = draw_result(
            frame,
            predicted_class,
            confidence,
            inference_time,
            fps
        )

        cv2.imshow(
            "Road Issue CNN - Webcam",
            frame
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        if key == ord("s"):
            filename = f"webcam_frame_{int(time.time())}.jpg"
            cv2.imwrite(filename, frame)
            print("Saved:", filename)

    cap.release()
    cv2.destroyAllWindows()


# ============================================================
# 10. RECORDED VIDEO TEST
# ============================================================

def video_test(video_path, confidence_threshold=60):

    video_path = Path(video_path)

    if not video_path.exists():
        print("ERROR: Video not found:", video_path)
        return

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print("ERROR: Could not open video.")
        return

    # Video information
    original_fps = cap.get(cv2.CAP_PROP_FPS)

    if original_fps <= 0:
        original_fps = 30

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print("\n================================")
    print("RECORDED VIDEO TEST STARTED")
    print("Video:", video_path)
    print(f"Resolution: {width} x {height}")
    print(f"Original FPS: {original_fps:.2f}")
    print("Press Q to quit.")
    print("Press S to save the current frame.")
    print("================================\n")

    frame_count = 0
    fps_start = time.perf_counter()
    display_fps = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            print("\nVideo finished.")
            break

        predicted_class, confidence, inference_time = process_frame(
            frame,
            confidence_threshold
        )

        # Calculate processing FPS
        frame_count += 1

        elapsed = time.perf_counter() - fps_start

        if elapsed >= 1.0:
            display_fps = frame_count / elapsed
            frame_count = 0
            fps_start = time.perf_counter()

        frame = draw_result(
            frame,
            predicted_class,
            confidence,
            inference_time,
            display_fps
        )

        cv2.imshow(
            "Road Issue CNN - Recorded Video",
            frame
        )

        # Wait according to original video FPS
        delay = max(1, int(1000 / original_fps))

        key = cv2.waitKey(delay) & 0xFF

        if key == ord("q"):
            break

        if key == ord("s"):
            filename = f"video_frame_{int(time.time())}.jpg"
            cv2.imwrite(filename, frame)
            print("Saved:", filename)

    cap.release()
    cv2.destroyAllWindows()


# ============================================================
# 11. MAIN PROGRAM
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Road Issue CNN - Image, Webcam and Video Testing"
    )

    # Image mode
    parser.add_argument(
        "--image",
        type=str,
        help="Path to a single image"
    )

    # Webcam mode
    parser.add_argument(
        "--webcam",
        action="store_true",
        help="Start webcam testing"
    )

    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera index, default = 0"
    )

    # Video mode
    parser.add_argument(
        "--video",
        type=str,
        help="Path to recorded video"
    )

    # Confidence
    parser.add_argument(
        "--threshold",
        type=float,
        default=60,
        help="Confidence threshold, default = 60"
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # IMAGE
    # --------------------------------------------------------

    if args.image:

        test_image(args.image)

    # --------------------------------------------------------
    # WEBCAM
    # --------------------------------------------------------

    elif args.webcam:

        webcam_test(
            camera_index=args.camera,
            confidence_threshold=args.threshold
        )

    # --------------------------------------------------------
    # RECORDED VIDEO
    # --------------------------------------------------------

    elif args.video:

        video_test(
            video_path=args.video,
            confidence_threshold=args.threshold
        )

    # --------------------------------------------------------
    # NOTHING SELECTED
    # --------------------------------------------------------

    else:

        print("\n==========================================")
        print("ROAD ISSUE CNN TESTING")
        print("==========================================")

        print("\n1. Test an image:")
        print("   python realtime_test.py --image test_images/road.jpg")

        print("\n2. Test webcam:")
        print("   python realtime_test.py --webcam")

        print("\n3. Test recorded video:")
        print("   python realtime_test.py --video videos/road_test.mp4")

        print("\n4. Custom confidence threshold:")
        print("   python realtime_test.py --video videos/road_test.mp4 --threshold 70")

        print("\n==========================================")

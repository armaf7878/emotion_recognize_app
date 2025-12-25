import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms

import numpy as np
from mtcnn import MTCNN
from PIL import Image, ImageTk

import tkinter as tk
from tkinter import filedialog, messagebox


MODEL_PATH = "./models/resnet18_gray2.pth"

EMOTION_CLASSES = [
    # "anger","angry", "disgust" ,"disgusted", "fear", "fearful", "happy", "neutral",
    # "sad", "surprise", "surprised"
    "angry","disgusted","fearful", "happy", "neutral",
    "sad", "surprised"
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class ResNet18Gray(nn.Module):

    def __init__(self, num_classes=7):
        super().__init__()
        self.model = models.resnet18(weights=None)
        self.model.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)


def load_emotion_model(model_path: str):
    model = ResNet18Gray(num_classes=len(EMOTION_CLASSES))
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


face_transform = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])          



class EmotionApp:

    def __init__(self, root):
        self.root = root
        self.root.title("Face Emotion Detector Nhom 05")
        self.root.geometry("1000x700")
        self.root.configure(bg="#ffffff")

        try:
            self.model = load_emotion_model(MODEL_PATH)
        except Exception as e:
            messagebox.showerror("Error", f"Không thể load model:\n{e}")
            raise

        self.detector = MTCNN()
        self.cap = None
        self.is_camera_running = False

        self._build_ui()


    def _build_ui(self):
   
        header = tk.Frame(self.root, bg="#ffffff")
        header.pack(side=tk.TOP, fill=tk.X, pady=(10, 0))

        title_label = tk.Label(
            header,
            text="Face Emotion Recognition",
            font=("Segoe UI", 18, "bold"),
            bg="#ffffff",
            fg="#D32F2F",
        )               
        title_label.pack(side=tk.LEFT, padx=20)

        subtitle_label = tk.Label(
            header,
            text=" Đừng quá bối rối khi nhận diện cảm xúc chưa đúng, vì cảm xúc con người là thứ khó hiểu nhất trên đời...",
            font=("Segoe UI", 10),
            bg="#ffffff",
            fg="#555555",
        )
        subtitle_label.pack(side=tk.LEFT, padx=10, pady=(6, 0))

        content = tk.Frame(self.root, bg="#ffffff")
        content.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=15, pady=10)

        video_frame = tk.Frame(content, bg="#f5f5f5", bd=1, relief=tk.SOLID)
        video_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        self.video_label = tk.Label(video_frame, bg="#000000")
        self.video_label.pack(fill=tk.BOTH, expand=True)

        right_panel = tk.Frame(content, bg="#ffffff")
        right_panel.pack(side=tk.RIGHT, fill=tk.Y)

        btn_frame = tk.LabelFrame(
            right_panel,
            text="Chọn chức năng",
            bg="#ffffff",
            fg="#D32F2F",
            font=("Segoe UI", 10, "bold")
        )
        btn_frame.pack(fill=tk.X, pady=(0, 10))

        btn_style = {
            "font": ("Segoe UI", 10, "bold"),
            "bg": "#D32F2F",
            "fg": "#ffffff",
            "activebackground": "#B71C1C",
            "activeforeground": "#ffffff",
            "relief": tk.FLAT,
            "bd": 0,
            "padx": 12,
            "pady": 6,
            "cursor": "hand2",
        }

        tk.Button(btn_frame, text="Mở camera",
                  command=self.start_camera, **btn_style).grid(row=0, column=0, padx=5, pady=5)

        tk.Button(btn_frame, text="Tắt camera",
                  command=self.stop_camera, **btn_style).grid(row=0, column=1, padx=5, pady=5)

        tk.Button(btn_frame, text="Tải ảnh lên",
                  command=self.upload_image, **btn_style).grid(row=1, column=0, padx=5, pady=5)

        tk.Button(btn_frame, text="Gỡ ảnh",
                  command=self.clear_display, **btn_style).grid(row=1, column=1, padx=5, pady=5)

        tk.Button(btn_frame, text="Thoát",
                  command=self.quit_app, **btn_style).grid(row=2, column=0, columnspan=2, padx=5, pady=(5, 10))

        status_box = tk.LabelFrame(
            right_panel,
            text="Thông tin",
            bg="#ffffff",
            fg="#D32F2F",
            font=("Segoe UI", 10, "bold")
        )
        status_box.pack(fill=tk.BOTH, expand=True)

        self.device_label = tk.Label(
            status_box,
            text=f"Thiết bị: {'GPU' if torch.cuda.is_available() else 'CPU'}",
            bg="#ffffff",
            fg="#333333",
            font=("Segoe UI", 10),
            anchor="w",
        )
        self.device_label.pack(fill=tk.X, padx=10, pady=(8, 4))

        self.face_count_label = tk.Label(
            status_box,
            text="Số khuôn mặt: 0",
            bg="#ffffff",
            fg="#333333",
            font=("Segoe UI", 10),
            anchor="w",
        )
        self.face_count_label.pack(fill=tk.X, padx=10, pady=(0, 8))

        self.hint_label = tk.Label(
            status_box,
            text="Gợi ý:\n• Dùng webcam hoặc upload ảnh\n• Hệ thống sẽ vẽ khung + emotion\n• Hỗ trợ nhiều khuôn mặt cùng lúc",
            bg="#ffffff",
            fg="#777777",
            font=("Segoe UI", 9),
            justify=tk.LEFT,
            anchor="nw",
        )
        self.hint_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

    def start_camera(self):
        if self.is_camera_running:
            return

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Không mở được camera!")
            return

        self.is_camera_running = True
        self._update_frame()

    def stop_camera(self):
        self.is_camera_running = False
        if self.cap:
            self.cap.release()
            self.cap = None

    def quit_app(self):
        self.stop_camera()
        self.root.destroy()

    def _update_frame(self):
        if not self.is_camera_running or self.cap is None:
            return

        ret, frame = self.cap.read()
        if not ret:
            self.stop_camera()
            return

        frame = cv2.flip(frame, 1)

        annotated, faces = self.process_and_draw(frame)
        self.face_count_label.config(text=f"Phát hiện khuôn mặt: {len(faces)}")

        self.show_frame(annotated)

        self.root.after(30, self._update_frame)

    def upload_image(self):
        self.stop_camera()

        file_path = filedialog.askopenfilename(
            title="Chọn ảnh",
            filetypes=[("Ảnh", "*.jpg;*.jpeg;*.png;*.bmp")]
        )
        if not file_path:
            return

        img = cv2.imread(file_path)
        if img is None:
            messagebox.showerror("Error", "Không đọc được ảnh!")
            return

        annotated, faces = self.process_and_draw(img)
        self.face_count_label.config(text=f"Số khuôn mặt: {len(faces)}")

        self.show_frame(annotated)

        if len(faces) == 0:
            messagebox.showinfo("Info", "Không phát hiện khuôn mặt nào trong ảnh này.")

    def clear_display(self):
        self.stop_camera()
        self.video_label.config(image="")
        self.face_count_label.config(text="Số khuôn mặt phát hiện: 0")


    def show_frame(self, frame_bgr):

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w = frame_rgb.shape[:2]

        max_w, max_h = 800, 550
        scale = min(max_w / w, max_h / h, 1.0)  # không upscale >1 để giữ nét

        new_w, new_h = int(w * scale), int(h * scale)
        frame_rgb = cv2.resize(frame_rgb, (new_w, new_h))

        img = Image.fromarray(frame_rgb)
        img_tk = ImageTk.PhotoImage(img)

        self.video_label.img_tk = img_tk
        self.video_label.config(image=img_tk)


    def process_and_draw(self, frame_bgr):

        frame_draw = frame_bgr.copy()
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        detections = self.detector.detect_faces(frame_rgb)
        faces = []

        h_img, w_img, _ = frame_rgb.shape

        for det in detections:
            box = det.get("box")
            if box is None:
                continue

            x, y, w, h = box
            x, y = max(0, x), max(0, y)
            x2, y2 = x + w, y + h

            x2 = min(x2, w_img)
            y2 = min(y2, h_img)

            if x2 <= x or y2 <= y:
                continue

            face_rgb = frame_rgb[y:y2, x:x2]
            if face_rgb.size == 0:
                continue

            emotion, conf = self.predict_emotion(face_rgb)

            faces.append((x, y, w, h, emotion, conf))

            color = (0, 0, 255) 
            cv2.rectangle(frame_draw, (x, y), (x2, y2), color, 2)

            text = f"{emotion} ({conf:.1f}%)"
            (tw, th), baseline = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )

            text_x1 = x
            text_y1 = y - th - 6
            text_x2 = x + tw
            text_y2 = y

            if text_y1 < 0:
                text_y1 = y2 + 6
                text_y2 = y2 + th + 6

            text_y1 = max(0, text_y1)
            text_y2 = min(h_img - 1, text_y2)

            cv2.rectangle(
                frame_draw,
                (text_x1, text_y1),
                (text_x2, text_y2),
                color,
                thickness=-1,
            )

            text_org_y = text_y2 - 4
            cv2.putText(
                frame_draw,
                text,
                (text_x1 + 2, text_org_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        return frame_draw, faces

    def predict_emotion(self, face_rgb: np.ndarray):

        face_pil = Image.fromarray(face_rgb).convert("L")

        img_t = face_transform(face_pil).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = self.model(img_t)
            probs = torch.softmax(logits, dim=1)[0]
            conf, idx = torch.max(probs, dim=0)

        return EMOTION_CLASSES[int(idx)], float(conf.item() * 100.0)


if __name__ == "__main__":
    root = tk.Tk()
    app = EmotionApp(root)
    root.mainloop()

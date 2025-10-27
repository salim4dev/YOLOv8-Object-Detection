import cv2
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
from ultralytics import YOLO
import numpy as np

import torch
print(torch.__version__)
print(torch.cuda.is_available())


class YOLOApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🧠 YOLOv8 Object Tracker")
        self.root.geometry("900x600")
        self.root.configure(bg="#1e1e1e")

       #Depends on your webcam
        #self.cap = cv2.VideoCapture(0)
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.model = YOLO("yolov8n.pt")  # Lightweight model
        self.effect = None

        self.detect_enabled = tk.BooleanVar(value=True)
        self.detect_person = tk.BooleanVar(value=True)
        self.detect_object = tk.BooleanVar(value=True)
        self.detect_obstacle = tk.BooleanVar(value=True)

        self.canvas = tk.Canvas(root, width=800, height=500, bg="black", highlightthickness=0)
        self.canvas.pack(pady=10)

        self.status = tk.Label(root, text="Status: Running YOLOv8", fg="white", bg="#1e1e1e", font=("Arial", 12))
        self.status.pack()

        self.menu_bar()
        self.update_frame()

        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def menu_bar(self):
        menubar = tk.Menu(self.root)

        effects_menu = tk.Menu(menubar, tearoff=0)
        effects_menu.add_command(label="Grayscale", command=lambda: self.set_effect("gray"))
        effects_menu.add_command(label="Edge Detection", command=lambda: self.set_effect("edge"))
        effects_menu.add_command(label="Sepia", command=lambda: self.set_effect("sepia"))
        effects_menu.add_command(label="Invert", command=lambda: self.set_effect("invert"))
        effects_menu.add_command(label="Blur", command=lambda: self.set_effect("blur"))
        effects_menu.add_command(label="Sketch", command=lambda: self.set_effect("sketch"))
        effects_menu.add_command(label="Emboss", command=lambda: self.set_effect("emboss"))
        effects_menu.add_command(label="Pixelate", command=lambda: self.set_effect("pixel"))
        effects_menu.add_command(label="Contrast Boost", command=lambda: self.set_effect("contrast"))
        effects_menu.add_command(label="None", command=lambda: self.set_effect(None))
        menubar.add_cascade(label="Effects", menu=effects_menu)

        toggle_menu = tk.Menu(menubar, tearoff=0)
        toggle_menu.add_checkbutton(label="Enable Detection", variable=self.detect_enabled)
        toggle_menu.add_checkbutton(label="Detect Person", variable=self.detect_person)
        toggle_menu.add_checkbutton(label="Detect Object", variable=self.detect_object)
        toggle_menu.add_checkbutton(label="Detect Obstacle", variable=self.detect_obstacle)
        menubar.add_cascade(label="Detection", menu=toggle_menu)

        camera_menu = tk.Menu(menubar, tearoff=0)
        camera_menu.add_command(label="Capture Frame", command=self.capture_frame)
        camera_menu.add_command(label="Exit", command=self.root.quit)
        menubar.add_cascade(label="Camera", menu=camera_menu)

        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="About", command=lambda: messagebox.showinfo("About", "YOLOv8 Tracker with Effects\nMade with ❤️ in Python"))
        menubar.add_cascade(label="Help", menu=help_menu)

        self.root.config(menu=menubar)

    def set_effect(self, effect):
        self.effect = effect

    def apply_effects(self, frame):
        if self.effect == "gray":
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif self.effect == "edge":
            edges = cv2.Canny(frame, 100, 200)
            frame = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        elif self.effect == "sepia":
            kernel = np.array([[0.272, 0.534, 0.131],
                               [0.349, 0.686, 0.168],
                               [0.393, 0.769, 0.189]])
            frame = cv2.transform(frame, kernel)
            frame = np.clip(frame, 0, 255)
        elif self.effect == "invert":
            frame = cv2.bitwise_not(frame)
        elif self.effect == "blur":
            frame = cv2.GaussianBlur(frame, (15, 15), 0)
        elif self.effect == "sketch":
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            inv = cv2.bitwise_not(gray)
            blur = cv2.GaussianBlur(inv, (21, 21), 0)
            sketch = cv2.divide(gray, 255 - blur, scale=256)
            frame = cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)
        elif self.effect == "emboss":
            kernel = np.array([[0,-1,-1],[1,0,-1],[1,1,0]])
            frame = cv2.filter2D(frame, -1, kernel)
        elif self.effect == "pixel":
            h, w = frame.shape[:2]
            temp = cv2.resize(frame, (w//20, h//20), interpolation=cv2.INTER_LINEAR)
            frame = cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)
        elif self.effect == "contrast":
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            cl = clahe.apply(l)
            frame = cv2.merge((cl, a, b))
            frame = cv2.cvtColor(frame, cv2.COLOR_LAB2BGR)
        return frame

    def capture_frame(self):
        if hasattr(self, 'frame'):
            filename = f"capture_{cv2.getTickCount()}.jpg"
            cv2.imwrite(filename, self.frame)
            messagebox.showinfo("Capture", f"Saved as {filename}")

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, -1) # 1 or -1 depends on what you chosed in cam capture
            frame = cv2.resize(frame, (800, 500))
            self.frame = frame.copy()

            if self.detect_enabled.get():
                results = self.model(frame, verbose=False)[0]
                for box in results.boxes:
                    cls = int(box.cls[0])
                    label = self.model.names[cls].lower()

                    if (label == "person" and not self.detect_person.get()) or \
                       (label in ["car", "truck", "bus"] and not self.detect_obstacle.get()) or \
                       (label not in ["person", "car", "truck", "bus"] and not self.detect_object.get()):
                        continue

                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = box.conf[0]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            frame = self.apply_effects(frame)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            imgtk = ImageTk.PhotoImage(image=img)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
            self.canvas.imgtk = imgtk

        self.root.after(10, self.update_frame)

    def on_closing(self):
        self.cap.release()
        self.root.destroy()

    def run(self):
        self.root.mainloop()

# 🚀 Launch
if __name__ == "__main__":
    root = tk.Tk()
    app = YOLOApp(root)
    app.run()
import cv2
import tkinter as tk
from PIL import Image, ImageTk
from ultralytics import YOLO


class YOLOApp:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLOv8 - Object Detection")
        self.root.geometry("900x600")
        self.root.configure(bg="#1e1e1e")

        # Capture caméra
        # Depends on your webcam
        # self.cap = cv2.VideoCapture(0)
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.model = YOLO("yolov8n.pt")  # Modèle léger

        # Canvas d'affichage
        self.canvas = tk.Canvas(root, width=800, height=500, bg="black", highlightthickness=0)
        self.canvas.pack(pady=10)

       # self.status = tk.Label(root, text="Status: YOLOv8 en cours...", fg="white", bg="#1e1e1e", font=("Arial", 12))
        #self.status.pack()

        self.update_frame()
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1) # 1 or -1 depends on what you chosed in cam capture
            frame = cv2.resize(frame, (900, 600))
            self.frame = frame.copy()

            # Détection
            results = self.model(frame, verbose=False)[0]
            for box in results.boxes:
                cls = int(box.cls[0])
                label = self.model.names[cls]
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Conversion pour tkinter
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


if __name__ == "__main__":
    root = tk.Tk()
    app = YOLOApp(root)
    app.run()

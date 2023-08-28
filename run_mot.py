from ultralytics import YOLO
from boxmot import StrongSORT
from pathlib import Path
from time import perf_counter
import cv2
import numpy as np
import torch

class Colors:
    def __init__(self, num_colors=80):
        self.num_colors = num_colors
        self.color_palette = self.generate_color_palette()

    def generate_color_palette(self):
        hsv_palette = np.zeros((self.num_colors, 1, 3), dtype=np.uint8)
        hsv_palette[:, 0, 0] = np.linspace(0, 180, self.num_colors, endpoint=False)
        hsv_palette[:, :, 1:] = 255
        bgr_palette = cv2.cvtColor(hsv_palette, cv2.COLOR_HSV2BGR)
        return bgr_palette.reshape(-1, 3)

    def __call__(self, class_id):
        color = tuple(map(int, self.color_palette[class_id]))
        return color
    
class ObjectDetection:
    def __init__(self, model_weights="yolov8s.pt", capture_index=0):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Using Device: ", self.device)
        self.model = self.load_model(model_weights)
        self.classes = self.model.names
        self.classes[0] = 'person'
        self.colors = Colors(len(self.classes))
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.capture_index = capture_index
        self.cap = self.load_capture()
        reid_weights = Path("osnet_x0_25_msmt17.pt")
        self.tracker = StrongSORT(reid_weights,
                                  torch.device(self.device),
                                  fp16 = False,
                                  )

    def load_model(self, weights):
        model = YOLO(weights)
        model.fuse()
        return model

    def predict(self, frame):
        results = self.model(frame, stream=True, verbose=False, conf=0.45, line_width=1)
        return results

    def draw_tracks(self, frame, tracks):
        for track in tracks:
            x1, y1, x2, y2 = int(track[0]), int(track[1]), int(track[2]), int(track[3])
            id = int(track[4])
            conf = track[5]
            class_id = int(track[6])
            class_name = self.classes[class_id]
            cv2.rectangle(frame, (x1,y1), (x2, y2), self.colors(class_id), 2)
            label = f'ID={id}'
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(frame, (x1, y1-h-15), (x1+w, y1), self.colors(class_id), -1)
            cv2.putText(frame, label, (x1,y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255) , 2)
        return frame

    def load_capture(self):
        cap = cv2.VideoCapture(self.capture_index)
        assert cap.isOpened()
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        #writer = cv2.VideoWriter(fr'output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        return cap

    def __call__(self):
        tracker = self.tracker
        while True:
            start_time = perf_counter()
            ret, frame = self.cap.read()
            if not ret:
                break
            cv2.rectangle(frame, (0,30), (220,80), (255,255,255),-1 )
            detections = self.predict(frame)
            for dets in detections:
                tracks = tracker.update(dets.boxes.data.to("cpu").numpy(), frame)
                if len(tracks.shape) == 2 and tracks.shape[1]==7:
                    frame = self.draw_tracks(frame, tracks)
            end_time = perf_counter()
            fps = 1/np.round(end_time - start_time, 2)
            cv2.putText(frame, f'FPS: {int(fps)}', (20,70), self.font, 1.5, (0,255,0), 2)
            #self.writer.write(frame)
            cv2.imshow('YOLOv8 Tracking', frame)
            if cv2.waitKey(5) & 0xFF == 27:
                break
        self.cap.release()
        cv2.destroyAllWindows()

test_vid = r"test_vid\video1.mp4"
model_weights = "yolov8n.pt"
detector = ObjectDetection(model_weights, test_vid)
detector()
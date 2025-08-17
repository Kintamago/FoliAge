#!/usr/bin/env python3
import json
import threading
import time
import queue

import cv2
import numpy as np
import pynmea2
import serial
from picamera2 import Picamera2
from ultralytics import YOLO
import supervision as sv

# ========= CONFIG =========
MODEL_PATH  = "yolov8n.pt"#/home/zicongyu/yolo/runs/detect/train/weights/best_ncnn_model"
CONF_THRESH = 0.50
FRAME_SIZE  = (1280, 720)         
JSONL_PATH  = "detections.jsonl"

GPS_PORTS   = [
    "/dev/serial/by-id/usb-u-blox_AG_-_www.u-blox.com_u-blox_7_-_GPS_GNSS_Receiver-if00",
    "/dev/ttyACM0"
]
GPS_BAUD    = 9600
# ==========================

# ---- Minimal GPS thread (lat/lon only) ----
class GPSReader(threading.Thread):
    def __init__(self, ports, baud=9600):
        super().__init__(daemon=True)
        self.ports = ports
        self.baud = baud
        self.lock = threading.Lock()
        self.lat = None
        self.lon = None
        self._stop = False

    def run(self):
        ser = None
        while not self._stop:
            try:
                if ser is None or not ser.is_open:
                    ser = None
                    for p in self.ports:
                        try:
                            ser = serial.Serial(p, self.baud, timeout=1)
                            break
                        except Exception:
                            ser = None
                    if ser is None:
                        time.sleep(1)
                        continue

                line = ser.readline().decode(errors="ignore").strip()
                if not line.startswith("$"):
                    continue

                if line.startswith(("$GPRMC", "$GNRMC")):
                    try:
                        msg = pynmea2.parse(line)
                    except pynmea2.ParseError:
                        continue
                    if getattr(msg, "status", None) == "A":
                        with self.lock:
                            self.lat = msg.latitude
                            self.lon = msg.longitude
            except Exception:
                if ser:
                    try: ser.close()
                    except Exception: pass
                ser = None
                time.sleep(0.5)

    def stop(self):
        self._stop = True

    def get(self):
        with self.lock:
            return (self.lat, self.lon)

# ---- Non-blocking JSONL writer thread ----
log_q = queue.Queue(maxsize=1000)

class JsonlWriter(threading.Thread):
    def __init__(self, path):
        super().__init__(daemon=True)
        self.path = path
        self._stop = False

    def run(self):
        with open(self.path, "a", buffering=1) as f:  # line-buffered
            while True:
                rec = log_q.get()
                if rec is None:  # sentinel for shutdown
                    log_q.task_done()
                    break
                f.write(json.dumps(rec) + "\n")
                log_q.task_done()

    def stop(self):
        # push sentinel and wait for queue to drain
        log_q.put(None)

def main():
    # Start GPS + writer
    gps = GPSReader(GPS_PORTS, GPS_BAUD); gps.start()
    writer = JsonlWriter(JSONL_PATH); writer.start()

    # Camera
    picam2 = Picamera2()
    config = picam2.create_video_configuration(
        main={"size": FRAME_SIZE, "format": "BGR888"},
        buffer_count=6,                     # integer, not {6}
        controls={"FrameRate": 30, "flip_horizontal": True, "flip_vertical": True}
    )

    picam2.configure(config)
    picam2.start()
    
    # YOLO + ByteTrack
    model = YOLO(MODEL_PATH)
    class_names = model.names
    tracker = sv.ByteTrack()
    box_annotator   = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_thickness=2, text_scale=0.6)

    # Dedup: log each (track_id, class_id) once
    seen = set()

    print("Running? press 'q' to quit.")
    try:
        while True:
            # Frame (RGB -> BGR)
            frame_rgb = picam2.capture_array()
            frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            # Inference
            results = model(frame, verbose=False)
            r = results[0]

            # Gather detections
            if r.boxes is not None and len(r.boxes) > 0:
                xyxy = r.boxes.xyxy.cpu().numpy().astype(np.float32)
                conf = r.boxes.conf.cpu().numpy().astype(np.float32)
                cls  = (r.boxes.cls.cpu().numpy().astype(int)
                        if r.boxes.cls is not None else np.zeros(len(conf), dtype=int))
                keep = conf >= CONF_THRESH
                xyxy, conf, cls = xyxy[keep], conf[keep], cls[keep]
            else:
                xyxy = np.empty((0, 4), dtype=np.float32)
                conf = np.empty((0,), dtype=np.float32)
                cls  = np.empty((0,), dtype=int)

            dets = sv.Detections(xyxy=xyxy, confidence=conf, class_id=cls)
            tracked = tracker.update_with_detections(dets)
            
                        # Labels + logging
            labels = []
            tids = tracked.tracker_id if tracked.tracker_id is not None else []
            lat, lon = gps.get()

            for i in range(len(tracked)):
                tid = int(tids[i]) if len(tids) > i else -1
                cid = int(tracked.class_id[i]) if tracked.class_id is not None else -1
                cname = class_names.get(cid, f"class_{cid}")
                score = float(tracked.confidence[i]) if tracked.confidence is not None else None

                labels.append(f"{cname} | ID {tid} | {score:.2f}" if score is not None else f"{cname} | ID {tid}")

                key = (tid, cid)
                if tid != -1 and key not in seen:
                    record = {
                        "track_id": tid,
                        "class_id": cid,
                        "class_name": cname,
                        "confidence": score,
                        "bbox_xyxy": [float(x) for x in tracked.xyxy[i].tolist()],
                        "latitude": float(lat) if lat is not None else None,
                        "longitude": float(lon) if lon is not None else None
                    }
                    # Non-blocking enqueue; drop if queue is full to avoid FPS hiccups
                    try:
                        log_q.put_nowait(record)
                    except queue.Full:
                        pass
                    seen.add(key)

            # Draw
            frame = box_annotator.annotate(scene=frame, detections=tracked)
            frame = label_annotator.annotate(scene=frame, detections=tracked, labels=labels)
            
            # Get inference time
            inference_time = results[0].speed['inference']
            fps = 1000 / inference_time  # Convert to milliseconds
            text = f'FPS: {fps:.1f}'
    
            # Define font and position
            font = cv2.FONT_HERSHEY_SIMPLEX
            text_size = cv2.getTextSize(text, font, 1, 2)[0]
            text_x = frame.shape[1] - text_size[0] - 10  # 10 pixels from the right
            text_y = text_size[1] + 10  # 10 pixels from the top

            # Draw the text on the annotated frame
            cv2.putText(frame, text, (text_x, text_y), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

            
            # GPS overlay
            gps_text = f"GPS: {lat:.6f}, {lon:.6f}" if (lat is not None and lon is not None) else "GPS: acquiring?"
            cv2.putText(frame, gps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

            # Show
            cv2.imshow("YOLOv8 + ByteTrack + GPS (queued log)", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        # Shutdown threads cleanly
        try: picam2.stop()
        except Exception: pass
        cv2.destroyAllWindows()

        gps.stop()
        writer.stop()
        # Wait for queue to flush
        log_q.join()

if __name__ == "__main__":
    main()

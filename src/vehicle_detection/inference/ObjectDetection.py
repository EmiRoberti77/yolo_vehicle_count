from ultralytics import YOLO
import cv2
import datetime
import uuid



class BaseDetection():
  def __init__(self, model:str, window_name=None, display_window=False):
    self.model = model
    self.class_list = ["car"]
    self.seen_ids = set()
    self.count = 0
    self.model = YOLO(self.model)
    self.window_name = window_name if window_name is not None else str(uuid.uuid4())
    self.display_window = display_window


class ObjectDetection(BaseDetection):
  def __init__(self, model: str, window_name=None, display_window=False):
    super().__init__(model, window_name, display_window)

  def update_frame_overlay(self, frame, point_1, point_2, color, label, count, thickness):
    cv2.rectangle(frame, point_1, point_2, color, thickness)
    x,y = point_1
    cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 0.8, color, thickness)
    cv2.putText(frame, count, (20, 40), cv2.FONT_HERSHEY_PLAIN, 1.2, color, thickness)
    return frame

  
  def track(self, frame, persist=True, tracker='bytetrack.yaml'):
    frame = cv2.resize(frame, (416, 240))
    results = self.model.track(frame, persist=persist, tracker=tracker)[0]
    for box in results.boxes:
      cls_id = int(box.cls[0])
      class_name = self.model.names[cls_id]

      if class_name == self.class_list[0]:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        car_id = int(box.id[0]) if box.id is not None else None
        if car_id and car_id not in self.seen_ids:
          self.seen_ids.add(car_id)
          self.count += 1
        
        frame = self.update_frame_overlay(frame, (x1,y1), (x2,y2), (0,255,0), f"{class_name}:{cls_id}", f"Cars:{self.count}", 2)

    if self.display_window:
      cv2.imshow(self.window_name, frame)
      
    return frame
  
  
  def release(self):
    cv2.destroyAllWindows()


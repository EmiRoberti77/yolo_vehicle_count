import cv2
from ultralytics import YOLO
from persistance.data_store import Detection, Detection_repo
from datetime import datetime

CAR = 'car'
VIDEO_WINDOW = "Video"
model = YOLO("yolo11n.pt")
cap = cv2.VideoCapture("../videos/cars_highway.mp4")
car_count = 0
seen_ids = set()


def update_car_counter_label(frame):
  global car_count
  cv2.putText(frame, f"Cars:{car_count}", (20, 40), cv2.FONT_HERSHEY_PLAIN, 1.2, (255,0,0), 2)

def persist_detection( object_id, 
                      class_name, 
                      confidence, 
                      x1, 
                      y1, 
                      x2, 
                      y2, 
                      frame_number, 
                      time_stamp, 
                      source):
  detection = Detection(
    object_id=object_id,
    class_name=class_name,
    confidence=confidence,
    x1=x1, y1=y1, x2=x2, y2=y2,
    frame_number=frame_number,
    time_stamp=time_stamp,
    source=source
  )
  Detection_repo(detection=detection).save()


def detected_frame(frame, display_frame=True):
  global car_count
  global seen_ids
  results = model.track(frame, persist=True, tracker="bytetrack.yaml")[0]
  for box in results.boxes:
    cls_id = int(box.cls[0])
    class_name = model.names[cls_id]

    # Only increment count for car
    if class_name == CAR:
      x1, y1, x2, y2 = map(int, box.xyxy[0])
      car_id = int(box.id[0]) if box.id is not None else None
      if car_id and car_id not in seen_ids:
        seen_ids.add(car_id)
        car_count += 1
        persist_detection(car_id, class_name, 0.9, x1, y1, x2, y2, car_count, datetime.utcnow(), "EMI")

      cv2.rectangle(frame, (x1, y1), (x2,y2), (0, 255, 0), 2)
      cv2.putText(frame, f"{class_name}:{car_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_PLAIN, 0.8, (0,255,0),2)
      update_car_counter_label(frame)
      
   
  
  if display_frame:
    cv2.imshow(VIDEO_WINDOW, frame)      
  # return the frame ( if car was detected, frame will have bbox and car label)
  return frame


def start_processing():
  while True:
    ret, frame = cap.read()
    if ret == False:
      break

    _ = detected_frame(frame)
    
    key = cv2.waitKey(10)
    if key == ord('q'):
      break

def release():
  print("Releasing resources")
  cap.release()
  cv2.destroyAllWindows()

if __name__ == "__main__":
  start_processing()
  release()
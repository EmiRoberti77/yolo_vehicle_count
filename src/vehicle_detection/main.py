import cv2
from ultralytics import YOLO
from persistance.data_store import Detection, Detection_repo
from datetime import datetime
from inference.ObjectDetection import ObjectDetection

models = ["yolo11n.pt"]
CAR = 'car'
VIDEO_WINDOW = "Video"
cap = cv2.VideoCapture("../videos/cars_highway.mp4")

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

def start_processing():
  objDetection = ObjectDetection(models[0], "Highway", True)
  while True:
    ret, frame = cap.read()
    if ret == False:
      break

    _ = objDetection.track(frame)
    
    key = cv2.waitKey(10)
    if key == ord('q'):
      objDetection.release()
      break

def release():
  print("Releasing resources")
  cap.release()
  cv2.destroyAllWindows()

if __name__ == "__main__":
  start_processing()
  release()
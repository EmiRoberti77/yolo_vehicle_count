import cv2
from ultralytics import YOLO
from persistance.data_store import Detection, Detection_repo
from datetime import datetime
from inference.ObjectDetection import ObjectDetection
import threading
from multiprocessing import Process
import multiprocessing

models = ["yolo11n.pt"]

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

def order_video_windows(position:int, title:str):
  # Move the window to a unique position based on process number
  window_width = 420
  window_height = 260
  columns = 4  # e.g. 4 windows per row
  row = position // columns
  col = position % columns
  x = col * window_width
  y = row * window_height
  cv2.namedWindow(title)
  cv2.moveWindow(title, x, y)

def process(path, title, index):
  cap = cv2.VideoCapture(path)
  # cap.set(cv2.CAP_PROP_FPS, 1)
  cap.set(cv2.CAP_PROP_FRAME_WIDTH, 416)
  cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
  objDetection = ObjectDetection(models[0], title, True)
  frame_count = 0

  order_video_windows(title=title, position=index)

  while True:
    ret, frame = cap.read()
    if ret == False:
      break
 
    frame_count +=1 
    if frame_count % 5 != 0:
      continue # skip frames

    _ = objDetection.track(frame)
    
    key = cv2.waitKey(10)
    if key == ord('q'):
      objDetection.release()
      break
   
  cap.release()
  cv2.destroyAllWindows()

if __name__ == "__main__":
  multiprocessing.set_start_method("spawn") # mac use
  processes = []
  for i in range(16):
    print("process", i)
    p = Process(target=process, args=('../videos/cars_highway.mp4', f"video-{i}", i))
    processes.append(p)
  
  for p in processes:
    p.start()

  for p in processes:
    p.join()
  
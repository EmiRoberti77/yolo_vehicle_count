import cv2
from ultralytics import YOLO

CAR = 'car'
VIDEO_WINDOW = "Video"
model = YOLO("yolo11n.pt")
cap = cv2.VideoCapture("../videos/cars_highway.mp4")
car_count = 0

def update_car_counter_label(frame):
  global car_count
  cv2.putText(frame, f"Cars:{car_count}", (20, 40), cv2.FONT_HERSHEY_PLAIN, 1.2, (255,0,0), 2)

def detected_frame(frame, display_frame=True):
  global car_count
  results = model(frame)[0]
  for box in results.boxes:
    cls_id = int(box.cls[0])
    class_name = model.names[cls_id]

    # Only increment count for car
    if class_name == CAR:
      car_count += 1
      x1, y1, x2, y2 = map(int, box.xyxy[0])
      cv2.rectangle(frame, (x1, y1), (x2,y2), (0, 255, 0), 2)
      cv2.putText(frame, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_PLAIN, 0.8, (0,255,0),2)
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
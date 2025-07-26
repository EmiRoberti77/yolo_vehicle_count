from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch
import cv2

class ObjectContext():
  device = "cuda" if torch.cuda.is_available() else "cpu"

  # Load model + processor
  processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
  model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

  @staticmethod
  def describe_scene(frame):
    # Convert OpenCV BGR to PIL RGB
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Process for captioning
    inputs = ObjectContext.processor(image, return_tensors="pt").to(ObjectContext.device)
    out = ObjectContext.model.generate(**inputs)

    caption = ObjectContext.processor.decode(out[0], skip_special_tokens=True)
    return caption

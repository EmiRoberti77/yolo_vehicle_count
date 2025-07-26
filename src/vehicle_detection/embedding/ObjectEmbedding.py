import torch
import clip
from PIL import Image
import cv2

class ObjectEmbedding:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    @staticmethod
    def embed_frame_with_auto_context(frame):
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        image_input = ObjectEmbedding.preprocess(pil_image).unsqueeze(0).to(ObjectEmbedding.device)

        labels = ["a car", "a person", "a truck", "a highway", "a building", "a bicycle", "a motorbike", "traffic", "a road"]
        text_inputs = clip.tokenize(labels).to(ObjectEmbedding.device)

        with torch.no_grad():
            image_features = ObjectEmbedding.model.encode_image(image_input)
            text_features = ObjectEmbedding.model.encode_text(text_inputs)

            similarity = (image_features @ text_features.T).squeeze(0)
            top_idx = similarity.argmax().item()
            context_text = labels[top_idx]

            print(f"📝 Auto-context: {context_text}")

            selected_text_input = clip.tokenize([context_text]).to(ObjectEmbedding.device)
            selected_text_features = ObjectEmbedding.model.encode_text(selected_text_input)

            combined_features = (image_features + selected_text_features) / 2
            return combined_features.cpu().numpy(), context_text

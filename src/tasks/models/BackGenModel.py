import PIL
import io
import requests
import torch
import torchvision.transforms as T
import torchvision
import numpy as np
from io import BytesIO
from diffusers import StableDiffusionInpaintPipeline
from rembg import remove


def remove_background(img):

    # Выполняем удаление фона
    input_image = PIL.Image.open(img)
    output_image = remove(input_image)
    image_bytes = io.BytesIO()
    output_image.save(image_bytes, format='PNG')
    output_image_binary = image_bytes.getvalue()

    return output_image_binary


class MaskRCNNInference:
    def __init__(self):
        self.model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
        self.model.eval()
        self.transform = T.Compose([T.ToTensor()])

    def mask(self, image_path):
        image = PIL.Image.open(image_path).convert("RGB")
        image_tensor = self.transform(image).unsqueeze(0)

        with torch.no_grad():
            prediction = self.model(image_tensor)

        mask = prediction[0]['masks'][0, 0].cpu().numpy()

        masked_image = np.zeros_like(np.array(image))
        masked_image[:, :, 0] = np.where(mask > 0.5, 0, 255)
        masked_image[:, :, 1] = np.where(mask > 0.5, 0, 255)
        masked_image[:, :, 2] = np.where(mask > 0.5, 0, 255)

        masked_image = PIL.Image.fromarray(masked_image.astype(np.uint8))
        return masked_image


class BackGenModel:
    def __init__(self):
        self.pipe = StableDiffusionInpaintPipeline.from_pretrained("runwayml/stable-diffusion-inpainting", torch_dtype=torch.float16)

    def infer(self, img, prompt):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.pipe.to(device)

        init_image = PIL.Image.open(img).resize((512, 512))
        #получаем маску изображения
        maskrcnn_model = MaskRCNNInference()
        mask_image = maskrcnn_model.mask(img).resize((512, 512))

        image_gen = self.pipe(prompt=prompt, image=init_image, mask_image=mask_image).images[0]

        # Конвертируем изображение в бинарное представление
        image_bytes = io.BytesIO()
        image_gen.save(image_bytes, format='PNG')
        image_binary = image_bytes.getvalue()

        return image_binary

import PIL
import io
import requests
import torch
import torchvision.transforms as T
import torchvision
import numpy as np
import random
from io import BytesIO
from diffusers import StableDiffusionInpaintPipeline
from rembg import remove


def remove_background(img):

    # Выполняем удаление фона
    output_image = remove(img)
    return output_image


class MaskRCNNInference:
    def __init__(self):
        self.model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
        self.model.eval()
        self.transform = T.Compose([T.ToTensor()])

    def mask(self, image_path):
        image = PIL.Image.open(io.BytesIO(image_path)).convert("RGB")
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
        self.pipe = None

    def infer(self, img):
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting",
            torch_dtype=torch_dtype
        )
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # List of Prompts
        prompts = [
            "vintage, oil painting, impressionism, artgallery.com, 3000x2000 pixels, serene landscape, pastel colors, soft lighting",
            "contemporary, digital art, abstract expressionism, abstractartworks.net, 4000x4000 pixels, vibrant splatter, bold colors, dynamic lighting",
            "still life, watercolor, realism, famousartworks.com, 2500x3500 pixels, lifelike details, warm color palette, natural lighting",
            "surrealism, mixed media, dreamworlds.org, 3500x2500 pixels, dreamlike scenes, surreal colors, dramatic lighting",
            "high fashion, fashion photography, couture, fashioniconmag.com, 3500x5000 pixels, designer clothing, bold colors, studio lighting",
            "minimalism, architectural rendering, modernism, architecturalwonders.com, 4500x3000 pixels, clean lines, earthy tones, even lighting",
            "botanical, botanical illustration, vintage, botanicalartprints.com, 3000x4500 pixels, intricate details, muted colors, soft diffused light",
            "sci-fi, digital painting, futuristic, sci-fifantasyworlds.com, 4000x4000 pixels, biomechanical elements, metallic hues, eerie lighting",
            "conceptual art, mixed media, symbolism, symbolicartworks.com, 3500x3500 pixels, personal narratives, vibrant colors, emotional lighting",
            "interior design, mid-century modern, Eames, interiorinspiration.com, 4500x3000 pixels, iconic furniture, retro colors, natural lighting",
            "abstract, fluid art, abstractartcreations.com, 4000x4000 pixels, flowing forms, vibrant colors, dynamic lighting",
            "fine art, classical painting, classicalartgallery.com, 3000x4000 pixels, timeless scenes, rich colors, dramatic lighting",
            "wildlife, wildlife painting, naturalism, birdwatchersparadise.com, 4000x3000 pixels, lifelike bird illustrations, natural colors, soft natural light",
            "sports, sports photography, action shots, sportsgallery.net, 5000x3500 pixels, intense moments, black and white, high contrast lighting",
            "cityscape, urban photography, neon-lit, futurecityscapes.com, 4500x2500 pixels, neon-lit metropolis, neon colors, rain-soaked streets",
            "food, food photography, gourmet, culinarydelights.com, 4000x6000 pixels, mouthwatering dishes, vibrant colors, diffused soft light",
            "history, historical painting, classicism, historicalmasterpieces.com, 3000x4000 pixels, epic scenes, rich colors, dramatic lighting",
            "abstract, contemporary art, abstractartworld.com, 4000x4000 pixels, expressive strokes, vibrant colors, dynamic lighting",
            "fashion, runway, haute couture, fashioniconmag.com, 3500x5000 pixels, designer collections, bold colors, studio lighting",
            "modern architecture, architectural photography, minimalist, modernarchitecturemag.com, 5000x3000 pixels, sleek designs, monochromatic, dramatic lighting",
            "electronics, product display, modern, no artist, no source, 4000x3000 pixels, sleek tech gadgets, minimalistic composition, white background, soft studio lighting",
            "furniture, interior decor, contemporary, no artist, no source, 4500x3500 pixels, stylish furnishings, neutral tones, natural lighting",
            "apparel, fashion products, trendy, no artist, no source, 3500x5000 pixels, fashionable clothing, folded neatly, pastel colors, studio lighting",
            "accessories, jewelry items, elegant, no artist, no source, 3000x3000 pixels, close-up details, metallic accents, controlled lighting",
            "culinary, gourmet food, appetizing, no artist, no source, 4000x6000 pixels, mouthwatering dishes, vibrant colors, diffused soft light",
            "technology, futuristic gadgets, sleek, no artist, no source, 4000x2500 pixels, high-tech devices, dark background, dramatic lighting",
            "automotive, luxury cars, high-end, no artist, no source, 5000x3500 pixels, premium vehicles, reflective surfaces, studio lighting",
            "cosmetics, beauty essentials, glamorous, no artist, no source, 3500x3500 pixels, luxury makeup, gold accents, soft diffused light",
            "home decor, vintage style, retro, no artist, no source, 3500x4500 pixels, nostalgic furnishings, warm colors, natural lighting",
            "toys, playful products, colorful, no artist, no source, 3000x2000 pixels, creative toy compositions, vivid hues, soft studio lighting",
            "sports gear, athletic equipment, dynamic, no artist, no source, 4500x3000 pixels, action-packed sports items, bold colors, high-speed lighting",
            "literature, classic books, timeless, no artist, no source, 2500x3500 pixels, well-arranged books, earthy tones, soft natural light",
            "timepieces, luxury watches, sophisticated, no artist, no source, 4000x4000 pixels, elegant wristwatches, metallic accents, controlled lighting",
            "office supplies, organized workspace, neat, no artist, no source, 3500x2500 pixels, tidy desk essentials, bright colors, soft diffused light",
            "kitchen essentials, culinary tools, inviting, no artist, no source, 3500x4500 pixels, gourmet kitchenware, warm tones, natural lighting",
            "pets and accessories, adorable animals, no artist, no source, 5000x3500 pixels, cute pets with their products, cheerful colors, soft natural light",
            "home decor, artistic furnishings, no artist, no source, 4000x3000 pixels, stylish decor pieces, artistic compositions, dramatic lighting",
            "athletic wear, sports clothing, energetic, no artist, no source, 3500x5000 pixels, athletes in action, vibrant colors, high-energy lighting",
            "gardening tools, outdoor equipment, no artist, no source, 4000x4000 pixels, essential garden tools, natural greenery, soft natural light",
            "tech accessories, futuristic gadgets, no artist, no source, 3500x2500 pixels, cutting-edge tech accessories, metallic accents, high-tech lighting",
            "home improvement, DIY tools, handyman, no artist, no source, 4500x3500 pixels, essential tools, organized workspace, natural lighting",
            "outdoor adventure, camping gear, rugged, no artist, no source, 4000x5000 pixels, outdoor equipment, natural landscapes, soft natural light",
            "beauty products, skincare essentials, fresh, no artist, no source, 3500x3500 pixels, skincare products, pastel colors, soft diffused light",
            "travel accessories, wanderlust, explorer, no artist, no source, 4000x3000 pixels, travel essentials, world map background, warm lighting",
            "fitness, workout equipment, active lifestyle, no artist, no source, 4500x3000 pixels, fitness gear, high-energy composition, studio lighting",
            "gaming, gaming peripherals, futuristic, no artist, no source, 3500x2500 pixels, gaming accessories, neon colors, cyberpunk lighting",
            "baby products, infant care, adorable, no artist, no source, 3500x4500 pixels, baby essentials, soft pastel colors, soft natural light",
            "vintage collectibles, antique items, nostalgia, no artist, no source, 3000x3000 pixels, collectible treasures, aged aesthetic, warm lighting",
            "kitchen gadgets, culinary tools, innovative, no artist, no source, 4000x3500 pixels, innovative kitchenware, modern design, natural lighting",
            "musical instruments, music gear, artistic, no artist, no source, 3500x5000 pixels, musical instruments, creative composition, studio lighting",
            "fitness apparel, activewear, dynamic, no artist, no source, 4500x3500 pixels, active lifestyle clothing, energetic poses, studio lighting",
            "pets and pet care, furry friends, no artist, no source, 5000x4000 pixels, pet care products, adorable pets, soft natural light",
            "home office, remote work setup, productive, no artist, no source, 3500x2500 pixels, organized workspace, technology integration, soft diffused light",
            "kitchen appliances, culinary tools, smart, no artist, no source, 4000x4500 pixels, smart kitchen gadgets, sleek design, natural lighting",
            "musical instruments, vintage classics, timeless, no artist, no source, 3500x4500 pixels, vintage musical instruments, warm tones, soft natural light",
            "fitness equipment, gym essentials, powerful, no artist, no source, 4500x3000 pixels, gym gear, high-intensity workout, dramatic lighting",
            "tech gadgets, innovative technology, no artist, no source, 4000x3000 pixels, cutting-edge tech, futuristic design, high-tech lighting",
            "hobbies, crafting supplies, creative, no artist, no source, 3500x3500 pixels, hobby materials, artistic compositions, soft natural light",
            "luxury accessories, high-end fashion, elegant, no artist, no source, 3500x5000 pixels, luxury accessories, exquisite details, studio lighting",
            "automotive parts, car accessories, performance, no artist, no source, 5000x3500 pixels, high-performance car parts, dynamic angles, studio lighting"]

        # List of Negative Prompts
        negative_prompts = [
            "blurry, lowres, abstract, no artist, no website, tiny resolution, random colors, poor lighting, no people",
            "out of focus, pixelated, caricature, unknown artist, no source, small image size, garish colors, harsh lighting, no people",
            "grainy, low quality, collage, no attribution, no source, small resolution, chaotic colors, inconsistent lighting, no people",
            "overexposed, distorted, clipart, anonymous artist, no website, low pixel count, clashing colors, harsh glare, no people",
            "low contrast, noisy, scribbles, no credits, no source, tiny image, incoherent colors, bad lighting, no people",
            "cropped, low quality, stick figures, no artist info, no source, minimal resolution, muddy colors, uneven lighting, no people",
            "watermarked, amateurish, doodles, no attribution, no website, minimal pixels, dull colors, flat lighting, no people",
            "low saturation, artifact-ridden, childish, no credits, no source, small dimensions, washed-out colors, flat lighting, no people",
            "distorted, oversaturated, clipart, anonymous artist, no portfolio, low res, jarring colors, eerie lighting, no people",
            "pixelation, blurred, sketches, no artist credits, no website, small size, clashing colors, harsh shadows, no people",
            "soft focus, low quality, random doodles, no attribution, no portfolio, minimal resolution, muted colors, poor lighting, no people",
            "vignetting, lowres, rough sketches, no artist info, no source, small image, dull colors, uneven lighting, no people",
            "compression artifacts, shaky, stick figures, no credits, no source, minimal pixels, flat colors, harsh lighting, no people",
            "distorted colors, pixelation, amateurish, no artist, no website, low pixel count, overexposed lighting, no people",
            "blurred lines, grainy, chaotic scribbles, no attribution, no source, small dimensions, clashing colors, poor lighting, no people",
            "low resolution, washed-out, abstract, no artist, no portfolio, minimal pixels, incoherent colors, uneven lighting, no people",
            "watermarked, shaky, childish drawings, no credits, no source, tiny resolution, muted colors, flat lighting, no people",
            "oversharpened, overexposed, clipart, anonymous artist, no website, low quality, jarring colors, harsh glare, no people",
            "low contrast, noisy, rough sketches, no artist info, no portfolio, small image, garish colors, poor lighting, no people",
            "cropped, low quality, random doodles, no attribution, no portfolio, small dimensions, muddy colors, uneven lighting, no people"
        ]

        self.pipe.to(device)

        init_image = PIL.Image.open(io.BytesIO(img))

        # вычисляем соотношение сторон картинки, чтобы восстановить его после генерации
        width, height = init_image.size
        ratio = height / width

        #получаем маску изображения
        maskrcnn_model = MaskRCNNInference()
        mask_image = maskrcnn_model.mask(img)

        #определяем целевые промпты
        prompt_preds = random.sample(prompts, 5)
        neg_prompt_preds = random.sample(negative_prompts, 5)

        img_binary_list = []

        for prompt, neg_prompt in zip(prompt_preds, neg_prompt_preds):

          image_gen = self.pipe(prompt=prompt, negative_prompt=neg_prompt, image=init_image, mask_image=mask_image).images[0].resize((512, int(512 * ratio)))

          # Конвертируем изображение в бинарное представление
          image_bytes = io.BytesIO()
          image_gen.save(image_bytes, format='PNG')
          image_binary = image_bytes.getvalue()

          img_binary_list.append(image_binary)

        return img_binary_list





# Scene-Extension-Outpainting-and-Inpainting-with-Stable-Diffusion

# Image Outpainting and Inpainting

This repository demonstrates how to extend an image using outpainting with the Pillow library and further enhance it with the Stable Diffusion inpainting model.

## Overview

The process involves two main steps:

1. **Outpainting**: Extending the image size by adding padding.
2. **Inpainting**: Enhancing the extended image using the Stable Diffusion inpainting model.

## Prerequisites

- Python 3.x
- Pillow
- Transformers (for Stable Diffusion)
- torch (PyTorch)
- torchvision

## Setup

1. **Install the required libraries**:

    ```bash
    pip install pillow transformers torch torchvision
    ```

2. **Clone this repository**:

    ```bash
    git clone https://github.com/yourusername/image-outpainting-inpainting.git
    cd image-outpainting-inpainting
    ```

## Outpainting with Pillow

The outpainting process extends the image by adding padding around it. Here’s a step-by-step breakdown:

1. **Loading the Image**:

    ```python
    from PIL import Image

    # Load and convert image to RGB format
    image = Image.open('path_to_image.jpg').convert('RGB')
    ```

2. **Defining the Outpainting Function**:

    ```python
    def outpaint_image(image, padding_size):
        # Create a new canvas with additional padding
        width, height = image.size
        new_size = (width + 2 * padding_size, height + 2 * padding_size)
        new_image = Image.new('RGB', new_size, (255, 255, 255))
        new_image.paste(image, (padding_size, padding_size))
        return new_image
    ```

3. **Applying the Outpainting Function**:

    ```python
    # Apply outpainting with 128 pixels padding
    outpainted_image = outpaint_image(image, 128)
    outpainted_image.show()
    ```

## Inpainting with Stable Diffusion

Enhance the outpainted image using Stable Diffusion for inpainting.

1. **Loading the Model**:

    ```python
    from transformers import AutoFeatureExtractor, AutoModelForImageInpainting
    import torch

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AutoModelForImageInpainting.from_pretrained('CompVis/stable-diffusion-inpainting-v1-0').to(device)
    ```

2. **Preparing the Image**:

    ```python
    from torchvision import transforms

    def resize_image(image, size=(512, 512)):
        transform = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor()
        ])
        return transform(image).unsqueeze(0).to(device)
    
    resized_image = resize_image(outpainted_image)
    ```

3. **Defining the Prompt and Inpainting**:

    ```python
    from transformers import DPTFeatureExtractor

    prompt = "Extend the image with a whimsical, magical style, maintaining fine details."
    feature_extractor = DPTFeatureExtractor()

    # Inpainting pipeline
    with torch.no_grad():
        output = model(prompt=prompt, image=resized_image, guidance_scale=17, num_inference_steps=50)
    
    # Convert output to PIL Image
    output_image = transforms.ToPILImage()(output[0].cpu())
    output_image.show()
    ```

## Result

By combining outpainting with Pillow and inpainting with Stable Diffusion, the original image is successfully extended while preserving its artistic style and enhancing its quality.

## Contributing

Feel free to submit issues or pull requests to improve this repository.

## License

This project is licensed under the MIT License.

## Acknowledgments

- [Pillow Documentation](https://pillow.readthedocs.io/)
- [Transformers Documentation](https://huggingface.co/transformers/)
- [Stable Diffusion Model](https://huggingface.co/CompVis/stable-diffusion-inpainting-v1-0)

---

Replace `path_to_image.jpg` and `yourusername` with the appropriate values. This README should give a clear overview of the process and how to set it up.

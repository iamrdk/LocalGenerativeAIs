import os
import torch
from diffusers import SanaPipeline, StableDiffusionXLPipeline, EulerAncestralDiscreteScheduler, StableDiffusionXLImg2ImgPipeline
from .gen_ai_utilities import file_saver, optimize_model


class SANA_IMAGE:
    """ A class that uses Sana 1.5 1.6B model to generate images.
    """
    def __init__(self, output_folder: str):
        """ Initialize the SanaPipeline
            Args:
                output_folder (str): The location to store the generated images.
        """
        self.output_folder = output_folder
        self.pipe = SanaPipeline.from_pretrained(
            "Efficient-Large-Model/SANA1.5_1.6B_1024px_diffusers",
            torch_dtype=torch.bfloat16,
        )
        self.pipe.text_encoder.to(torch.bfloat16)
        
        # This makes the model fit in low VRAM
        optimize_model(self.pipe)
    
    def generate_image(self, prompt: str, img_size: list = [1920, 1080], seed: int = None) -> str:
        """ Create the image with optional deterministic seed.
            Args:
                prompt (str): The image prompt.
                (Optional)
                img_size (list): The image width and height.
                seed (int): The manual seed for consistent results.
            Return:
                (str): The path of the generated image.
        """
        image = self.pipe(
            prompt=prompt,
            width=img_size[0],
            height=img_size[1],
            # guidance_scale=4.5,
            num_inference_steps=25,
            **({"generator":torch.Generator("cuda").manual_seed(seed)} if seed else {})
        )[0]

        return file_saver(self.output_folder, image=image[0])


class SDXL_IMAGE:
    """ A class to generate images using SDXL.
    """
    def __init__(self, output_folder:str, lora_path:str=None):
        """ Initialize the base and refiner.
            Args:
                output_folder (str): The location to store the generated images.
                lora_path (str): The SDXL LoRA you want to add.
        """
        self.output_folder = output_folder
        # Load SDXL base
        self.base = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True
        )
        self.base.scheduler = EulerAncestralDiscreteScheduler.from_config(self.base.scheduler.config)

        optimize_model(self.base)

        if lora_path:
            if os.path.exists(lora_path):
                self.base.load_lora_weights(lora_path, adapter_name="lora")
                self.base.set_adapters(["lora"])
            else:
                print("LoRA path doesn't exist.")

        # Load SDXL refiner 
        self.refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-refiner-1.0",
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
        )
        self.refiner.scheduler = EulerAncestralDiscreteScheduler.from_config(self.refiner.scheduler.config)
        
        optimize_model(self.refiner)
    
    def generate_image(self, prompt:str) -> str:
        """ This function generated the image and refines it as well.
            Args:
                prompt (str): The prompt to generate the image.
            Returns:
                str: The path of the file generated.
        """
        # Inference settings
        n_steps = 40
        high_noise_frac = 0.8

        # Step 1: Generate latent image with base
        latents = self.base(
            prompt=prompt,
            num_inference_steps=n_steps,
            denoising_end=high_noise_frac,
            output_type="latent"
        ).images

        # Step 2: Refine image
        image = self.refiner(
            prompt=prompt,
            num_inference_steps=n_steps,
            denoising_start=high_noise_frac,
            image=latents
        ).images[0]

        return file_saver(self.output_folder, image=image)

    
if __name__=="__main__":
    obj = SANA_IMAGE("outputs")
    prompt = "A man standing in a garden of roses with erupting volcanoes in the background"
    obj.generate_image(prompt)
    obj.generate_image(prompt, seed=42)

    del obj

    obj = SDXL_IMG("outputs")
    obj.generate_image(prompt)


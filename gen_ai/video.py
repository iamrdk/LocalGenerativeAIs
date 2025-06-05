import torch
from diffusers import AutoencoderKLWan, WanPipeline, LTXPipeline

from .gen_ai_utilities import file_saver, optimize_model

class WAN_VIDEO:
    """ A class to generate videos.
    """
    def __init__(self, output_folder:str):
        """ Initialize the class and apply optimizations.
        """
        self.output_folder = output_folder
        model_id = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"

        vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.bfloat16)
        self.pipe = WanPipeline.from_pretrained(model_id, vae=vae, torch_dtype=torch.bfloat16)

        optimize_model(self.pipe, quantize=True)
    
    def generate_video(self, prompt:str, seed=None):
        """ Passes the prompt to the model to generate the video.
        """
        output = self.pipe(prompt=prompt, 
                           **({"generator":torch.Generator("cuda").manual_seed(seed)} if seed else {})).frames[0]
        return file_saver(self.output_folder, video=output)


class LTX_VIDEO:
    """ Generates video using LTX-Video.
    """
    def __init__(self, output_folder:str):
        """ Init the model.
        """
        self.output_folder = output_folder
        self.pipe = LTXPipeline.from_pretrained("Lightricks/LTX-Video", torch_dtype=torch.float16)
        
        optimize_model(self.pipe)
    
    def generate_video(self, prompt: str, num_frames: int =241, seed: int=None):
        """ Generate videos from text prompt.
        """
        video = self.pipe(prompt=prompt, 
                        num_frames=num_frames,
                        **({"generator":torch.Generator("cuda").manual_seed(seed)} if seed else {})
                        ).frames[0]

        return file_saver(self.output_folder, video=video)

if __name__=="__main__":
    from diffusers import MochiPipeline
    from diffusers.utils import export_to_video

    pipe = MochiPipeline.from_pretrained("genmo/mochi-1-preview", variant="bf16", torch_dtype=torch.bfloat16)

    # Enable memory savings
    optimize_model(pipe, quantize=True)

    prompt = "Close-up of a chameleon's eye, with its scaly skin changing color. Ultra high resolution 4k."
    frames = pipe(prompt, num_frames=84).frames[0]

    export_to_video(frames, "mochi.mp4", fps=30)

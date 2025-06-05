This repo has some open-source Generative AI models that you can run in 8GB VRAM. </br>
Kept this repo as simple as possible with models from HuggingFace (non-gated access).


Hardware Requirements:
|Type|Requirement|Tested on|
|----|-----------|---------|
|VRAM| 8GB | 8GB |
|RAM| 32GB | 40GB |
|Virtual Memory (swap) | 50GB | 55GB|
|CPU| Any modern CPU | 14700HX|

Software requirements:

[Python3](https://python.org) </br>
[Ollama](https://ollama.com)


Ollama Model: `llama3.1:8b` 
Install using: `ollama pull llama3.1:8b`

If you want to get started with this, let's go through with the steps in command prompt.
1. Clone the repo into your machine.
    ```
    git clone --recurse-submodules https://github.com/iamrdk/LocalGenerativeAIs.git
    cd LocalGenerativeAIs
    ```
2. Create a new virtual env and install all the dependencies with a simple batch script: `setup_env_win.bat` </br>
    The dependencies needs to be installed in a serial way to properly install the torch packages.

3. You can either create your own file and use the classes inside `gen_ai` or use the `simple_pipeline.py` file to automate everything. Code samples for each generator provided below in Easy Inference.</br>
    (UI will be there soon)

#
### Let's talk about the HuggingFace models and why only these were selected:
* Image Models:
    1. SDXL is popular and is the best model so far with decent results. It's limited to a max of 1024x1024 resolution images. This generates better than Sana for some reason. </br>
    More details can be found here for [SDXL Base](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) and [SDXL Refiner](https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0).
    2. Sana 1.5 is from NVIDIA and the only model that can go upto 4K resolution.
    More details on Sana 1.5 1.6B model can be found [here](https://huggingface.co/Efficient-Large-Model/SANA1.5_1.6B_1024px_diffusers).

* Speech Model:
    1. XTTS-v2 was selected as it has the ability to clone a voice with a 6s audio file. More details on XTTS-v2 can be found [here](https://huggingface.co/coqui/XTTS-v2)/

* Music Model:
    1. Musicgen Small was used as it's lightweight and can create decent 30s audio from a text prompt. More details can be found [here](https://huggingface.co/facebook/musicgen-small).

* Video Models:
    1. LTX-Video is very fast and can generate a 240 frame video of 704x512 pixels in less than 200 seconds or around 3min 15secs. The quality is not so good. More details [here](https://huggingface.co/Lightricks/LTX-Video).
    2. WAN 2.1 T2V is a good model, it runs in 8GB VRAM but it takes a long time to generate a video. Almost 30 mins for a 6s video of 832x480 pixels. The quality is good. More details [here](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B).

#
### Huggingface model size and download size
The AI models are quite heavy, this is why we need such high VRAM and memory. Let me explain and provide the file sizes for each of the models.</br>
|Model Name| Model Size|
|----------|-----------|
|SDXL Base| 6.62 GB|
|SDXL Refiner| 5.81 GB|
|Sana1.5 1.6B| 9.05 GB|
|XTTS-v2| 1.94 GB|
|Musicgen Small| 2.20 GB|
|LTX-Video| 26.4 GB|
|WAN2.1 T2V 1.3B| 26.9 GB|

If you are using only one of these models, those files will be automatically downloaded when you initialize the class of a particular model, like `SDXL_IMAGE`, `SANA_IMAGE`, etc. </br>
The download location defaults to the hugginface's `HF_HOME` or `HF_HUB_CACHE` location by default, if you want to set a different location, set these environment variables and point it to the location of your choice. If nothing is added it defaults to `~/.cache/huggingface/hub`. </br>
More details on huggingface caches can be found [here](https://huggingface.co/docs/huggingface_hub/en/guides/manage-cache). </br>

#
### Dependency on Ollama
Ollama is needed to run the text to text models for script writing and prompt enhancement for the above HF models. It was chosen specifically due to the ease of usage and also you can use Ollama as a standalone tool as well. For those who don't know about Ollama, it's like local ChatGPT but with opensource models from Facebook, Mistral, Qwen, Google, Deepseek, etc. </br>

In my tests, I found that `llama`, `qwen3` and `gemma3` were the best in following system prompts for a particular seed. Find more on Extra details.
#
### Extra details
* **SDXL LoRA support:** </br>
    You can also pass a LoRA to the SDXL base model during initialization, it should be compatible with SDXL and add the trigger word into your prompt to activate the LoRA.
* **Ollama:**
    1. ***What is a system prompt?*** </br>
    A system prompt in AI models is a foundational instruction, provided by the developers, that dictates how the AI should behave, interpret user input, and generate responses. It's a set of guidelines that shape the AI's role, personality, and overall tone, acting as a kind of "job description" for the AI.
    2. ***What is a seed?*** </br>
    In AI, a "seed" is a starting point for random number generation, crucial for various processes like initializing model weights, sampling data, and generating outputs. It ensures a level of controlled randomness, allowing for reproducible results, debugging, and controlled creativity in generative tasks. 
    3. ***Why does the code use a certain model and a certain seed?*** </br>
    Currently `llama3.1:8b` with a seed of `42` is being used as this is the seed value where the model generates the correct output that is required for the pipeline. Changing the seed or the system_prompt (even adding a space), can alter the results of the model. The system prompt was tested with multiple seeds and found that `42` was the best one. 
* **Extending virtual memory in Windows:** </br>
    Virtual memory gets used when your system RAM is fully utilized, this helps your system to run the program while using your SSD or HDD space as the extended RAM, it's very slow, but get's the job done. This is the Windows version of Linux's `swap` space. </br>
    Here is a link explaining about the [same](https://answers.microsoft.com/en-us/windows/forum/all/how-do-i-increase-virtual-ram-in-windows-10/4e98f34b-9bf7-4b45-b6c3-a6c9ba326294). I wouldn't bother about the comments at the end as they are saying that performance won't increase, here we are not extending the performance, but preparing it as a virtual memory.
#

### Easy inference after cloning
Assuming you are at the root location `./LocalGenerativeAIs/` and you want the outputs to be in `./outputs`.
#### For image generation using Sana1.5 1.6B-
```
from gen_ai import SANA_IMAGE

# Create an object.
obj = SANA_IMAGE("outputs")

# Pass your prompt.
prompt = "Self-portrait oil painting, a beautiful cyborg with golden hair, 8k"
generated_file = obj.generate_image(prompt)

# Print the location of the generated image file.
print(generated_file)
```

#### For image generation using Sana1.5 1.6B-
```
from gen_ai import SDXL_IMAGE

# Create an object.
obj = SDXL_IMAGE("outputs")

# Pass your prompt.
prompt = "A majestic lion jumping from a big stone at night"
generated_file = obj.generate_image(prompt)

# Print the location of the generated image file.
print(generated_file)
```

#### For music generation using Musicgen Small-
```
from gen_ai import GEN_MUSIC

# Create an object.
obj = GEN_MUSIC("outputs")

# Pass your prompt.
prompt = "lo-fi music with a soothing melody"
generated_file = obj.generate_music(prompt)

# Print the location of the generated music file.
print(generated_file)
```

#### For script generation using Ollama and Llama-
```
from gen_ai import OLLAMA_SCRIPT

# Create an object.
obj = OLLAMA_SCRIPT("outputs")

# Pass your prompt.
prompt = "The importance of internet"
generated_file = obj.generate_script(prompt)

# Print the location of the generated image.
print(generated_file)
```

#### For speech generation using XTTS-v2-
```
from gen_ai import XTTSv2_SPEECH

# Create an object.
obj = XTTSv2_SPEECH("outputs")

# Pass your prompt.
prompt = "It took me quite a long time to develop a voice, and now that I have it I'm not going to be silent."
generated_file = obj.generate_speech(prompt)

# Print the location of the generated speech file.
print(generated_file)
```

#### For video generation using LTX-Video-
```
from gen_ai import LTX_VIDEO

# Create an object.
obj = LTX_VIDEO("outputs")

# Pass your prompt.
prompt = "The video depicts a winding mountain road covered in snow, with a single vehicle traveling along it. The road is flanked by steep, rocky cliffs and sparse vegetation. The landscape is characterized by rugged terrain and a river visible in the distance."
generated_file = obj.generate_video(prompt)

# Print the location of the generated video file.
print(generated_file)
```

#### For video generation using WAN_VIDEO-
```
from gen_ai import WAN_VIDEO

# Create an object.
obj = WAN_VIDEO("outputs")

# Pass your prompt.
prompt = "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."
generated_file = obj.generate_video(prompt)

# Print the location of the generated video file.
print(generated_file)
```

import os
from datetime import datetime
from scipy.io import wavfile
from torchao.quantization import quantize_, int8_weight_only
from diffusers.utils import export_to_video


def get_current_time() -> str:
    """ Gets the current time.
    """
    return datetime.now().strftime("%d_%m_%Y_%H_%M_%S")

def optimize_model(pipe, quantize=False, seq=False):
    """ Quantizes the model to take less memory.
    """
    if quantize:
        # The below lines save some memory by quantizing the model while loading
        quantize_(pipe.text_encoder, int8_weight_only())
        quantize_(pipe.transformer, int8_weight_only())

    if seq:
        # Super slow but requires less than 2GB VRAM as everything is offloaded to CPU.
        pipe.enable_sequential_cpu_offload()
    else:
        pipe.enable_model_cpu_offload()


def initialize_project_name(project: str) -> str:
    """ Checks the project name and makes sure it's unique.
    """
    project_initials = ""
    output_folder = "outputs"
    os.makedirs(output_folder, exist_ok=True)

    for i in project.split():
        project_initials += i[0]
    
    project_initials = project_initials.upper()

    if project_initials in os.listdir(output_folder):
        os.makedirs(f"{output_folder}/{project_initials}_{get_current_time()}", exist_ok=True)
        output_folder = f"{output_folder}/{project_initials}_{get_current_time()}"
    else:
        os.makedirs(f"{output_folder}/{project_initials}", exist_ok=True)
        output_folder = f"{output_folder}/{project_initials}"
    
    return output_folder

def initialize_project(topic, output_folder=None):
    """ Checks the project name and makes sure it's unique.
        Updates the output_folder.
    """
    project_initials = ""
    if not output_folder:
        output_folder = "outputs"
        for i in topic.split():
            project_initials += i[0]
        project_initials = project_initials.upper()
        output_folder = f"{output_folder}/{project_initials}_{get_current_time()}"
    
    # Generate the other sub folders
    os.makedirs(os.path.join(output_folder, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_folder, "speech"), exist_ok=True)
    os.makedirs(os.path.join(output_folder, "scripts"), exist_ok=True)
    os.makedirs(os.path.join(output_folder, "music"), exist_ok=True)
    os.makedirs(os.path.join(output_folder, "videos"), exist_ok=True)

    return output_folder


def file_saver(output_folder, script=None, image=None, speech=None, music=None, video=None) -> str:
    """ Saves content based on what was passed.
    """
    if script:
        folder = f"{output_folder}/scripts"
        os.makedirs(folder, exist_ok=True)
        name = f"{folder}/script.txt"
        with open(name, 'w', encoding='utf-8') as f:
            f.write(script)
        return name
    elif image:
        folder = f"{output_folder}/images"
        os.makedirs(folder, exist_ok=True)
        name = f"{folder}/image_{get_current_time()}.png"
        image.save(name)
        return name
    elif speech:
        folder = f"{output_folder}/speech"
        os.makedirs(folder, exist_ok=True)
        name = f"{folder}/speech_{get_current_time()}.wav"
        wavfile.write(name, rate=22050, data=speech["wav"])
        return name
    elif music:
        folder = f"{output_folder}/music"
        os.makedirs(folder, exist_ok=True)
        name = f"{folder}/music_{get_current_time()}.wav"
        wavfile.write(name, rate=music["sampling_rate"], data=music["audio"])
        return name
    # elif video:
    else:
        folder = f"{output_folder}/videos"
        os.makedirs(folder, exist_ok=True)
        name = f"{folder}/video_{get_current_time()}.mp4"
        try:
            export_to_video(video, name, fps=12)
            return name
        except:
            return None

script = """
You are Scripter, an AI Agent. Your sole task is to generate a complete, highly structured YouTube video script approximately 10 minutes long, based on the user's topic.

You MUST follow these formatting rules exactly:
1. Structure: Divide the script into clearly defined sections: INTRO, multiple main sections, CONCLUSION, OUTRO.
2. Timing: Start each section with a timestamp in this format:
`[START_TIME - END_TIME]` (e.g., `[0:00 - 0:30]`).
Break the script into \~30-second increments. Total duration should be about 10 minutes.
3. Speaker: All spoken lines must be by the single speaker labeled exactly as `HOST:` — no other speakers or characters.
4. Visuals: Every timestamped block must include a visual instruction line starting with `VISUAL:`. Describe clearly what is shown, including on-screen text, animations, or scene setups.
5. Music: Every block must also contain a music cue line starting with `MUSIC:`. Describe mood, genre, or changes.
6. Dialogue: The `HOST:` dialogue should follow the visual and music cues. Dialogue must be detailed, informative, and engaging. Avoid short or vague statements.
7. Perspective: Use only the first-person monologue style from the HOST. No second- or third-person references, no other characters.
8. Plain Text Only: No markdown formatting. No bullets, asterisks, hashtags, or other symbols. Use plain text only.
9. No Meta Content: Do NOT explain what you're doing, introduce the script, summarize it, or ask questions. The output must be the script only.
10. Example Format (follow this structure exactly):
    ```
    [0:00 - 0:30]  
    MUSIC: Gentle ambient intro music fades in, light piano and synth pads  
    VISUAL: A wide drone shot of Earth from space, slowly zooming in  
    HOST: Imagine a world without the Earth, our home planet. No blue skies, no green forests, no vast oceans. Just an empty expanse of space and rock.
    ```

Continue this formatting for each 30-second segment until the 10-minute mark. Each block must contain `MUSIC:`, `VISUAL:`, and `HOST:` in that order.
"""

image = """
You will receive a topic followed by a short image prompt. Your task is to expand it into a richly detailed, visually descriptive prompt optimized for high-fidelity image generation.

Guidelines:
- Do NOT repeat or mention the topic in your response.
- Focus on concrete visual elements: environment, subjects, objects, lighting, mood, texture, atmosphere, and perspective.
- Be descriptive but concise. Use compact, information-dense phrasing to enhance clarity.
- Prioritize clarity and specificity over poetic or abstract language.
- Use natural language, not keywords or prompt tags.

Input format:
[TOPIC] : [Basic image description]
"""

music = """
You will receive a topic followed by a short music prompt. Expand it into a richly detailed description optimized for symbolic music generation.

Guidelines:
- Do NOT repeat or reference the topic in your response.
- Describe the mood, instruments, style/genre, tempo, and any unique elements (e.g., rhythm changes, harmony, ambient layers).
- Mention how the music evolves or transitions.
- Write in a clear and natural sentence, suitable for direct model input.

Input format:
[TOPIC] : [Basic music description]
"""

video = """
You will receive a topic followed by a short video prompt. Expand it into a fluid, cinematic description optimized for video generation.

Guidelines:
1. Begin with a short sentence capturing the main action.
2. Describe motion, gesture, and behavior of subjects.
3. Specify appearance of characters, props, and key objects.
4. Include detailed background and environmental context.
5. Specify camera work: angles, transitions, zooms, panning.
6. Note lighting, shadows, and color tones.
7. Mention notable visual changes or events if any.
8. Write as a single flowing paragraph, no bullets.
9. Keep the entire description under 100 words.

Input format:
[TOPIC] : [Basic visual prompt]
"""

dialogue = """
    You will receive a topic followed by a snippet of dialogue or narration. Your task is to expand the dialogue into a richer, more detailed first-person monologue or host-style commentary, staying within the context of the topic.

    Guidelines:
    1. The input format is: "[TOPIC] : [Dialogue or narration]"
    2. Do NOT repeat or restate the topic in your response.
    3. Only one speaker should be present — the HOST. No third-person narration, no dialogue tags, and no other characters.
    4. Expand naturally with added insight, context, emotion, and rhetorical depth, as if the speaker is elaborating for an audience.

    Example Input and Output:
    INPUT:  "What if the Earth never existed? : As we continue our journey into the unknown, how do we ensure that our impact on the universe is positive? Should we focus on preserving the planet as it stands or create a new one like it?"
    
    WRONG RESPONSE: “Actually, the idea that Earth's existence is necessary...” she said with a hint of emphasis.
    
    CORRECT RESPONSE: Actually, the idea that Earth's existence is necessary for life is a narrow perspective. We've been so Earth-centric for so long that we forget the universe doesn't need our permission to be creative.
"""

system_prompts = {"img": image,
                  "vid": video,
                  "dia": dialogue,
                  "music": music,
                  "script": script}

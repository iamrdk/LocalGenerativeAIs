from gen_ai import *


# A simple pipeline starting with an idea, generates a script and goes till images, video, speech and music.

# _____Generate Script_____
idea = "What if the earth was flat?"
output_folder = INIT_PROJECT(idea)
print(output_folder)

# Initialize the script generator
obj = OLLAMA_SCRIPT(output_folder)
obj.generate_script(idea)
generated_files = obj.script_detailer()
print(generated_files)

# Release memory for the other models as the Ollama model gets unloaded.
del obj

# _____Generate the music_____
obj = GEN_MUSIC(output_folder)
with open(generated_files["music"], 'r', encoding="utf-8")as f:
    for line in f.readlines():
        line = line.rstrip("\n")
        print(obj.generate_music(line))



# _____Generate the speech_____
obj = XTTSv2_SPEECH(output_folder)
with open(generated_files["dialogue"], 'r', encoding="utf-8")as f:
    for line in f.readlines():
        line = line.rstrip("\n")
        print(obj.generate_speech(line))


# _____Generate the images using the SDXL model_____
obj = SDXL_IMAGE(output_folder)
with open(generated_files["image"], 'r', encoding="utf-8")as f:
    for line in f.readlines():
        line = line.rstrip("\n")
        print(obj.generate_image(line))


# OR

# _____Generate the images using the Sana model_____
# obj = SANA_IMAGE(output_folder)
# with open(generated_files["image"], 'r', encoding="utf-8")as f:
#     for line in f.readlines():
#         line = line.rstrip("\n")
#         print(obj.generate_image(line))


# _____Generate the video using LTX-Video_____
obj = LTX_VIDEO(output_folder)
with open(generated_files["video"], 'r', encoding="utf-8")as f:
    for line in f.readlines():
        line = line.rstrip("\n")
        print(obj.generate_video(line))



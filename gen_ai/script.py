import ollama
import os
import atexit

from .gen_ai_utilities import file_saver, system_prompts


class OLLAMA_SCRIPT:
    """ Utilizes Ollama to create scripts.
    """
    def __init__(self, output_folder:str="outputs", model:str="llama3.1:8b", think=False):
        """ Initialize Ollama model.
            Args:
                output_folder (str): The folder where the script will be stored.
                model (str): The Ollama model you want to use.
                think (bool): Indicates if it's a thinking model or non-thinking model. Both output differently.
        """
        self.output_folder = output_folder

        # Suggested models: llama3, qwen3 and gemma3 models.
        self.model_name = model
        self.thinking_model = think

        self.idea = ""

        self.generated_files = {}

        atexit.register(self.model_unload)

    def response_handler(self, response):
        """ Handles the response based on thinking or non thinking model. Also trims unnecessary new lines.
            Args:
                response (ChatResponse): The response from Ollama chat.
            Returns:
                str: The formatted content.
        """
        message = response.message.content
        if self.thinking_model:
            message = message.split('</think>')[1]
        
        # Remove extra new lines.
        message = message.replace('\n\n', '\n')
        return message
    
    def prompt_handler(self, prompt, think=True):
        """ Handles the prompt by adding a /no_think to the prompt if it's a thinking model.
            Args:
                prompt (str): The user's input message.
                think (bool): Allow thinking
            Returns:
                str: The formatted prompt with /no_think added if it'd s thinking model.
        """
        # In thinking models, this helps in generating responses quickly.
        if self.thinking_model and not think:
            return prompt + '/no_think'
        
        return prompt
    
    def generate_script(self, prompt: str, seed=10) -> str:
        """ Generate a script based on the user's prompt.
            Args:
                prompt (str): The user's input message.
                seed (int): The seed to generate the best and consistent results.
            Returns:
                str: The file path of the generated script.
        """
        options = {}
        options["seed"] = seed
        self.idea = prompt
        response = ollama.chat(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompts["script"]},
                    {"role": "user", "content": self.prompt_handler(prompt)}],
                options=options
            )
        
        self.script = self.response_handler(response)

        self.generated_files["script"] = file_saver(self.output_folder, script=self.script)
    
        return self.generated_files["script"]
    
    def simply_chat(self, prompt: str, seed=None) -> str:
        """ Chat (without context) based on the user's prompt. Every chat will be like a temporary chat.
            Args:
                prompt (str): The user's input message.
                seed (int): The seed to generate the best and consistent results.
            Returns:
                str: The file path of the generated script.
        """
        options = {}
        if seed:
            options["seed"] = seed
        
        response = ollama.chat(
                model=self.model_name,
                messages=[
                    {"role": "user", "content": prompt}],
                options=options
            )
    
        return response.message.content

    def script_detailer(self):
        """ Adds details to the script it generated and saved them individually.
        """
        img_pattern = "VISUAL: "
        dia_pattern = "HOST: "
        music_pattern = "MUSIC: "

        music_file = f"{self.output_folder}/scripts/music.txt"
        dialogue_file = f"{self.output_folder}/scripts/dialogue.txt"
        image_file = f"{self.output_folder}/scripts/image.txt"
        video_file = f"{self.output_folder}/scripts/video.txt"

        with open(image_file, 'w+', encoding="utf-8")as im, open(music_file, 'w+', encoding="utf-8")as ms, open(dialogue_file, 'w+', encoding="utf-8")as di, open(video_file, 'w+', encoding="utf-8")as vd:
            for line in self.script.split('\n'):
                if line.startswith(img_pattern):
                    im.write(self.detailer(line, "img", 44)+"\n")
                    vd.write(self.detailer(line, "vid", 1)+"\n")
                elif line.startswith(dia_pattern):
                    di.write(self.detailer(line, "dia", 86)+"\n")
                elif line.startswith(music_pattern):
                    ms.write(self.detailer(line, "msc", 96)+"\n")
        
        self.generated_files["music"] = music_file
        self.generated_files["dialogue"] = dialogue_file
        self.generated_files["image"] = image_file
        self.generated_files["video"] = video_file
        
        return self.generated_files

    
    def detailer(self, line:str, desc_type:str, seed:int=42):
        """ Add detail to the line via the chosen model.
            Args:
                line (str): A simple sentence.
                desc_type (str): The type of description. Eg: img, vid, dia or music. 
                (Optional)
                seed (int): A seed for consistent results.
            Returns:
                str: Details added to the sentence.
        """
        system_prompt = ""

        match(desc_type):
            case "img": system_prompt = system_prompts["img"]
            case "vid": system_prompt = system_prompts["vid"]
            case "dia": system_prompt = system_prompts["dia"]
            case "msc": system_prompt = system_prompts["music"]
        
        options = {}
        options["seed"] = seed

        input_format = f"{self.idea} : {line}"
        response = ollama.chat(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": self.prompt_handler(input_format)}],
                options=options
            )
        
        # Replace any new line we got, as details should be a single line only.
        text = self.response_handler(response).replace('\n', ' ')
        
        return text
    
    def model_unload(self):
        """ This executes a command to stop ollama to save VRAM.
        """
        os.popen(f"ollama stop {self.model_name}")


if __name__=="__main__":
    from gen_ai_utilities import initialize_project

    prompt = "What if the Earth never existed?"
    output_folder = initialize_project(prompt)

    obj = OLLAMA_SCRIPT(output_folder)
    obj.generate_script(prompt)
    print(obj.script_detailer())

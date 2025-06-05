from transformers import pipeline
from .gen_ai_utilities import file_saver


class GEN_MUSIC:
    """ Generates music based on a prompt.
    """
    def __init__(self, output_folder:str):
        self.output_folder = output_folder
        self.synthesiser = pipeline("text-to-audio", "facebook/musicgen-small", device="cuda")
    
    def generate_music(self, prompt) -> str:
        """ Creates music based on a prompt.
            Args:
                user_prompt (str): The prompt passed by the user.
            Returns:
                str: The path to the generated music.
        """
        music = self.synthesiser(prompt, forward_params={"do_sample": True})

        return file_saver(self.output_folder, music=music)


if __name__=="__main__":
    obj = GEN_MUSIC("outputs")
    prompt = "A rising synth is playing an arpeggio with a lot of reverb. It is backed by pads, sub bass line and soft drums. This song is full of synth sounds creating a soothing and adventurous atmosphere. It may be playing at a festival during two songs for a buildup."
    obj.generate_music(prompt)
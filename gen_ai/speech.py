import torch
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts, XttsAudioConfig, XttsArgs
from TTS.config.shared_configs import BaseDatasetConfig

from .gen_ai_utilities import file_saver


class XTTSv2_SPEECH:
    """ This converts texts to speech and saves them with the timestamp.
    """
    def __init__(self, output_folder:str):
        """ Initialize with the model name and the system prompt.
        """
        self.output_folder = output_folder

        # Load config
        self.config = XttsConfig()
        self.config.load_json("models/XTTS-v2/config.json")

        # Init model
        self.model = Xtts.init_from_config(self.config)

        # Allow all required classes
        with torch.serialization.safe_globals({XttsConfig, XttsAudioConfig, BaseDatasetConfig, XttsArgs}):
            self.model.load_checkpoint(self.config, checkpoint_dir="models/XTTS-v2/", eval=True)

        self.model.cuda()

    def generate_speech(self, prompt: str) -> str:
        """
        Generate a voice based on the user's prompt.
        Args:
            prompt (str): The user's input message.
        Returns:
            str: The path of the generated wav file.
        """
        outputs = self.model.synthesize(
            prompt,
            self.config,
            gpt_cond_len=3,
            language="en",
            speaker_wav="models/voices/liam.wav",
            enable_text_splitting=True
        )

        return file_saver(self.output_folder, speech=outputs)


if __name__=="__main__":
    obj = XTTSv2_SPEECH("outputs")
    prompt = "In the ancient land of Eldoria, where the skies were painted with shades of mystic hues and the forests whispered secrets of old, there existed a dragon named Zephyros. Unlike the fearsome tales of dragons that plagued human hearts with terror, Zephyros was a creature of wonder and wisdom, revered by all who knew of his existence."
    obj.generate_speech(prompt)

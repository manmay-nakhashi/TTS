import torch
import librosa
import torchaudio
from TTS.config.shared_configs import BaseAudioConfig
from TTS.enhancer.configs.hifigan_2_config import Hifigan_2Config
from TTS.enhancer.datasets.dataset import load_wav_data
from TTS.config import load_config
from TTS.enhancer.models.hifigan_2 import Hifigan2, Hifigan2Args
from TTS.utils.audio import AudioProcessor
import sys
import collections
def average_checkpoints(inputs):
    """Loads checkpoints from inputs and returns a model with averaged weights.
    Args:
      inputs: An iterable of string paths of checkpoints to load from.
    Returns:
      A dict of string keys mapping to various values. The 'model' key
      from the returned dict should correspond to an OrderedDict mapping
      string parameter names to torch Tensors.
    """
    params_dict = collections.OrderedDict()
    params_keys = None
    new_state = None
    num_models = len(inputs)

    for fpath in inputs:
        with open(fpath, "rb") as f:
            state = torch.load(
                f,
                map_location=(
                    lambda s, _: torch.serialization.default_restore_location(s, "cpu")
                ),
            )
        # Copies over the settings from the first checkpoint
        if new_state is None:
            new_state = state

        model_params = state["model"]

        model_params_keys = list(model_params.keys())
        if params_keys is None:
            params_keys = model_params_keys
        elif params_keys != model_params_keys:
            raise KeyError(
                "For checkpoint {}, expected list of params: {}, "
                "but found: {}".format(f, params_keys, model_params_keys)
            )

        for k in params_keys:
            p = model_params[k]
            if isinstance(p, torch.HalfTensor):
                p = p.float()
            if k not in params_dict:
                params_dict[k] = p.clone()
                # NOTE: clone() is needed in case of p is a shared parameter
            else:
                params_dict[k] += p

    averaged_params = collections.OrderedDict()
    for k, v in params_dict.items():
        averaged_params[k] = v
        if averaged_params[k].is_floating_point():
            averaged_params[k].div_(num_models)
        else:
            averaged_params[k] //= num_models
    new_state["model"] = averaged_params
    return new_state
# check_list = ["/home/manmay/TTS-1/recipes/enhancer_hifigan2_vctk-November-02-2022_01+04PM-15f1333d/checkpoint_1035000.pth",
# "/home/manmay/TTS-1/recipes/enhancer_hifigan2_vctk-November-02-2022_01+04PM-15f1333d/checkpoint_1030000.pth",
# "/home/manmay/TTS-1/recipes/enhancer_hifigan2_vctk-November-02-2022_01+04PM-15f1333d/checkpoint_1025000.pth",
# "/home/manmay/TTS-1/recipes/enhancer_hifigan2_vctk-November-02-2022_01+04PM-15f1333d/checkpoint_1020000.pth"]
# avg_check = average_checkpoints(check_list)
# torch.save(avg_check, "average.pth")
config_path = "/home/manmay/TTS-1/recipes/enhancer_hifigan2_vctk-October-22-2022_05+25AM-15f1333d/config.json"
# wav_file= "/mnt/datasets/hindi_vctk_24/wav/news_speaker6_male/manatullah_Khan_की_आज_कोर्ट_में_पेशी__Today_News__Fast_News__Latest_Hindi_News_V99_qaFgDEc_001_13.wav"
wav_file = sys.argv[1]#"/mnt/datasets/hindi_vctk_24/wav/studio_iitm_female/text_01002.wav"
(sig, rate) = librosa.load(wav_file, sr=24000)
hifigan2_config = load_config(config_path)
hifigan2 = Hifigan2.init_from_config(hifigan2_config)
# hifigan2.load_checkpoint(hifigan2_config, "/home/manmay/TTS-1/recipes/enhancer_hifigan2_vctk-October-22-2022_05+25AM-15f1333d/checkpoint_950000.pth")
# hifigan2.load_checkpoint(hifigan2_config, "/home/manmay/TTS-1/recipes/enhancer_hifigan2_vctk-October-10-2022_10+18AM-15f1333d/checkpoint_605000.pth")
# hifigan2.load_checkpoint(hifigan2_config,"/home/manmay/TTS-1/recipes/enhancer_hifigan2_vctk-October-23-2022_05+34AM-15f1333d/checkpoint_705000.pth")
hifigan2.load_checkpoint(hifigan2_config,"/home/manmay/TTS-1/recipes/enhancer_hifigan2_vctk-October-25-2022_02+11PM-15f1333d/checkpoint_1000000.pth")
# hifigan2.load_checkpoint(hifigan2_config,"/home/manmay/TTS-1/recipes/enhancer_hifigan2_vctk-November-02-2022_01+04PM-15f1333d/checkpoint_1100000.pth")
# hifigan2.load_checkpoint(hifigan2_config,"/home/manmay/TTS-1/recipes/enhancer_hifigan2_vctk-November-04-2022_06+00PM-15f1333d/checkpoint_1160000.pth")
# hifigan2.load_checkpoint(hifigan2_config, "average.pth")
print(sig.shape)

sig = torch.tensor(sig, dtype=torch.float32)

upsampled_wav = hifigan2.inference(sig)
upsampled_wav = upsampled_wav.detach().numpy()
audio_path = sys.argv[2]#"audio_us_1.wav"
import soundfile as sf
sf.write(audio_path, upsampled_wav.squeeze(), 48000)

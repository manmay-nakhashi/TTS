import os

from trainer import Trainer, TrainerArgs

from TTS.config.shared_configs import BaseAudioConfig
from TTS.enhancer.configs.hifigan_2_config import Hifigan_2Config
from TTS.enhancer.datasets.dataset import load_wav_data
from TTS.enhancer.models.hifigan_2 import Hifigan2, Hifigan2Args
from TTS.utils.audio import AudioProcessor
import torch
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.manual_seed(54321)
use_cuda = torch.cuda.is_available()
num_gpus = torch.cuda.device_count()
print(" > Using CUDA: ", use_cuda)
print(" > Number of GPUs: ", num_gpus)
output_path = os.path.dirname(os.path.abspath(__file__))

audio_config = BaseAudioConfig(
    sample_rate=48000,
    fft_size = 1024,
    win_length = 1024,
    hop_length = 256,
    resample=False,
    num_mels=80,
    preemphasis=0.0,
    ref_level_db=20,
    log_func="np.log",
    do_trim_silence=True,
    trim_db=50,
)

bweArgs = Hifigan2Args(
    num_channel_wn=128,
    dilation_rate_wn=3,
    kernel_size_wn=3,
    num_blocks_wn=2,
    num_layers_wn=7,
    n_mfcc=18,
    n_mels=80,
)

config = Hifigan_2Config(
    model_args=bweArgs,
    audio=audio_config,
    run_name="enhancer_hifigan2_vctk",
    batch_size=32,
    eval_batch_size=4,
    num_loader_workers=8,
    num_eval_loader_workers=8,
    run_eval=True,
    test_delay_epochs=-1,
    target_sr=48000,
    input_sr=24000,
    segment_train=True,
    segment_len=0.30,
    cudnn_benchmark=False,
    epochs=1000,
    print_step=25,
    save_step=5000,
    print_eval=False,
    mixed_precision=False,
    lr_gen = 1e-6,
    lr_disc = 1e-6,
    lr_scheduler_gen= "StepLR",
    lr_scheduler_disc= "StepLR",
    lr_scheduler_gen_params= {"step_size": 30000, "gamma": 0.9},
    lr_scheduler_disc_params= {"step_size": 30000, "gamma": 0.9},
    scheduler_after_epoch=False,
    output_path=output_path,
    datasets=["path_to_dataset"],
    gt_augment = False
)

# INITIALIZE THE AUDIO PROCESSOR
# Audio processor is used for feature extraction and audio I/O.
# It mainly serves to the dataloader and the training loggers.
ap = AudioProcessor.init_from_config(config)

#
train_samples, eval_samples = load_wav_data(config.datasets)

# init model
model = Hifigan2(config, ap)

# init the trainer and 🚀
trainer = Trainer(
    TrainerArgs(),
    config,
    output_path,
    model=model,
    train_samples=train_samples,
    eval_samples=eval_samples,
)

trainer.fit()

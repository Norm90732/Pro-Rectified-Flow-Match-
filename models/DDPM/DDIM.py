from models.DDPM.DDPM import ConditionalDDPM 
from monai.networks.schedulers import DDIMScheduler


class ConditionalDDIM(ConditionalDDPM):
    def __init__(self, cfg):
        super().__init__(cfg)
        
        self.scheduler = DDIMScheduler(
            num_train_timesteps=1000,
            schedule="linear_beta",
            beta_start=0.0001,
            beta_end=0.02,
            clip_sample=False, 
            set_alpha_to_one=False
        ) # type: ignore
from denoising_diffusion_pytorch.D3M import Unet, GaussianDiffusion, Trainer
from denoising_diffusion_pytorch.deformation_estimation import U_Network, SpatialTransformer

if __name__ == '__main__':
    model = Unet(
        out_dim=1,
        channels=5,
        dim=64,
        dim_mults=(1, 2, 4, 8),
        flash_attn=True,
        self_condition=True,
    )

    nf_enc = [16, 32, 32, 32]
    nf_dec = [32, 32, 32, 32, 32, 16, 16]
    STN_UNet = U_Network(2, nf_enc, nf_dec)
    STN = SpatialTransformer((256, 256))

    diffusion = GaussianDiffusion(
        None,
        model,
        STN_UNet,
        STN,
        image_size=256,
        timesteps=1000,
        sampling_timesteps=250,
        objective='pred_v',
    )

    trainer = Trainer(None,
                      diffusion,
                      r'BraTS2023-TrainingData_png_t1n',
                      r'BraTS2023-TrainingData_png_t2w',
                      r'BraTS2023-TrainingData_png_t2f',
                      r'BraTS2023-TrainingData_png_t1c',
                      r'BraTS2023-TrainingData_png_seg',
                      r'BraTS2023-ValidationData_png_t1n',
                      r'BraTS2023-ValidationData_png_t2w',
                      r'BraTS2023-ValidationData_png_t2f',
                      r'BraTS2023-ValidationData_png_t1c',
                      r'BraTS2023-ValidationData_png_seg',
                      train_batch_size=16,
                      train_lr=8e-5,
                      train_num_steps=200000,  # total training steps
                      save_and_sample_every=10000,
                      num_samples=16,
                      gradient_accumulate_every=1,  # gradient accumulation steps
                      ema_decay=0.995,  # exponential moving average decay
                      amp=True,  # turn on mixed precision
                      calculate_fid=False,  # whether to calculate fid during training
                      augment_horizontal_flip=False,
                      results_folder='./experiments/****-**-**-**-**-**/results',
                      )
    trainer.load('20')

    trainer.test(r'BraTS2023-TestData_png_t1n',
                 r'BraTS2023-TestData_png_t2w',
                 r'BraTS2023-TestData_png_t2f',
                 r'BraTS2023-TestData_png_t1c',
                 r'BraTS2023-TestData_png_seg',
                 r'./experiments/****-**-**-**-**-**/test',
                 8)

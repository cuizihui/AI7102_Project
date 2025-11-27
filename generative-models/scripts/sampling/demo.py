import math
import os
import sys
from glob import glob
from pathlib import Path
from typing import List, Optional

sys.path.append(os.path.realpath(os.path.join(os.path.dirname(__file__), "../../")))
import numpy as np
import torch
from einops import rearrange, repeat
from fire import Fire
from omegaconf import OmegaConf
from PIL import Image
from scripts.util.detection.nsfw_and_watermark_dectection import DeepFloydDataFiltering
from sgm.inference.helpers import embed_watermark
from sgm.util import default, instantiate_from_config, append_dims
from torchvision.transforms import ToTensor
from tqdm import tqdm


def sample(
    root: str = "",
    num_frames: Optional[int] = None,  # 21 for SV3D
    num_steps: Optional[int] = None,
    version: str = "svd_xt",
    fps_id: int = 24,
    motion_bucket_id: int = 127,
    cond_aug: float = 0.02,
    seed: int = 23,
    decoding_t: int = 4,  # Number of frames decoded at a time! This eats most VRAM. Reduce if necessary.
    device: str = "cuda",
    cfg_scale: float = 1.0,
    cfg_scale_flip: float = 1.0,
    output_folder: Optional[str] = None,
    verbose: Optional[bool] = False,
    my_args=None,
):
    """
    If you run out of VRAM, try decreasing `decoding_t`.
    """
    if version == "svd_xt":
        num_frames = default(num_frames, 25)
        num_steps = default(num_steps, 25)
        output_folder = default(output_folder, "outputs/simple_video_sample/svd_xt/")
        model_config = "scripts/sampling/configs/svd_xt.yaml"
    else:
        raise ValueError(f"Version {version} does not exist.")

    model, filter = load_model(
        model_config,
        device,
        num_frames,
        num_steps,
        verbose,
    )
    torch.manual_seed(seed)
    
    print('my_args:', my_args)


    for subdir in sorted(os.listdir(root)):
        folder = os.path.join(root, subdir)
        if not os.path.isdir(folder):
            continue


        imgs = sorted([
            os.path.join(folder, f)
            for f in sorted(os.listdir(folder))
        ])
        print(imgs)

        if len(imgs) < 2:
            continue

        input_start_path, input_end_path = imgs[0], imgs[-1]
        sample_single(input_start_path,input_end_path,device,num_frames,
                  motion_bucket_id,fps_id,cond_aug,cfg_scale,
                  cfg_scale_flip,decoding_t,output_folder,num_steps,model,filter, my_args=my_args)

def sample_single(input_start_path,input_end_path,device,num_frames,
                  motion_bucket_id,fps_id,cond_aug,cfg_scale,
                  cfg_scale_flip,decoding_t,output_folder,num_steps,model,filter, my_args=None):
    model = model.to(torch.float16)
    with Image.open(input_start_path) as image:
        input_image = image.convert("RGB")
        input_image = input_image.resize((1024, 576))
        w, h = input_image.size

        if h % 64 != 0 or w % 64 != 0:
            width, height = map(lambda x: x - x % 64, (w, h))
            input_image = input_image.resize((width, height))
            print(
                f"WARNING: Your image is of size {h}x{w} which is not divisible by 64. We are resizing to {height}x{width}!"
            )

    image = ToTensor()(input_image)
    image = image * 2.0 - 1.0
    image = image.unsqueeze(0).to(device).to(torch.float16)
    latent = model.encode_first_stage(image)

    ## load end frame
    input_image_end = Image.open(input_end_path).convert("RGB").resize((1024, 576))
    image_end = ToTensor()(input_image_end)
    image_end = image_end * 2.0 - 1.0
    image_end = image_end.unsqueeze(0).to(device).to(torch.float16)
    latent_end = model.encode_first_stage(image_end)
    ##

    H, W = image.shape[2:]
    assert image.shape[1] == 3
    F = 8
    C = 4
    shape = (num_frames, C, H // F, W // F)
    if motion_bucket_id > 255:
        print(
            "WARNING: High motion bucket! This may lead to suboptimal performance."
        )

    if fps_id < 5:
        print("WARNING: Small fps value! This may lead to suboptimal performance.")

    if fps_id > 30:
        print("WARNING: Large fps value! This may lead to suboptimal performance.")

    value_dict = {}
    value_dict["cond_frames_without_noise"] = image
    value_dict["motion_bucket_id"] = motion_bucket_id
    value_dict["fps_id"] = fps_id
    value_dict["cond_aug"] = cond_aug
    value_dict["cond_frames"] = image + cond_aug * torch.randn_like(image)

    ## symmetric condition
    value_dict_end = {}
    value_dict_end["cond_frames_without_noise"] = image_end
    value_dict_end["motion_bucket_id"] = motion_bucket_id
    value_dict_end["fps_id"] = fps_id
    value_dict_end["cond_aug"] = cond_aug
    value_dict_end["cond_frames"] = image_end + cond_aug * torch.randn_like(image_end)
    ##

    with torch.no_grad():
        with torch.autocast(device):
            batch, batch_uc = get_batch(
                get_unique_embedder_keys_from_conditioner(model.conditioner),
                value_dict,
                [1, num_frames],
                T=num_frames,
                device=device,
            )
            c, uc = model.conditioner.get_unconditional_conditioning(
                batch,
                batch_uc=batch_uc,
                force_uc_zero_embeddings=[
                    "cond_frames",
                    "cond_frames_without_noise",
                ],
            )

            for k in ["crossattn", "concat"]:
                uc[k] = repeat(uc[k], "b ... -> b t ...", t=num_frames)
                uc[k] = rearrange(uc[k], "b t ... -> (b t) ...", t=num_frames)
                c[k] = repeat(c[k], "b ... -> b t ...", t=num_frames)
                c[k] = rearrange(c[k], "b t ... -> (b t) ...", t=num_frames)

            ## symmetric condition
            batch_end, batch_uc_end = get_batch(
                get_unique_embedder_keys_from_conditioner(model.conditioner),
                value_dict_end,
                [1, num_frames],
                T=num_frames,
                device=device,
            )
            c_end, uc_end = model.conditioner.get_unconditional_conditioning(
                batch_end,
                batch_uc=batch_uc_end,
                force_uc_zero_embeddings=[
                    "cond_frames",
                    "cond_frames_without_noise",
                ],
            )

            for k in ["crossattn", "concat"]:
                uc_end[k] = repeat(uc_end[k], "b ... -> b t ...", t=num_frames)
                uc_end[k] = rearrange(uc_end[k], "b t ... -> (b t) ...", t=num_frames)
                c_end[k] = repeat(c_end[k], "b ... -> b t ...", t=num_frames)
                c_end[k] = rearrange(c_end[k], "b t ... -> (b t) ...", t=num_frames)
            ##

            randn = torch.randn(shape, device=device)

            additional_model_inputs = {}
            additional_model_inputs["image_only_indicator"] = torch.zeros(
                2, num_frames
            ).to(device)
            additional_model_inputs["num_video_frames"] = batch["num_video_frames"]
            
            def denoiser(x, sigma, c, uc):
                c_out = dict()
                for k in c:
                    if k in ["vector", "crossattn", "concat"]:
                        c_out[k] = torch.cat((uc[k], c[k]), 0)
                    else:
                        assert c[k] == uc[k]
                        c_out[k] = c[k]
                denoiser_input, denoiser_sigma, denoiser_c = torch.cat([x] * 2), torch.cat([sigma] * 2), c_out
                sigma_shape = denoiser_sigma.shape
                denoiser_sigma = append_dims(denoiser_sigma, x.ndim)
                c_skip = 1.0 / (denoiser_sigma**2 + 1.0)
                c_out = -denoiser_sigma / (denoiser_sigma**2 + 1.0) ** 0.5
                c_in = 1.0 / (denoiser_sigma**2 + 1.0) ** 0.5
                c_noise = 0.25 * denoiser_sigma.log()
                c_noise = c_noise.reshape(sigma_shape)
                ## Denoise
                denoised = model.model(denoiser_input * c_in, c_noise, denoiser_c, **additional_model_inputs) * c_out + denoiser_input * c_skip # x_0
                ## CFG++ guidance
                x_u, x_c = denoised.chunk(2)
                return x_u, x_c 
            
            def CFG(x_u, x_c, scale):
                x_u = rearrange(x_u, "(b t) ... -> b t ...", t=num_frames)
                x_c = rearrange(x_c, "(b t) ... -> b t ...", t=num_frames)
                scale = torch.linspace(scale, scale, steps=num_frames).unsqueeze(0)
                scale = repeat(scale, "1 t -> b t", b=x_u.shape[0])
                scale = append_dims(scale, x_u.ndim).to(x_u.device)
                denoised =  rearrange(x_u + scale * (x_c - x_u), "b t ... -> (b t) ...")
                return denoised


            def CG(A, b, x, n_inner=5, eps=1e-5):
                r = b - A(x)
                p = r.clone()
                rsold = torch.sum(r * r, dim=[0, 1, 2, 3], keepdim=True)
                for _ in range(n_inner):
                    Ap = A(p)
                    denom = torch.sum(p * Ap, dim=[0, 1, 2, 3], keepdim=True) + 1e-12
                    a = rsold / denom
                    x = x + a * p
                    r = r - a * Ap
                    rsnew = torch.sum(r * r, dim=[0, 1, 2, 3], keepdim=True)
                    if rsnew.item() < eps**2:  
                        break
                    p = r + (rsnew / rsold) * p
                    rsold = rsnew
                return x

            def masking(x, index):
                mask = torch.zeros_like(x)
                mask[index, :, :, :] = 1
                return x * mask


            def temporal_laplacian(x):
                # x: (T, C, H, W)
                T = x.shape[0]
                out = torch.zeros_like(x)
                if T == 1:
                    return out
                out[0]    = x[0]    - x[1]
                out[1:-1] = 2*x[1:-1] - x[0:-2] - x[2:]
                out[-1]   = x[-1]   - x[-2]
                return out

            def temporal_biharmonic(x):
                return temporal_laplacian(temporal_laplacian(x))


            def DDS_boundary(x, n_inner, latent, lambda_end=1.0, end_index=-1):

                measurement = torch.zeros_like(x)
                measurement[end_index, :, :, :] = latent

                S  = lambda z: masking(z, end_index)   # S
                ST = lambda z: masking(z, end_index)   # S^T

                A  = lambda z: lambda_end * ST(S(z))   # λ_end S^T S
                b  = lambda_end * ST(measurement)      # λ_end S^T y

                return CG(A, b, x, n_inner=n_inner)


            def DDS_smooth_laplacian(x, n_inner, lambda_tv=0.1, tau=1.0):
                L2 = lambda z: temporal_laplacian(z)                 # (D^T D)^2
                A  = lambda z: z + tau * lambda_tv * L2(z)            # I + τλ_bi L2
                b  = x.clone()                                     
                return CG(A, b, x, n_inner=n_inner)

            def DDS_smooth_biharmonic(x, n_inner, lambda_bi=0.1, tau=1.0):
                L2 = lambda z: temporal_biharmonic(z)                 # (D^T D)^2
                A  = lambda z: z + tau * lambda_bi * L2(z)            # I + τλ_bi L2
                b  = x.clone()                                       
                return CG(A, b, x, n_inner=n_inner)

            
            x, s_in, sigmas, num_sigmas, cond, uc = model.sampler.prepare_sampling_loop(randn, c, uc, num_steps)
            
            for i in tqdm(model.sampler.get_sigma_gen(num_sigmas), total=num_sigmas-1):
                ## parameter setting
                gamma = (
                    min(model.sampler.s_churn / (num_sigmas - 1), 2**0.5 - 1)
                    if model.sampler.s_tmin <= sigmas[i] <= model.sampler.s_tmax
                    else 0.0
                )
                sigma = s_in * sigmas[i]
                next_sigma = s_in * sigmas[i + 1]
                sigma_hat = sigma * (gamma + 1.0)

                if gamma > 0:
                    eps = torch.randn_like(x) * model.sampler.s_noise
                    x = x + eps * append_dims(sigma_hat**2 - sigma**2, x.ndim) ** 0.5
                
                ### Forward sample ###
                # Prepare denoising parameters
                x_u, x_c = denoiser(x, sigma_hat, cond, uc)
                
                # CFG
                denoised = CFG(x_u, x_c, scale=cfg_scale) 
                
                # DDS update
                if my_args.dds:
                    denoised_hat = DDS_boundary(denoised, n_inner=5, latent=latent_end, lambda_end=my_args.lambda_end)
                    if my_args.lambda_tv > 0:
                        denoised_hat = DDS_smooth_laplacian(denoised_hat, n_inner=5, lambda_tv=my_args.lambda_tv)
                    elif my_args.lambda_bi > 0:
                        denoised_hat = DDS_smooth_biharmonic(denoised_hat, n_inner=5, lambda_bi=my_args.lambda_bi)
                else:
                    denoised_hat = denoised
                    
                # CFG++
                if my_args.cfgpp:
                    d = (x - x_u) / append_dims(sigma_hat, x.ndim)
                    dt = append_dims(next_sigma, x.ndim)
                    x = denoised_hat + d * dt
                else:
                    x = denoised_hat + append_dims(next_sigma / sigma_hat, x.ndim) * (x - denoised_hat)

                ###



                ### Backward sample ###
                if my_args.back:
                    # re-noise
                    eps = torch.randn_like(x) * model.sampler.s_noise
                    x = x + eps * append_dims(sigma_hat**2 - next_sigma**2, x.ndim) ** 0.5

                    #x = denoised_hat + append_dims(sigma_hat, x.ndim) * eps

                    x = torch.flip(x, dims=[0])
                    # Prepare denoising parameters
                    x_u, x_c = denoiser(x, sigma_hat, c_end, uc_end)
                    # CFG
                    denoised = CFG(x_u, x_c, scale=cfg_scale_flip)
                    
                    # DDS update
                    if my_args.dds:
                        denoised_hat = DDS_boundary(denoised, n_inner=5, latent=latent, lambda_end=my_args.lambda_end)
                        if my_args.lambda_tv > 0:
                            denoised_hat = DDS_smooth_laplacian(denoised_hat, n_inner=5, lambda_tv=my_args.lambda_tv)
                        elif my_args.lambda_bi > 0:
                            denoised_hat = DDS_smooth_biharmonic(denoised_hat, n_inner=5, lambda_bi=my_args.lambda_bi)
                    else:
                        denoised_hat = denoised
                        
                    # CFG++
                    if my_args.cfgpp:
                        d = (x - x_u) / append_dims(sigma_hat, x.ndim)
                        dt = append_dims(next_sigma, x.ndim)
                        x = denoised_hat + d * dt
                    else:
                        x = denoised_hat + append_dims(next_sigma / sigma_hat, x.ndim) * (x - denoised_hat)

                    x = torch.flip(x, dims=[0])
                    
                else:
                    pass
                ###

            samples_z = x
            model.en_and_decode_n_samples_a_time = decoding_t
            model = model.to(torch.float32)
            samples_x = model.decode_first_stage(samples_z)
            samples = torch.clamp((samples_x + 1.0) / 2.0, min=0.0, max=1.0)

            os.makedirs(output_folder, exist_ok=True)
            base_count = len(glob(os.path.join(output_folder, "*.gif")))

            samples = embed_watermark(samples)
            samples = filter(samples)
            vid = (
                (rearrange(samples, "t c h w -> t h w c") * 255)
                .cpu()
                .numpy()
                .astype(np.uint8)
            )
            video_path = os.path.join(output_folder, f"{base_count:06d}.gif")

            ## To gif
            images = [Image.fromarray(vid[i]) for i in range(vid.shape[0])]                
            duration = 125              
            images[0].save(video_path, save_all=True, append_images=images[1:], duration=duration, loop=0)


def get_unique_embedder_keys_from_conditioner(conditioner):
    return list(set([x.input_key for x in conditioner.embedders]))


def get_batch(keys, value_dict, N, T, device):
    batch = {}
    batch_uc = {}

    for key in keys:
        if key == "fps_id":
            batch[key] = (
                torch.tensor([value_dict["fps_id"]])
                .to(device)
                .repeat(int(math.prod(N)))
            )
        elif key == "motion_bucket_id":
            batch[key] = (
                torch.tensor([value_dict["motion_bucket_id"]])
                .to(device)
                .repeat(int(math.prod(N)))
            )
        elif key == "cond_aug":
            batch[key] = repeat(
                torch.tensor([value_dict["cond_aug"]]).to(device),
                "1 -> b",
                b=math.prod(N),
            )
        elif key == "cond_frames" or key == "cond_frames_without_noise":
            batch[key] = repeat(value_dict[key], "1 ... -> b ...", b=N[0])
        elif key == "polars_rad" or key == "azimuths_rad":
            batch[key] = torch.tensor(value_dict[key]).to(device).repeat(N[0])
        else:
            batch[key] = value_dict[key]

    if T is not None:
        batch["num_video_frames"] = T

    for key in batch.keys():
        if key not in batch_uc and isinstance(batch[key], torch.Tensor):
            batch_uc[key] = torch.clone(batch[key])
    return batch, batch_uc


def load_model(
    config: str,
    device: str,
    num_frames: int,
    num_steps: int,
    verbose: bool = False,
):
    config = OmegaConf.load(config)
    if device == "cuda":
        config.model.params.conditioner_config.params.emb_models[
            0
        ].params.open_clip_embedding_config.params.init_device = device

    config.model.params.sampler_config.params.verbose = verbose
    config.model.params.sampler_config.params.num_steps = num_steps
    config.model.params.sampler_config.params.guider_config.params.num_frames = (
        num_frames
    )
    if device == "cuda":
        with torch.device(device):
            model = instantiate_from_config(config.model).to(device).eval()
    else:
        model = instantiate_from_config(config.model).to(device).eval()
    
    model = model.to(torch.float16)

    filter = DeepFloydDataFiltering(verbose=False, device=device)
    return model, filter


import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Sampling configuration for SVD / EDM sampler")

    parser.add_argument("--dds", type=int, default=1,
                        help="Use DDS energy guidance (1 to enable, 0 to disable)")
    parser.add_argument("--cfgpp", type=int, default=1,
                        help="Use CFG++ update (1 to enable, 0 to disable)")
    
    parser.add_argument("--back", type=int, default=1)
    
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    my_args = get_args()

    lambda_end = 1.0
    lambda_bi = 0.0001
    my_args.lambda_bi = lambda_bi
    my_args.lambda_end = lambda_end
    my_args.lambda_tv = 0
    root = "test_data" # test_data dir path
    fps = 6

    output_folder = f'output_test_data' # output folder path
    sample(root=root, output_folder=output_folder, fps_id=fps, my_args=my_args)
    
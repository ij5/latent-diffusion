import argparse, os, sys, glob
import requests
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from einops import rearrange
from torchvision.utils import make_grid

sys.path.append('/app')

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler

from flask import Flask, jsonify, request, send_file

def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cuda")
    # sd = pl_sd["state_dict"]
    sd = pl_sd
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model


outdir = "outputs"
ddim_steps = 50
plms = False
ddim_eta = 0.0
n_iter = 1
H = 256
W = 256
n_samples = 1
scale = 5.0

if not os.path.isfile("latent-diffusion/models/ldm/text2img-large/model.ckpt"):
    r = requests.get("https://huggingface.co/multimodalart/compvis-latent-diffusion-text2img-large/resolve/main/txt2img-f8-large-fp16.ckpt", stream=True)
    with open("latent-diffusion/models/ldm/text2img-large/model.ckpt", "wb") as f:
        for chunk in r.iter_content(chunk_size=1024):
            f.write(chunk)

config = OmegaConf.load("latent-diffusion/configs/latent-diffusion/txt2img-1p4B-eval.yaml")
model = load_model_from_config(config, "latent-diffusion/models/ldm/text2img-large/model.ckpt")
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model = model.to(device)
if plms:
    sampler = PLMSSampler(model)
else:
    sampler = DDIMSampler(model)

os.makedirs(outdir, exist_ok=True)
sample_path = os.path.join(outdir, "samples")
os.makedirs(sample_path, exist_ok=True)
base_count = len(os.listdir(sample_path))

def generate(prompt: str):
    global base_count, sample_path
    all_samples=list()
    with torch.no_grad():
        with model.ema_scope():
            uc = None
            if scale != 1.0:
                uc = model.get_learned_conditioning(n_samples * [""])
            for n in trange(n_iter, desc="Sampling"):
                c = model.get_learned_conditioning(n_samples * [prompt])
                shape = [4, H//8, W//8]
                samples_ddim, _ = sampler.sample(S=ddim_steps,
                                                    conditioning=c,
                                                    batch_size=n_samples,
                                                    shape=shape,
                                                    verbose=False,
                                                    unconditional_guidance_scale=scale,
                                                    unconditional_conditioning=uc,
                                                    eta=ddim_eta)

                x_samples_ddim = model.decode_first_stage(samples_ddim)
                x_samples_ddim = torch.clamp((x_samples_ddim+1.0)/2.0, min=0.0, max=1.0)

                for x_sample in x_samples_ddim:
                    x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                    Image.fromarray(x_sample.astype(np.uint8)).save(os.path.join(sample_path, f"{base_count:04}.png"))
                    base_count += 1
                all_samples.append(x_samples_ddim)

    grid = torch.stack(all_samples, 0)
    grid = rearrange(grid, 'n b c h w -> (n b) c h w')
    grid = make_grid(grid, nrow=n_samples)

    grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
    filename = f'{prompt.replace(" ", "-")}.png'
    Image.fromarray(grid.astype(np.uint8)).save(os.path.join(outdir, filename))
    return filename



app = Flask(__name__)


@app.route('/', methods=["GET"])
def index():
    return jsonify({"status": "healthy"})

@app.route('/detect', methods=["POST"])
def detect():
    text = request.json["text"]
    filename = generate(text)
    return send_file(os.path.join(outdir, filename), mimetype='image/jpeg')

if __name__ == "__main__":
    app.run('0.0.0.0', 8080, False)

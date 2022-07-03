FROM python:3

WORKDIR /app

COPY requirements.txt ./

COPY latent-diffusion ./

RUN pip install -r requirements.txt
RUN pip install -e git+https://github.com/CompVis/taming-transformers.git@master#egg=taming-transformers
RUN pip install -e git+https://github.com/openai/CLIP.git@main#egg=clip

WORKDIR /app/latent-diffusion

RUN mkdir -p models/ldm/text2img-large/

# COPY model.ckpt models/ldm/text2img-large/

WORKDIR /app

COPY ldm ./

COPY run.py ./

ENTRYPOINT ["python", "./run.py"]
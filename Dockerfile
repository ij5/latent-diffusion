FROM nvidia/cuda:11.4.3-cudnn8-devel-ubuntu20.04

EXPOSE 8080

WORKDIR /app

RUN apt update

RUN apt install -y python3-pip

RUN apt install -y git

COPY requirements.txt ./

COPY latent-diffusion/ ./latent-diffusion/

RUN pip3 install -r requirements.txt
RUN pip3 install -e git+https://github.com/CompVis/taming-transformers.git@master#egg=taming-transformers
RUN pip3 install -e git+https://github.com/openai/CLIP.git@main#egg=clip

WORKDIR /app/latent-diffusion

RUN mkdir -p models/ldm/text2img-large/

# COPY model.ckpt models/ldm/text2img-large/

WORKDIR /app

COPY ldm/ ./ldm/

COPY run.py ./

ENTRYPOINT ["python3", "run.py"]

# git clone https://github.com/openai/shap-e.git
#pip install virtualenv
#virtualenv shap_env
#source shap_env/bin/activate
# pip install -e .     ##  cd /notebooks/shap-e
#pip install cchardet chardet pyyaml ipywidgets boto3 zipfile pymongo
#pip install 'git+https://github.com/facebookresearch/pytorch3d.git'
# python textto3d.py    ## cd /notebooks/shap-e/shap_e/examples

id = '01'

import os
import zipfile
import pymongo
import boto3
import torch

from shap_e.diffusion.sample import sample_latents
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
from shap_e.models.download import load_model, load_config
from shap_e.util.notebooks import create_pan_cameras, decode_latent_images, save_gif
from shap_e.util.notebooks import decode_latent_mesh

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

xm = load_model('transmitter', device=device)
model = load_model('text300M', device=device)
diffusion = diffusion_from_config(load_config('diffusion'))

# MongoDB connection
client = pymongo.MongoClient("mongodb+srv://vrchatAdmin:il4FA64i1Mbeo8Ay@cluster0.r5gre5i.mongodb.net")
db = client['prompt']
collection = db['textTo3d']

# Fetch prompt from MongoDB
prompt_document = collection.find_one({}, sort=[('date', pymongo.DESCENDING)])
prompt = prompt_document['prompt']

batch_size = 4
guidance_scale = 15.0

latents = sample_latents(
    batch_size=batch_size,
    model=model,
    diffusion=diffusion,
    guidance_scale=guidance_scale,
    model_kwargs=dict(texts=[prompt] * batch_size),
    progress=True,
    clip_denoised=True,
    use_fp16=True,
    use_karras=True,
    karras_steps=64,
    sigma_min=1e-3,
    sigma_max=160,
    s_churn=0,
)

render_mode = 'nerf'
size = 64

cameras = create_pan_cameras(size, device)
for i, latent in enumerate(latents):
    images = decode_latent_images(xm, latent, cameras, rendering_mode=render_mode) 
    save_gif(images, f'results/output_{i}.gif')

for i, latent in enumerate(latents):
    t = decode_latent_mesh(xm, latent).tri_mesh()
    with open(f'results/example_mesh_{i}.ply', 'wb') as f:
        t.write_ply(f)
    with open(f'results/example_mesh_{i}.obj', 'w') as f:
        t.write_obj(f)

        
# Create a zip file of the results
with zipfile.ZipFile('results.zip', 'w') as zipf:
    for filename in os.listdir('results'):
        zipf.write(os.path.join('results', filename), arcname=filename)


# Upload zip file to S3
s3 = boto3.client('s3', aws_access_key_id='AKIAVSYBEZQGCM6UUNMM', aws_secret_access_key='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')
bucket_name = 'meeting-bot-processed-files'
key = 'text_to_3d_'+id +'.zip'
s3.upload_file('results.zip', bucket_name, key)

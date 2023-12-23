# pip install torch🥺
!pip install --upgrade -q diffusers[torch]
!pip install transformers[torch]
!pip install -q peft

%cd /kaggle/working/
!git clone https://github.com/huggingface/diffusers
%cd diffusers
!pip install -q .


import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from diffusers import StableDiffusionPipeline
import requests

class CFG:
    gpt = 'sberbank-ai/rugpt3small_based_on_gpt2'
    gpt_weights = 'history.pt'
    
    SD = 'runwayml/stable-diffusion-v1-5'
    SD_weights = '/kaggle/input/weights-sd/weight'
    
    API_URL = "https://api-inference.huggingface.co/models/Helsinki-NLP/opus-mt-ru-en"
    headers = {"Authorization": "Bearer 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"}

class Inference:
    def __init__(self):
        self.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.gpt_tokenizer = GPT2Tokenizer.from_pretrained(CFG.gpt)
        self.gpt_model = GPT2LMHeadModel.from_pretrained(CFG.gpt).to(self.DEVICE)
#         model.load_state_dict(torch.load(CFG.gpt_weights))
        
        self.SD_pipe = StableDiffusionPipeline.from_pretrained(CFG.SD, torch_dtype=torch.float16)
        self.SD_pipe.unet.load_attn_procs(CFG.SD_weights)
        self.SD_pipe.to("cuda")
        
    def generate_img(self, prompt):
#         prompt = "Painting in style of Valentin Alexandrovich Serov a painting of a woman"
        image = self.SD_pipe(prompt, num_inference_steps=50, guidance_scale=7, height=512, width=512).images[0]
        image.save("img.png")
        
    def translate(self, text):
        response = requests.post(CFG.API_URL, headers=CFG.headers, json={"inputs": text})    
        return response.json()
    
    def generate_prompt(self, painting_name, author):
        text = f"КАРТИНА {painting_name}, автор {author}\nОПИСАНИЕ:"
        input_ids = self.gpt_tokenizer.encode(text, return_tensors="pt").to(self.DEVICE)
        self.gpt_model.eval()
        with torch.no_grad():
            out = self.gpt_model.generate(input_ids, do_sample=True, num_beams=2, temperature=1.5, top_p=0.9, max_length=100)

        return list(map(self.gpt_tokenizer.decode, out))[0]

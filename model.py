from PIL import Image
import requests
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch

from openai import OpenAI
client = OpenAI()

device = "cuda" if torch.cuda.is_available() else "cpu"

processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-opt-2.7b", load_in_8bit=True, device_map={"": 0}, torch_dtype=torch.float16
) 

url = "https://th.bing.com/th/id/OIP.bUyv4EfPip1VEQnDcUNtUwHaKl?rs=1&pid=ImgDetMain"
image = Image.open(requests.get(url, stream=True).raw)

inputs = processor(images=image, return_tensors="pt").to(device, torch.float16)

generated_ids = model.generate(**inputs , max_length=40 ,min_length=30) 
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
print(generated_text)

prompt = "This is the image description: "+generated_text+" with Emotion: fear generate instagram caption for it -->"


response = client.completions.create(
    model="gpt-3.5-turbo",
    prompt=prompt,
)

print(response['choices'][0]['text'].strip())


import csv
import os
import torch
from PIL import Image
import requests
import pandas as pd
import json 
from transformers import Blip2Processor, Blip2ForConditionalGeneration

device = "cuda" if torch.cuda.is_available() else "cpu"
processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-opt-2.7b", load_in_8bit=True, device_map={"": 0}, torch_dtype=torch.float16
)

image_directory = "images/"  

with open('captions.csv', 'r', encoding='utf-8') as csvfile:
    csvreader = csv.reader(csvfile)
    
    headers = next(csvreader)
    
    headers.append('generated_caption')
    
    updated_rows = [headers]
    
    for row in csvreader:
        image_path = row[1]
        
        image_file = os.path.join(image_directory +image_path +'.jpg')
        if os.path.exists(image_file):
            try:
                image = Image.open(image_file)
                
                inputs = processor(images=image, return_tensors="pt").to(device, torch.float16)
                
                generated_ids   = model.generate(**inputs, max_length=64, min_length=32)
                generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
                
                row.append(generated_text)
            except Exception as e:
                row.append(f"Error: {str(e)}")
        else:
            row.append("Image not found")
        
        updated_rows.append(row)

with open('updated_caption.csv', 'w', newline='', encoding='utf-8') as new_csvfile:
    csvwriter = csv.writer(new_csvfile)
    csvwriter.writerows(updated_rows)

print("Captions generated and saved in updated_caption.csv")


df = pd.read_csv("updated_caption.csv")

with open("captions_data.jsonl", "w") as f:
    for _, row in df.iterrows():
        prompt = f"generated_caption: {row['generated_caption']} Emotion: {row['Emotion']} -->"
        completion = f" {row['Caption']} ###"
        json_record = {"prompt": prompt, "completion": completion}
        f.write(json.dumps(json_record) + "\n")

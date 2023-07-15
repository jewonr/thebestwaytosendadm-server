from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

import torch

from kobert_tokenizer import KoBERTTokenizer
from transformers import AutoModelForSequenceClassification

from transformers import pipeline

app = FastAPI()

app.add_middleware(
  CORSMiddleware,
  allow_origins=["http://localhost:3000"],
  allow_credentials=True,
  allow_methods=["*"],
  allow_headers=["*"],
)

# 데이터 형식 클래스
class Classification(BaseModel):
  text: str

class Transform(BaseModel):
  text: str
  style: str

# 텍스트 분류용 tokenizer, model
classification_tokenizer = KoBERTTokenizer.from_pretrained('style_classification')
classification_model = AutoModelForSequenceClassification.from_pretrained('style_classification', num_labels=12)

# 텍스트 카테고리 
style_map = {
    'formal': '문어체',
    'informal': '구어체',
    'azae': '아재',
    'chat': '채팅',
    'choding': '초등학생',
    'emoticon': '이모티콘',
    'enfp': 'enfp',
    'gentle': '신사',
    'halbae': '할아버지',
    'halmae': '할머니',
    'joongding': '중학생',
    'sosim': '소심한',
}

text_styles = list(style_map.keys())

# 텍스트 변환 함수
nlg_pipeline = pipeline('text2text-generation', model="style_transform", tokenizer="gogamza/kobart-base-v2")

def transform_text(pipe, text, target_style, num_return_sequences=5, max_length=60):
  target_style_name = style_map[target_style]
  text = f"{target_style_name} 말투로 변환:{text}"

  out = pipe(text, num_return_sequences=num_return_sequences, max_length=max_length)
  return out



@app.post("/classification")
def classification(data: Classification):
  inputs = classification_tokenizer(data.text, padding=True, truncation=True, return_tensors="pt")

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  classification_model.to(device)
  inputs = {k: v.to(device) for k, v in inputs.items()}

  outputs = classification_model(**inputs)
  predictions = torch.argmax(outputs.logits, dim=1)

  print(style_map[text_styles[predictions.item()]])

  return text_styles[predictions.item()]

@app.post("/transform")
def classification(data: Transform):
  transformed_text = transform_text(nlg_pipeline, data.text, data.style, num_return_sequences=1, max_length=60)
  print(transformed_text[0]["generated_text"])
  return transformed_text[0]["generated_text"]
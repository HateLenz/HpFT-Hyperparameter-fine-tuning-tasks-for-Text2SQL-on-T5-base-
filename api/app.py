# app.py: FastAPI 实现模型的API服务
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

# 加载模型和分词器
MODEL_DIR = 'models/saved_model'
model = T5ForConditionalGeneration.from_pretrained(MODEL_DIR)
tokenizer = T5Tokenizer.from_pretrained(MODEL_DIR)

# 初始化FastAPI应用
app = FastAPI()

# 定义请求数据模型
class QueryRequest(BaseModel):
    question: str

# 定义响应数据模型
class QueryResponse(BaseModel):
    sql: str

# 创建一个POST端点，用于生成SQL
@app.post("/predict/", response_model=QueryResponse)
async def predict(request: QueryRequest):
    try:
        # 使用模型进行推理
        inputs = tokenizer("translate: " + request.question, return_tensors="pt")
        outputs = model.generate(**inputs)
        predicted_sql = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return QueryResponse(sql=predicted_sql)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # 启动API服务器的命令：
    # uvicorn app:app --reload
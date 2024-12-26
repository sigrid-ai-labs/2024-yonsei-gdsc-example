from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# ------------------------------
# 0) 사전 준비 (모델 로드)
# ------------------------------
checkpoint_dir = "/Users/sigridjineth/Desktop/work/2024-yonsei-gdsc-example/step3/router_bert_ckpt/checkpoint-150"
tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint_dir)
model.eval()

label2id = {"gpt-o1": 0, "Claude": 1, "gemini-mini": 2}
id2label = {v: k for k, v in label2id.items()}


def infer_llm_route(query_text: str) -> str:
    inputs = tokenizer(query_text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    pred_id = torch.argmax(outputs.logits, dim=-1).item()
    return id2label[pred_id]


# ------------------------------
# 1) FastAPI 생성
# ------------------------------
app = FastAPI(title="LLM Router API", version="1.0")


# ------------------------------
# 2) Request Body 모델
# ------------------------------
class QueryRequest(BaseModel):
    text: str


# ------------------------------
# 3) POST 엔드포인트
# ------------------------------
@app.post("/route")
def route_query(req: QueryRequest):
    """
    - Body: {"text": "I want help with my essay."}
    - Returns: {"routing_label": "Claude"}
    """
    routing_label = infer_llm_route(req.text)
    return {"routing_label": routing_label}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

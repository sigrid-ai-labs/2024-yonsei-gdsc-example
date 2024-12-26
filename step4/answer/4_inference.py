import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 라벨 매핑(우리가 정의했던 3개 클래스)
label2id = {"gpt-o1": 0, "Claude": 1, "gemini-mini": 2}
id2label = {v: k for k, v in label2id.items()}

# 1. 모델 & 토크나이저 로드
checkpoint_dir = "/Users/sigridjineth/Desktop/work/2024-yonsei-gdsc-example/step3/router_bert_ckpt/checkpoint-150"  # Trainer 학습 후 save_model() 경로
tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint_dir)

model.eval()


def infer_llm_route(query_text: str) -> str:
    """
    사용자 쿼리(query_text)를 입력받아,
    "gpt-o1", "Claude", "gemini-mini" 중 하나의 라우팅 라벨을 반환
    """
    inputs = tokenizer(query_text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    pred_id = torch.argmax(outputs.logits, dim=-1).item()
    return id2label[pred_id]


if __name__ == "__main__":
    # 간단한 테스트
    test_query = "ModernBERT 파인튜닝 방법"
    routed_label = infer_llm_route(test_query)
    print(f"[Query] {test_query}")
    print(f"[Routed to] {routed_label}")  # gpt-o1

import torch
from datasets import load_from_disk
from transformers import (
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    AutoTokenizer,
)
from sklearn.metrics import accuracy_score, f1_score

# ==========================================
# 1) 전처리 완료 Dataset 로드
# ==========================================
# 저장 경로에서 토큰화+레이블 인코딩까지 완료된 Dataset 불러오기
save_path = "./step2/encoded_dataset"  # 예: 이전 단계에서 save_to_disk()
dataset = load_from_disk(save_path)

# (선택) Train/Validation 분할
train_test = dataset.train_test_split(test_size=0.2, seed=42)
train_dataset = train_test["train"]
eval_dataset = train_test["test"]

# ==========================================
# 2) 라우팅용 분류 모델 (3개 클래스) 준비
# ==========================================
model_id = "sigridjineth/ModernBERT-korean-large-preview"

# 라벨 매핑 - 실제 LLM 라우팅 목적
label2id = {"gpt-o1": 0, "Claude": 1, "gemini-mini": 2}
id2label = {v: k for k, v in label2id.items()}

# 모델 로드
model = AutoModelForSequenceClassification.from_pretrained(
    model_id,
    num_labels=len(label2id),
    label2id=label2id,
    id2label=id2label,
)

# 추론/파이프라인용 토크나이저(필요 시)
tokenizer = AutoTokenizer.from_pretrained(model_id)


# ==========================================
# 3) Metrics (평가지표 함수)
# ==========================================
def compute_metrics(eval_pred):
    # eval_pred: (predictions, labels)
    predictions, labels = eval_pred
    preds = torch.argmax(torch.tensor(predictions), dim=-1)
    labels = torch.tensor(labels)

    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="weighted")
    return {"accuracy": acc, "f1": f1}


# ==========================================
# 4) Train 설정 (TensorBoard 로그 사용)
# ==========================================
training_args = TrainingArguments(
    output_dir="./step3/router_bert_ckpt",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=5e-5,
    logging_dir="./step3/logs",  # TensorBoard 로그 디렉토리
    logging_steps=10,
    report_to="tensorboard",  # wandb 대신 tensorboard 사용
    bf16=True,
    optim="adamw_torch_fused",
)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# 실제 학습 실행
train_result = trainer.train()

# ==========================================
# 5) 평가 & 결과 확인
# ==========================================
eval_result = trainer.evaluate()
print("[Eval] metrics:", eval_result)

# 예: tensorboard --logdir=./logs 로 학습 로그 시각화 가능

# ==========================================
# 6) 추론
#
# 1. **Dataset 로드**
#    - 이전 단계까지 `CSV → DataFrame → Dataset` 변환 및 토큰화, 레이블 인코딩 완료 후 `save_to_disk()`로 저장한 데이터셋을 다시 불러옵니다.
# 2. **라우팅 모델 (3 클래스)**
#    - 우리는 **“Information/Coding → gpt-o1”, “Writing/Language → Claude”, “기타 → gemini-mini”**라는 **3개 클래스**로 분류하려는 목표가 있습니다.
#    - `label2id`, `id2label`를 {gpt-o1:0, Claude:1, gemini-mini:2}로 설정하고, 이 분류 모델(`AutoModelForSequenceClassification`)을 준비합니다.
# 3. **Metrics**
#    - `compute_metrics()`에서 **정확도(accuracy)**, **가중치 평균 F1(f1_score)**를 계산.
#    - 필요 시 Precision, Recall, Confusion Matrix 등 추가 가능
# 4. **Train**
#    - **Trainer** 사용 시, `report_to="tensorboard"`로 설정하면 학습 로그(`loss`, `eval_loss`, `eval_accuracy` 등)를 TensorBoard로 확인 가능
#    - **학습 중/끝난 후** `tensorboard --logdir=./logs`로 대시보드 접속 가능
# 5. **추론 → LLM 라우팅**
#    - 최종적으로 이 모델은 “쿼리” 입력 → “gpt-o1/Claude/gemini-mini” 출력을 내므로, **여기서 나온 레이블**에 따라 후속 처리를 할 수 있겠죠.
#
# ### **추가 참고 사항**
# - 실제 대규모 데이터셋에서는 **에폭 수, 배치 크기, 러닝레이트** 등을 조정하며 최적화 필요
# - ModernBERT(`answerdotai/ModernBERT-base`)로 **bf16, FlashAttention** 등 활용 시 **학습/추론 속도 상승** 및 메모리 이점
# - 모델 학습 완수 후, **`trainer.save_model()`** 등으로 최종 체크포인트를 저장하거나 **추론 파이프라인**(Hugging Face pipeline)을 구성해 실제 서비스 라우팅에 적용할 수 있습니다.
# ==========================================

print("[Train] metrics:", train_result)

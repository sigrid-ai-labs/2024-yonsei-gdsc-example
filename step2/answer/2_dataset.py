from datasets import Dataset
from transformers import AutoTokenizer
import pandas as pd

# 1. CSV 읽어서 df 생성 (이미 llm 컬럼 존재)
csv_path = "./step1/data/labeled_data.csv"
df = pd.read_csv(csv_path)

# 2. df를 Dataset으로 변환
dataset = Dataset.from_pandas(df)

# 3. 토크나이저 준비 (https://huggingface.co/sigridjineth/ModernBERT-korean-large-preview)
checkpoint = "sigridjineth/ModernBERT-korean-large-preview"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# 4. 토큰화 함수
def tokenize_function(examples):
    return tokenizer(examples["user_message"], truncation=True, max_length=128)

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# 5. llm → 정수 레이블로 변환
label2id = {"gpt-o1": 0, "Claude": 1, "gemini-mini": 2}
def encode_label(examples):
    examples["labels"] = [label2id[val] for val in examples["llm"]]
    return examples

encoded_dataset = tokenized_dataset.map(encode_label, batched=True)

print("Tokenized example:")
example_text = tokenizer.decode(tokenized_dataset[0]["input_ids"])
print(example_text)

# 6. 토큰화 & 레이블 인코딩 완료된 Dataset 저장
save_path = "./step2/encoded_dataset"
encoded_dataset.save_to_disk(save_path)

# 7. 추후 사용 예시
# from datasets import load_from_disk
# restored_dataset = load_from_disk(save_path)
# print(restored_dataset[0])  # 토큰화 정보, labels 등 확인

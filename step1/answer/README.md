## 코드 예시

```python
import pandas as pd

# CSV 파일 경로
csv_path = "path/to/your_data.csv"

# CSV 불러오기
df = pd.read_csv(csv_path)

# 규칙 기반 label 매핑 함수
def map_to_label(row):
    # row["intent"], row["category"] 등을 참조하여 로직 구성
    intent = row["intent"]

    # 예시: intent 이름에 특정 키워드가 있으면 gpt-o1, etc.
    # 실제 사용 시에는 정확한 조건(If ~ elif ~ else)을 세분화하세요
    if intent in ["Information Seeking", "Coding"]:
        return "gpt-o1"
    elif intent in ["Writing", "Language"]:
        return "Claude"
    else:
        return "gemini-mini"

# apply 함수를 이용해 각 row마다 label 부여
df["label"] = df.apply(map_to_label, axis=1)

# 결과 확인
print(df.head())

# 필요하다면 새로운 CSV로 저장
df.to_csv("labeled_data.csv", index=False)
```

### 코드 설명

1. **pandas를 사용해 CSV 로드**:
   - `pd.read_csv(csv_path)`로 파일을 DataFrame(`df`) 형태로 불러옵니다.

2. **`map_to_label` 함수**:
   - 각 행(row)의 `intent` 컬럼 등을 확인해, 사전에 정의한 규칙에 따라 `"gpt-o1"`, `"Claude"`, `"gemini-mini"`를 반환합니다.

3. **DataFrame에 `label` 컬럼 추가**:
   - `df.apply(map_to_label, axis=1)`로 모든 행에 대해 함수를 적용, 결과를 `df["label"]`로 저장.

4. **결과 확인 & 저장**:
   - `print(df.head())`로 상위 5행을 확인하거나,
   - 필요 시 `df.to_csv("labeled_data.csv")`로 레이블이 포함된 데이터를 CSV로 저장할 수 있습니다.

---

## 정리

- 이렇게 만들어진 `df`에는 이제 `label` 컬럼이 생겼습니다.
- 이 레이블을 이용해 **모델 학습**(예: BERT 분류 모델) 또는 **규칙 기반 라우팅** 등 다양한 처리를 할 수 있습니다. 
- 만약 `intent + category` 두 컬럼을 모두 보고 싶다면 조건문을 더욱 세분화하거나, 혹은 직접 **머신러닝 모델**을 학습해 라벨을 예측하는 방식으로 확장할 수 있습니다.

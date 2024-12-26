# 미션의 목표, 요구사항
## 목표

이전 단계에서 CSV 파일을 불러와서, intent와 category 등의 정보를 토대로 llm 컬럼(예: "gpt-o1", "Claude", "gemini-mini")을 생성했습니다.
이제 이 확장된 DataFrame을 활용하여 추가적인 처리를 진행하려고 합니다.

## 요구사항

CSV + llm 컬럼이 있는 상태에서, 다음과 같은 과정을 시도할 수 있습니다.
* Hugging Face datasets 형식으로 변환 (모델 학습/추론 편의)
* 토큰화 및 전처리: llm이 최종 레이블이 되도록 준비
* Hugging Face datasets에서는 save_to_disk() 방식으로 Dataset을 디스크에 저장한 뒤, **load_from_disk()**로 재로딩이 가능합니다. 
* CSV로 내보낼 수도 있지만, input_ids, attention_mask 등 배열 데이터를 CSV로 저장하면 불편하므로, Hugging Face Dataset 포맷(Arrow)을 사용하는 편이 좋습니다.

## 참고 링크
* https://huggingface.co/sigridjineth/ModernBERT-korean-large-preview
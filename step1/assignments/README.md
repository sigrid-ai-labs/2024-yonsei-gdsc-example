## 구현 목표
* **pandas**로 CSV 파일을 불러온 뒤
* `intent`와 `category` 정보 등을 참조하여 새 레이블(`label`) 컬럼을 추가하는 미션 입니다.

---

## 1. CSV 파일 예시 구조

현재 `data/example.csv` 폴더에 존재하는 CSV에 다음과 같은 컬럼들이 있다고 합시다.

| user_message                       | intent             | category             |
|-----------------------------------|--------------------|----------------------|
| "일본 혼슈에 태풍이 상륙했어?"        | News and Updates   | Information Seeking  |
| "protools aaeエラー 941の原因は？" | Technical Issues   | Information Seeking  |
| "이미 변작된 전산기록을 출력한 문서는 무슨문서인지?" | Fact-Finding       | Information Seeking  |
| "수소 관련 유망주"                 | Strategic Analytics| Information Seeking  |
| ...                               | ...                | ...                  |

`intent`와 `category`를 보고 최종 라우팅할 레이블을 `gpt-o1`, `claude`, `gemini-mini` 중 하나로 지정하려고 합니다.

---

## 2. 레이블 매핑 규칙 정의

예를 들어 다음과 같은 간단한 규칙을 가정해 보겠습니다.

1. `(intent이 Information Seeking 또는 Coding)`: `"gpt-o1"`  
2. `(intent이 Writing 또는 Language)`: `"Claude"`  
3. 그 밖의 모든 경우: `"gemini-mini"`

> 실제 비즈니스 로직에 따라 자유롭게 규칙(혹은 ML 모델)을 적용하면 됩니다.
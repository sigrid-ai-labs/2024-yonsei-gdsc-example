# 2024-responsible-ai-in-action-gdsc-example

## [Hands-on] 나만의 RAG Agentic 모델 만들기

본 실습은 RAG 과정에서 실제 프로덕션 환경에서 활용할 수 있는 LLM 라우팅 시스템을 구현해야 하는데 RPS 100 기준으로 인퍼런스는 50ms 이하로 달성되어야 하는 요구사항이 주어졌다고 생각합니다.
ModernBERT를 활용하여, 사용자의 질의 데이터가 주어지고 이를 바탕으로 어떤 LLM에 라우팅을 하면 좋을지 평가하는 300M 짜리 BERT classifier 를 만들어봅니다.

## 교수자

- Sigrid Jin (Jin Hyung Park)

## 학습 목표

- LLM 라우팅의 개념과 필요성 이해
- 데이터 전처리와 모델 학습 과정 실습
- 실제 서비스에 적용 가능한 라우팅 시스템 구축
- 성능 평가와 최적화 방법 습득

## 수강 대상

- Python 기초 프로그래밍 능력 보유자
- LLM 서비스 구축에 관심 있는 개발자

## 커리큘럼
- step0 : 파이썬 개발환경 `uv` 익히기
- step1 : 허깅페이스 모델 로드하기
- step2 : Dataset 데이터 전처리하기
- step3 : 모델 학습 및 평가하기
- step4 : FastAPI 를 이용한 모델 인퍼런스 하기
- step5 : 모델 성능 개선하기 (자율 과제)

## 실습 프로젝트

- TPU 환경이 제공되는데 일정 기간 동안 수강생들이 사용 가능 합니다.
- 과제를 제출한 사람들에 한하여 온라인 추가 세션이 1회 진행 됩니다.
    - 간단한 라우터 의도 분류기 구현
    - 성능 평가 리포트 작성
    - 코드 리뷰 세션

## 필요 도구와 환경

- Python 3.11 이상 구동 가능한 노트북
- Google Cloud 에서 TPU 환경이 제공 됩니다.
    - https://docs.google.com/presentation/d/1fnQNauWcxgt5eqhTAuaBGn2giNQU6f_f_fv3R7nAvp4/edit?resourcekey=0-dlnAU4LYN8QV27uVMLeIQA

## 참고 자료

- ModernBERT 공식 문서
    - https://www.philschmid.de/fine-tune-modern-bert-in-2025
- Hugging Face 트랜스포머 문서
    - https://huggingface.co/blog/modernbert
- 튜토리얼 일부
    - https://github.com/AnswerDotAI/ModernBERT/blob/main/examples/finetune_modernbert_on_glue.ipynb
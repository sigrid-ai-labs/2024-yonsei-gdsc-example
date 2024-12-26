# 2024-yonsei-gdsc-example

## [Hands-on] 나만의 RAG Agentic 모델 만들기

본 실습은 RAG 과정에서 실제 프로덕션 환경에서 활용할 수 있는 LLM 라우팅 시스템을 구현하는 실전 프로젝트입니다. ModernBERT를 활용하여, 사용자의 질의 데이터가 주어지고 이를 바탕으로 어떤 LLM에 라우팅을 하면 좋을지 평가하는 300M 짜리 BERT classifier 를 만들어봅니다.

## 강의자

- Sigrid Jin
    - ML DevRel Engineer @ Sionic AI
    - Product Engineer @ Wanot.AI

## 학습 목표

- LLM 라우팅의 개념과 필요성 이해
- 데이터 전처리와 모델 학습 과정 실습
- 실제 서비스에 적용 가능한 라우팅 시스템 구축
- 성능 평가와 최적화 방법 습득

## 수강 대상

- Python 기초 프로그래밍 능력 보유자
- LLM 서비스 구축에 관심 있는 개발자

## 커리큘럼

### Part 1: 이론과 설계

1. RAG 에서 필요한 LLM 라우팅 시스템 개요
    - LLM 라우팅의 필요성
    - 시스템 아키텍처 설계
2. ModernBERT 이해하기
    - BERT vs ModernBERT
    - 모델 아키텍처와 특징
    - 분류 태스크에서의 장점

### Part 2: 데이터 준비와 전처리

1. 데이터셋 구축
    - 데이터 수집과 레이블링
    - 클래스 불균형 처리
    - 데이터 품질 관리
2. 전처리 파이프라인 구현
    - 텍스트 정제
    - 토큰화와 인코딩
    - 데이터 증강 기법

### Part 3: 모델 구현과 학습

1. 모델 구현
    - ModernBERT 설정
    - 분류 헤드 구현
    - 학습 파이프라인 구축
2. 모델 학습과 평가
    - 하이퍼파라미터 튜닝
    - 교차 검증
    - 성능 평가와 분석
3. 서비스 배포
    - API 서버 구축
    - 모니터링 시스템 구축
    - 에러 핸들링

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

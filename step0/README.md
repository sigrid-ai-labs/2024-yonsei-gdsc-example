# Step 1. Python 개발 환경 익히기
* uv 를 통하여 파이썬 개발 환경을 익히는 방법에 대해 학습 합니다.

## 관련 링크
* https://github.com/astral-sh/uv

## 도전 과제
* `uv` 를 이용하여 프로젝트를 셋업 합니다.
* 아래의 의존성을 `uv` 를 이용하여 설치합니다.
```
# Install Pytorch & other libraries
%pip install "torch==2.4.1" tensorboard 
%pip install flash-attn "setuptools<71.0.0" scikit-learn 
 
# Install Hugging Face libraries
%pip install  --upgrade \
  "datasets==3.1.0" \
  "accelerate==1.2.1" \
  "hf-transfer==0.1.8"
  #"transformers==4.47.1" \
 
# ModernBERT is not yet available in an official release, so we need to install it from github
%pip install "git+https://github.com/huggingface/transformers.git@6e0515e99c39444caae39472ee1b2fd76ece32f1" --upgrade
```
* `uv run hello.py` 를 통하여 실행해봅니다.
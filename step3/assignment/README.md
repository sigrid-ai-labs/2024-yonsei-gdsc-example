# 미션의 목표, 요구사항

우리가 궁극적으로 원하는 것은 **“LLM Router 모델”**입니다.

* CSV에 존재하는 쿼리(user_message 등)를 받아, 3개 클래스("gpt-o1", "Claude", "gemini-mini")로 정확히 분류하여, 해당 LLM으로 라우팅(전달)하는 것이 최종 목표입니다. 
* 이제, 전처리 완료된 Dataset(CSV → Dataset 변환 & 토큰화 & llm 레이블 인코딩 완료)을 실제 학습(Train) 및 평가하고, 
* TensorBoard(PyTorch/TensorBoard)로 로그를 확인하는 코드를 작성 해봅시다.

Note:
* 여기서는 wandb 사용 없이, TensorBoard만 사용합니다. 
* 학습 로그는 report_to="tensorboard", logging_dir="./logs" 설정으로 남기며, 
* 실행 후 tensorboard --logdir=./logs 명령어로 대시보드에서 학습 과정을 모니터링할 수 있습니다.
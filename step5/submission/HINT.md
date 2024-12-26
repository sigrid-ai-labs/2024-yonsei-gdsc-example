# 힌트

1. **클래스 매핑 규칙**  
   - `lv2_intent_to_prompt_category` 함수로 **다양한 intent**(예: `"Fact-Finding"`, `"Advice and Recommendation"`)를 **5개 대분류**  
     (`"Information Seeking"`, `"Coding"`, `"Writing"`, `"Language"`, `"Others"`)로 매핑하고 있습니다.  
   - 실제 라벨링 과정에서 **분류 규칙**을 단순 if-else로 처리하는데, **세분화된 intent**가 **최종 5대 분류**에 올바로 매핑되는지 **재점검**해보세요.  
   ```python
    import util
    
    import numpy as np
    from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
    
    # 데이터 읽기
    file_path = './data.csv'
    all_data, class_dic = util.read_and_process_csv(file_path)
    
    # 분할
    train_data, test_data = util.split_data(all_data, train_ratio=0.5)
    
    # 전체 데이터
    # train_data = test_data = all_data
    
    print(f"훈련 데이터 크기: {len(train_data)}")
    print(f"테스트 데이터 크기: {len(test_data)}")
    
    # 각 클래스별로 데이터를 나눠둠 (학습데이터만)
    class_data = {class_id: [] for class_id in class_dic.values()}
    for t in train_data:
        class_data[t[2]].append(t)
    
    for class_id, data in class_data.items():
        print(f"Class {class_id}: {len(data)} samples")
        
    # 훈련 데이터 크기: 1991
    # 테스트 데이터 크기: 1991
    # Class 0: 11 samples
    # Class 1: 1523 samples
    # Class 2: 49 samples
    # Class 3: 83 samples
    # Class 4: 325 samples
    
    # 제로샷용 모델 로딩
    from sentence_transformers import (
        SentenceTransformer)
    embed_model = SentenceTransformer(
        "BAAI/bge-m3") #, use_fp16=True)
    
    # 클래스별 임베딩 계산
    class_embeddings = {class_id: [] for class_id in class_dic.values()}
    
    def lv2_intent_to_prompt_category(lv2_intent: str):
        intent_mapping = {
            "Fact-Finding": "Information Seeking",
            "Product or Service Inquiry": "Information Seeking",
            "News and Updates": "Information Seeking",
            "Technical Issues": "Information Seeking",
            "Advice and Recommendation": "Information Seeking",
            "Strategic Analytics": "Information Seeking",
            "Tutorials and How-Tos": "Information Seeking",
            "Calculation and Logical Solving": "Information Seeking",
            "Knowledge Abstraction": "Writing",
            "Content Generation": "Writing",
            "Editing and Proofreading": "Writing",
            "Language Learning": "Language",
            "Academic Learning": "Language",
            "Translation": "Language",
            "Code Generation and Refactoring": "Coding",
            "Debugging Assistance": "Coding",
            "Development Assistance": "Coding"
        }
        return intent_mapping.get(lv2_intent, "Others")
    ```

   - 이후, 학습된 라우터 모델이 예측에 실패했을 때, **어떤 intent**가 **어떤 대분류**로 잘못 매핑되었는지 **에러 분석**(예: Confusion Matrix) 시 유용합니다.
   ```
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    print(f"precision: {precision:.2f}")
    print(f"recall: {recall:.2f}")
    print(f"F1 score: {f1:.2f}")
    
    # 혼동 행렬 출력
    cm = confusion_matrix(y_true, y_pred)
    print("\nconfusion_matrix:")
    print(cm)
    precision: 0.89
    recall: 0.86
    F1 score: 0.87
    
    confusion_matrix:
    [[  14    2    0    0    0]
     [  13 1403   40   37   70]
     [   0    7   24    5   12]
     [   1    9   10   31   24]
     [   1   28    8    4  248]]
    ```

2. **데이터 분할과 클래스 불균형**  
   - 코드에서 `train_data`, `test_data`를 50:50으로 분할하고 있습니다.  
   - 클래스별 데이터가 매우 적은 경우(`"Coding"`, `"Others"` 등), 오분류가 잦을 수 있습니다. **증강**(예: `Back Translation`)이나 **샘플링**(Over-sampling, Re-weighting) 기법을 고려해보세요.
    ```
    # preprompt = """<|im_start|>system
    # You are an helpful assistant.<|im_end|>
    # preprompt_before_1shot = """<BOS_TOKEN><|START_OF_TURN_TOKEN|><|USER_TOKEN|>
    # Which is the most appropriate purpose of the user input: ```"""
    
    # preprompt_after_1shot = """```
    # PURPOSE_CLASS = ['Information Seeking', 'Calculation and Logical Solving', 'Knowledge Abstraction for Writing', 'Content Generation', 'Editing and Proofreading', 'Language Learning', 'Translation', 'Coding', 'Debugging', 'Others']
    # <|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>
    # The most appropriate purpose class of given sentence is '"""
    
    # postprompt_before_1shot = """'.
    # <|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|USER_TOKEN|>
    # Which is the most appropriate purpose of the user input: ```"""
    
    # postprompt_after_1shot = """```
    # PURPOSE_CLASS = ['Information Seeking', 'Calculation and Logical Solving', 'Knowledge Abstraction for Writing', 'Content Generation', 'Editing and Proofreading', 'Language Learning', 'Translation', 'Coding', 'Debugging', 'Others'.]
    # <|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>
    # The most appropriate purpose class of given sentence is '"""
    
    preprompt = "[|system|]You are an helpful assistant.[|endofturn|]"
    preprompt_before_1shot = """[|user|]Which is the most appropriate purpose of the user input: ```"""
    preprompt_after_1shot = """```
    PURPOSE_CLASS = ['Information Seeking', 'Calculation and Logical Solving', 'Content Generation in Writing', 'Editing and Proofreading', 'Language Learning', 'Translation', 'Coding', 'Debugging', 'Others']
    [|assistant|]The most appropriate purpose class of given sentence is '"""
    postprompt_before_1shot = """'.
    [|user|]Which is the most appropriate purpose of the user input: ```"""
    postprompt_after_1shot = """```
    PURPOSE_CLASS = ['Information Seeking', 'Calculation and Logical Solving', 'Content Generation in Writing', 'Editing and Proofreading', 'Language Learning', 'Translation', 'Coding', 'Debugging', 'Others'.]
    [|assistant|]The most appropriate purpose class of given sentence is '"""
    ```

3. **임베딩 접근 vs 파인튜닝 접근**  
   - 예시 코드에서는 `BGEM3FlagModel`(bge-m3)나 `SentenceTransformer`를 이용해 **임베딩 기반**(유사도 비교)으로 분류합니다.  
   - 이 방법은 데이터가 적을 때 편리하나, **성능 한계**가 있을 수 있습니다.  
   - **파인튜닝**(모델 직접 학습)이나, **Triplet Loss**(예: `triplet_data`) 등을 통해 **임베딩을 맞춤형으로 재학습**하면 더 나은 성능이 나올 수도 있습니다.

4. **Confusion Matrix & F1 Score**  
   - 출력 예시에 보면 Confusion Matrix가 5×5 형태로 나옵니다.  
   - **어떤 클래스**가 **어떤 다른 클래스**로 가장 자주 혼동되는지 살펴보면, **데이터 증강/샘플 보강** 아이디어를 얻을 수 있습니다.  
   - Precision, Recall, F1 중 **어느 지표**가 낮은지에 따라 보완책(예: Recall이 낮으면 소수 클래스에 대한 샘플 확대)이 달라집니다.

5. **추가 힌트: 데이터 & 인퍼런스 속도**  
   - **데이터 편향**:  
     - `"Information Seeking"`이 대부분(주로 1,400샘플 이상)인 반면, `"Coding"`, `"Others"`, `"Language"`, `"Writing"`은 훨씬 적습니다.  
     - 이 불균형을 **해결**하거나 **감안**해야 모델이 안정적인 예측을 합니다.  
   - **인퍼런스 속도**:  
     - 현재 임베딩을 구해 **유사도(dot product)**를 계산하는 방식은, **대량 요청** 시 느려질 수 있습니다.  
     - 배치 추론, GPU 병렬 처리, ONNX 최적화, Distil 모델 활용 등을 검토해, **RPS 100**에도 **50ms 이하** 응답이 가능한 구조를 설계해보세요.

6. **데이터 전처리 & 예시**  
   - `read_and_process_csv` 함수에서 CSV를 읽을 때, **헤더 스킵**, **strip()** 처리, **매핑**을 진행합니다.  
   - 실데이터 상황에선 **불필요한 공백/특수문자** 제거, **입력 문장 길이 분포** 등을 분석해볼 필요가 있습니다.

7. **추가적인 아이디어**  
   - **Triplet Loss** 훈련 시, anchor-positive-negative를 **정교하게 선정**하면, **의도 유사 문장**끼리는 가깝게, **전혀 다른 의도**끼리는 멀어지도록 임베딩을 학습할 수 있습니다.  
   - **Hugging Face Trainer**나 **SentenceTransformerTrainer**에서 **torch.compile**, **bf16** 등을 활용해 **학습/추론 최적화**도 시도해보세요.


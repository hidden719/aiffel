🔑 **PRT(Peer Review Template)**

[O]  **1. 주어진 문제를 해결하는 완성된 코드가 제출되었나요?**
- 문제에서 요구하는 최종 결과물이 첨부되었는지 확인
	- (문제를 해결하는 완성된 코드란 프로젝트 루브릭 3개 중 2개, 퀘스트 문제 요구조건 등을 지칭)
- 해당 조건을 만족하는 코드를 캡쳐해 근거로 첨부

> 독립된 test data 검증 결과를 제공해 주셨습니다.
12/12 - 0s - loss: 4.5059 - accuracy: 0.3499 - 311ms/epoch - 26ms/step
test_loss: 4.505851745605469 
test_accuracy: 0.3498622477054596
  
    
[O]  **2. 전체 코드에서 가장 핵심적이거나 가장 복잡하고 이해하기 어려운 부분에 작성된 
	주석 또는 doc string을 보고 해당 코드가 잘 이해되었나요?**
- 해당 코드 블럭에 doc string/annotation이 달려 있는지 확인
- 해당 코드가 무슨 기능을 하는지, 왜 그렇게 짜여진건지, 작동 메커니즘이 뭔지 기술.
- 주석을 보고 코드 이해가 잘 되었는지 확인
- 잘 작성되었다고 생각되는 부분을 캡쳐해 근거로 첨부합니다.

> 잘못 예측된 결과에 대해서 확인을 진행해 주셨습니다.
import random
wrong_predict_list=[]
for i, _ in enumerate(predicted_labels):
    # i번째 test_labels과 y_test이 다른 경우만 모아 봅시다. 
    if predicted_labels[i] != y_test[i]:
        wrong_predict_list.append(i)

# wrong_predict_list 에서 랜덤하게 5개만 뽑아봅시다.
samples = random.choices(population=wrong_predict_list, k=5)

for n in samples:
    print("예측확률분포: " + str(predicted_result[n]))
    print("라벨: " + str(y_test[n]) + ", 예측결과: " + str(predicted_labels[n]))
    plt.imshow(x_test[n])
    plt.show()


        
[O]  **3. 에러가 난 부분을 디버깅하여 문제를 “해결한 기록"을 남겼거나 "새로운 시도 
또는 추가 실험"을 수행해봤나요?**
- 문제 원인 및 해결 과정을 잘 기록하였는지 확인 또는
- 문제에서 요구하는 조건에 더해 추가적으로 수행한 나만의 시도, 실험이 기록되어 있는지 확인
- 잘 작성되었다고 생각되는 부분을 캡쳐해 근거로 첨부합니다.

> Early stop을 적용하여 overfitting을 극복할 수 있는지 검토 하셨습니다.
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

        
[X]  **4. 회고를 잘 작성했나요?**
- 주어진 문제를 해결하는 완성된 코드 내지 프로젝트 결과물에 대해 배운점과 아쉬운점, 느낀점 등이 상세히 기록되어 있는지 확인
- 딥러닝 모델의 경우, 인풋이 들어가 최종적으로 아웃풋이 나오기까지의 전체 흐름을 도식화하여 모델 아키텍쳐에 대한 이해를 돕고 있는지 확인

> 시간이 부족하셔서 천천히 주시기 바랍니다.


[O]  **5. 코드가 간결하고 효율적인가요?**
- 파이썬 스타일 가이드 (PEP8) 를 준수하였는지 확인
- 코드 중복을 최소화하고 범용적으로 사용할 수 있도록 모듈화(함수화) 했는지
- 잘 작성되었다고 생각되는 부분을 캡쳐해 근거로 첨부합니다.


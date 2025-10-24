# ExecuTorch Model Verification & Benchmark CLI

### Structure
```
├── executorch_report.pdf  # Task 1: 오픈 소스 분석 레포트
├── requirements.txt       # 의존성 패키지
├── run_bench.py           # Task 3: 실행 로깅 CLI
├── unit_test.py           # Task 3: 모델 유효성, 추론, 벤치마킹 단위 테스트
├── utils.py               # Task 2 & 3: 유틸 메서드(모델 읽어오기, pte 파일 생성, 로깅)
└── verification.py        # Task 2: 모델 검증 테스트
```
## ⚙️ Environment
- Python 3.12.9  
- PyCharm 2024.1.7  

### Install Dependencies
```bash
pip install -r requirements.txt
```
---

## 🧩 Task 1
🎯 Objective
ExecuTorch 오픈소스 구조를 분석하고, 핵심 구성요소를 문서화한다.
### 📄 Description
크게 개요 및 작동 방식, 장점과 핵심 기능, 아키텍쳐, 소스 코드 구조, 지원 범위 및 한계점으로 나누어 구성했습니다. <br>
PDF 형식으로 작성되었으며, 해당 내용에 맞는 참고 문서가 하이퍼링크로 연결되어 있습니다.


---

## 🧠 Task 2
🎯 Objective
PyTorch 모델과 ExecuTorch 모델의 정확도 및 성능을 비교 검증한다.
### 📄 Description
- 모델의 경우 MobileNetV2, 반복 횟수는 100회로 지정하여 진행했습니다.
- PyTorch 모델과 ExecuTorch 모델의 정확도 지표로 MAD(Mean Absolute Difference)를 사용하여 비교했습니다.
- PyTorch와 ExecuTorch 추론 속도(latency)를 측정하고, 평균값과 최댓값을 도출했습니다. 측정 시에는 1회 웜업을 진행하여 안정된 테스트가 가능하도록 처리했습니다.
- 위 과정을 자동화하는 pytest 테스트 코드를 작성했습니다.

### 🧪 Model Validation Example
```
pytest -s verification.py
```
<img width="1378" height="399" alt="스크린샷 2025-10-25 오전 5 13 19" src="https://github.com/user-attachments/assets/0c614386-ad1e-453b-a6e3-fcf10b90b15b" />

---

## ⚡ Task 3
🎯 Objective
ExecuTorch 모델 실행 과정을 로깅하고, 사용자가 지정한 횟수만큼 추론하여 벤치마크 결과를 구조적으로 정리하는 CLI 구현.
### 📄 Description
- 모델명과 반복 횟수를 입력받아 ExecuTorch 추론 벤치마크를 수행합니다. 모델은 MobileNetV2와 ResNet18만을 지원합니다.
- 모델명을 딕셔너리 형태로 저장하여 추후 신규 모델 추가가 용이하도록 처리했습니다.
- `utils.py` 에 모델 불러오기, pte 파일 생성 그리고 로깅 기능을 공통 모듈로 분리했습니다.
- Validation과 마찬가지로, 1회 웜업 진행 후 벤치마킹을 진행합니다.
- 모델명, 평균 레이턴시, 반복 횟수를 JSON 형식으로 콘솔에 출력합니다.
- pytest 단위 테스트로 모델 지원 여부, 추론, 벤치마크 기능을 검증합니다.

### 🧪 Model Benchmark Example
```
python run_bench.py --model resnet18 --repeat 5
```
<img width="1241" height="398" alt="스크린샷 2025-10-25 오전 6 06 13" src="https://github.com/user-attachments/assets/b91118cf-8882-42ef-8230-fe086291c69c" />

### 🧪 Unit test Example
```
pytest -v unit_test.py
```
<img width="2462" height="705" alt="image" src="https://github.com/user-attachments/assets/dcd0c5ef-59f6-4b9b-b8ce-6d92545b56a6" />


---

## 📚 Reference
- [ExecuTorch 모델 로드 및 실행](https://github.com/meta-pytorch/executorch-examples/tree/main/mv2/python)
- [Warm-up을 사용하는 이유](https://github.com/pytorch/vision/blob/main/torchvision/models/mobilenetv2.py)
- [CLI Docs](https://docs.python.org/ko/3.13/library/argparse.html)
- [torch.export API Reference](https://docs.pytorch.org/docs/stable/export/api_reference.html#torch.export.export)
- [Logging](https://docs.python.org/3/library/logging.html)


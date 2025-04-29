# MEMORIA_AI

## 프로젝트 소개
- 치매 진단을 목표로 다양한 딥러닝 모델을 적용
- Django, Web 개발, Figma를 활용하여 AI 팀과 UX/UI 팀이 협업한 종합 프로젝트
- 라틴어로 '기억(Memoria)'을 의미, 기억을 지키기 위한 AI 개발

## 팀원
| 이정수 | 최영환 | 김건 | 정민지 | 박지은 | 이예림 |
|:---|:---|:---|:---|:---|:---|
| [GitHub](https://github.com/sw930718) | [GitHub](https://github.com/cyh5757) | [GitHub](https://github.com/Polar-Bear-Poby) | [GitHub](https://github.com/dustywindow) | [GitHub](https://github.com/JiEuNparrk) | [GitHub](https://github.com/yeliiim) |

## 성과
- 창업 경직 대회 대상
- 프로젝트 발표회 최우수상

## 파일 구조
```
MEMORIA_AI/
├── img/
│   └── award.jpg
├── InceptionResNetV2/
│   ├── tensor_py/  (TensorFlow 버전)
│   └── torch_py/   (PyTorch 버전)
├── PatchSVDD/
│   ├── codes/      (Patch-based Anomaly Detection)
├── Transformer/
│   └── (VisionTransformer, SwinTransformer)
├── EDA.ipynb       (데이터 탐색)
├── preprocessing.py (전처리 스크립트)
├── sharpening_png.py (이미지 색인 가장)
├── README.md
```

---

## 🧠 Alzheimer's MRI 진단 모델 with Inception-ResNet + Tabular Fusion

본 프로젝트는 뇌 MRI 이미지와 인구통계 정보(성별, 연령대, 이미지 번호)를 결합하여  
알츠하이머 진단 여부를 예측하는 **멀티모달 딥러닝 모델**을 구현한 것입니다.  
백본으로는 `Inception-ResNetV2`를 사용하고, Tabular 정보를 병렬로 처리한 후 통합하여 최종 예측을 수행합니다.

---

### 📁 핵심 코드 구성

| 파일명         | 설명                                               |
|----------------|----------------------------------------------------|
| `train.py`     | 전체 학습 파이프라인 구성                          |
| `model.py`     | 이미지+탭 구조 멀티모달 모델 구현                 |
| `utils.py`     | 데이터 전처리, 시드 고정, Dataset 생성             |
| `inference.py` | GradCAM 기반 추론 및 시각화 로직                  |

---

### 🧪 모델 구조 및 학습 흐름

- `Inception-ResNetV2` 기반 이미지 인코더 사용
- 성별, 나이그룹, 이미지 번호를 MLP로 인코딩 후 이미지 특징과 병합
- 최종 분류기(Dense Layer)로 이진 분류 수행
- `train.py`에서 전체 데이터 로딩 → Dataset 구성 → 모델 학습 수행
- `inference.py`에서는 단일 환자 이미지 추론 및 GradCAM 시각화 수행

```python
# 예시 실행
inference(
    folder_path="/path/to/Mild AD/002_S_0729_110816",
    gender=0,
    age_group=0,
    model_path="/path/to/best_model.h5"
)
```

---

### 🧑‍💻 Award

![Award](img/award.jpg)

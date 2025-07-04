# Project Mini-T: End-to-End Local MLOps Pipeline

이 프로젝트는 Transformer 언어 모델(Mini-T)과 학습 가능한 PID 제어기(PIDNet)를 통합하여, 100% 로컬 환경을 기반으로 개발하고 클라우드 자원을 활용해 실행하는 완전한 하이브리드 MLOps 파이프라인을 구축하는 것을 목표로 합니다.

## 핵심 특징

- **하이브리드 워크플로우:** 로컬 환경에서는 코드 개발, 테스트, 추론을, Google Colab에서는 GPU를 활용한 무거운 학습 및 하이퍼파라미터 최적화(HPO)를 수행합니다.
- **재현성 보장:** Docker와 Conda를 통해 로컬 개발 환경을, `requirements.txt`를 통해 실행 환경의 라이브러리 버전을 통제하여 재현성을 확보합니다.
- **모듈식 설계:** 각 기능(모델, 데이터, 학습)이 독립된 모듈로 구성되어 유지보수와 확장이 용이합니다.
- **자동화 파이프라인:** `scripts/run_pipeline.py` CLI를 통해 파이프라인의 각 단계를 명렁어로 실행하고, 최종 리포트를 자동으로 생성합니다.

## 프로젝트 구조

## 시작하기

### 전제 조건
* Git
* Docker Desktop (Windows 사용자의 경우 WSL2 백엔드 사용)
* NVIDIA GPU (로컬에서 GPU 작업을 실행할 경우)
* GitHub 계정

### A. 로컬 Docker 환경 설정 (개발 및 추론용)

1.  **저장소 복제:**
    ```bash
    git clone <your-repository-url>
    cd mini-t-project
    ```
2.  **환경 파일 확인:** `Dockerfile`, `docker-compose.yml`, `environment.yml`, `requirements.txt` 파일이 올바르게 구성되어 있는지 확인합니다. (완성한 최종 버전을 사용합니다.)
3.  **컨테이너 빌드 및 실행:**
    ```bash
    # (선택) 가장 깨끗하게 빌드하려면
    docker-compose down --rmi all -v
    docker system prune -af

    # 컨테이너 빌드 및 백그라운드 실행
    docker-compose up --build -d
    ```
4.  **컨테이너 접속:**
    ```bash
    docker-compose exec app bash
    ```

### B. Google Colab 환경 설정 (학습 및 HPO용)

1.  **프로젝트 업로드:** 로컬 프로젝트를 GitHub 비공개 저장소에 Push 합니다.
2.  **Colab 설정:**
    * 새 Colab 노트북을 열고, 메뉴에서 `런타임 > 런타임 유형 변경`을 통해 하드웨어 가속기를 **GPU**로 설정합니다.
3.  **환경 구성 및 실행:**
    * Colab 코드 셀에서 아래 명령어들을 순차적으로 실행합니다.

    ```python
    # 1. GitHub 저장소 복제
    !git clone <your-github-repository-url>
    %cd mini-t-project

    # 2. 라이브러리 설치
    !pip install -r requirements.txt

    # 3. HPO 또는 학습 스크립트 실행
    !python src/hpo/optimize.py
    ```

## 파이프라인 사용법

로컬 컨테이너 또는 Colab 환경에서 `scripts/run_pipeline.py`를 사용하여 주요 작업을 자동화할 수 있습니다.

* **모든 명령어 확인:**
    ```bash
    python scripts/run_pipeline.py --help
    ```
* **최종 리포트 생성:**
    (먼저 `hpo_results.json` 파일이 프로젝트 루트에 있어야 합니다.)
    ```bash
    python scripts/run_pipeline.py generate-report
    ```
* **최종 모델 추론 실행:**
    (먼저 `final_pid_net.pth` 파일이 프로젝트 루트에 있어야 합니다.)
    ```bash
    python scripts/use_final_model.py
    ```

## 프로젝트 개발 여정과 교훈

초기 목표는 100% 로컬 실행이었으나, Windows/WSL2/Docker 환경에서 Ray와 PyTorch 간의 깊은 수준의 호환성 문제로 인해 HPO 실행에 어려움을 겪었습니다.

이 문제를 해결하기 위해, 최종적으로 **하이브리드 MLOps 워크플로우**를 채택했습니다:
* **로컬 환경:** 코드 개발, 단위 테스트, 최종 모델 추론.
* **Colab 환경:** GPU를 활용한 무거운 작업(QLoRA, HPO) 실행.

이 과정을 통해 코드 구현뿐만 아니라, 실제 MLOps 환경에서 마주할 수 있는 복잡한 의존성 관리, 시스템 수준의 디버깅, 그리고 유연한 아키텍처 전환의 중요성을 배웠습니다. 이 프로젝트의 가장 큰 성과는 이 모든 문제 해결의 경험 그 자체입니다.
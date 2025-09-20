📂 폴더 구조 (Folder Structure)
/dynamic-mmm-simulator/
|
|-- app.py                   # Streamlit 메인 애플리케이션
|-- advanced_simulator.py    # 핵심 시뮬레이션 엔진
|-- analysis.py              # 민감도 분석 등 전략 분석 모듈
|-- visualization.py         # Plotly 차트 생성 모듈
|-- scenarios.yaml           # 시나리오 및 모델 가정 설정 파일
|-- pyproject.toml           # Poetry 의존성 및 프로젝트 설정 파일
|-- poetry.lock              # 정확한 의존성 버전 잠금 파일
|-- README.md                # 사용 가이드 (아래 내용)
|
|-- /data/ (선택 사항)
|   |-- backtest_example.csv # 백테스팅용 샘플 데이터 파일

🛠️ 개발 환경 설정 (Environment Setup)
이 프로젝트는 Poetry를 사용하여 의존성을 관리하고 실행 환경을 구성합니다.

1. Poetry 설치 (컴퓨터에 최초 1회만 필요)
Poetry가 설치되어 있지 않다면, 터미널에서 아래 공식 명령어를 실행하여 설치합니다.

# macOS / Linux / WSL
curl -sSL [https://install.python-poetry.org](https://install.python-poetry.org) | python3 -

# Windows (PowerShell)
(Invoke-WebRequest -Uri [https://install.python-poetry.org](https://install.python-poetry.org) -UseBasicParsing).Content | py -

설치 후 터미널을 재시작해야 할 수 있습니다.

2. 프로젝트 의존성 설치
프로젝트 폴더 내에서 터미널을 열고 아래 명령어를 실행합니다. poetry install 명령어는 자동으로 가상 환경을 생성하고, pyproject.toml 파일에 명시된 모든 라이브러리를 설치합니다.

# 필요한 모든 라이브러리를 pyproject.toml로부터 설치
poetry install

최초 프로젝트 설정 시
만약 pyproject.toml 파일이 없다면, 아래 명령어로 프로젝트를 초기화하고 라이브러리를 추가할 수 있습니다.

# 1. Poetry 프로젝트로 초기화
poetry init --no-interaction
# 2. 라이브러리 추가
poetry add streamlit pandas numpy plotly scipy pyyaml

3. 시뮬레이터 실행
모든 준비가 완료되었습니다. 아래 명령어를 실행하여 Poetry 가상 환경에서 Streamlit 애플리케이션을 시작합니다.

poetry run streamlit run app.py

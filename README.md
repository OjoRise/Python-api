Python API

📦 실행 방법

1. 프로젝트 디렉토리 이동

cd c://../python-api

2. 가상환경 활성화 (Windows 기준)

venv\Scripts\activate

3. 서버 실행

uvicorn main:app --reload --host 0.0.0.0 --port 8000

🔧 사전 준비 사항

✅ Python 설치

Python 3.10 이상 필요

✅ 패키지 설치

pip install -r requirements.txt

✅ .env 파일 생성

루트 디렉토리에 .env 파일 생성 후 다음과 같이 작성:

OPENAI_API_KEY=your_openai_key
QDRANT_API_KEY=your_qdrant_key
QDRANT_URL=http://localhost:6333

.env는 보안상 .gitignore에 포함되어 있어야 합니다.

📌 기타

포트: 기본 8000

개발 환경에서는 --reload 옵션으로 변경사항 자동 반영

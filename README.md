## 환경 구축하기

- **파이썬이 깔려있지 않다면 먼저 해야할 것**
    1.  https://nodejs.org/en 에서 python 설치
    2. pip 설치를 위해 cmd 창에 다음 명령어 입력
        `curl https://bootstrap.pypa.io/get-pip.py -o [get-pip.py](http://get-pip.py/)` 
        `python3 [get-pip.py](http://get-pip.py/)`
        
- **파이썬이 깔려있다면**
    1. cmd 창에서 프로젝트 디렉토리 위치로 이동
    2. 가상환경 생성 `python3 -m venv env`
    3. 가상환경 실행 `source env/bin/activate`
    4.  필요한 모듈 설치
        `pip3 install fastapi "uvicorn[standard]" torch transformers sentencepiece`
        `pip3 install 'git+https://github.com/SKTBrain/KoBERT.git#egg=kobert_tokenizer&subdirectory=kobert_hf'`
    5. AI 모델 폴더(style_classification, style_transform)를 프로젝트 디렉토리 내부에 넣어준 후, 
    6. 서버 실행 `uvicorn main:app --reload`

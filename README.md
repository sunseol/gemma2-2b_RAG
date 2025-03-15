# Gemma 2-2B-IT 기반 RAG 시스템

## 한국어 버전

### 소개
이 프로젝트는 Google의 Gemma 2-2B-IT 모델을 활용한 검색 증강 생성(RAG) 시스템입니다. 사용자의 질문에 대해 관련 문서를 검색하고, 검색된 문서를 기반으로 정확하고 상세한 답변을 생성합니다.

### 주요 기능
- 사용자 질의에 대한 의미론적 검색
- 검색된 문서를 기반으로 한 답변 생성
- 한국어 문법 및 언어 규칙에 대한 질의 응답
- GPU 가속을 통한 빠른 응답 생성

### 시스템 구조
- `src/ui/`: 웹 인터페이스 관련 코드
- `src/retrieval/`: 문서 검색 및 인덱싱 관련 코드
- `src/generation/`: 텍스트 생성 관련 코드
- `src/preprocessing/`: 데이터 전처리 관련 코드
- `data/`: 검색 대상 문서 저장소

### 설치 및 실행 방법
1. 필요한 패키지 설치:
   ```
   pip install -r requirements.txt
   ```

2. 웹 서버 실행:
   ```
   python src/ui/app.py
   ```

3. 브라우저에서 접속:
   ```
   http://localhost:5000
   ```

### 요구사항
- Python 3.10 이상
- CUDA 지원 GPU (권장)
- PyTorch 2.3.0 이상 (CUDA 지원 버전)

---

# Gemma 2-2B-IT based RAG System

## English Version

### Introduction
This project is a Retrieval-Augmented Generation (RAG) system utilizing Google's Gemma 2-2B-IT model. It searches for relevant documents based on user queries and generates accurate and detailed answers based on the retrieved documents.

### Key Features
- Semantic search for user queries
- Answer generation based on retrieved documents
- Q&A for Korean grammar and language rules
- Fast response generation through GPU acceleration

### System Structure
- `src/ui/`: Web interface related code
- `src/retrieval/`: Document retrieval and indexing related code
- `src/generation/`: Text generation related code
- `src/preprocessing/`: Data preprocessing related code
- `data/`: Repository for documents to be searched

### Installation and Execution
1. Install required packages:
   ```
   pip install -r requirements.txt
   ```

2. Run the web server:
   ```
   python src/ui/app.py
   ```

3. Access in browser:
   ```
   http://localhost:5000
   ```

### Requirements
- Python 3.10 or higher
- CUDA-supported GPU (recommended)
- PyTorch 2.3.0 or higher (with CUDA support) 
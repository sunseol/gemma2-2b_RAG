# Gemma 2-2B-IT 기반 RAG 시스템

## 한국어 버전

### 소개
이 프로젝트는 Google의 Gemma 2-2B-IT 모델을 활용한 검색 증강 생성(RAG) 시스템입니다. 사용자의 질문에 대해 관련 문서를 검색하고, 검색된 문서를 기반으로 정확하고 상세한 답변을 생성합니다.

### 주요 기능
- 사용자 질의에 대한 의미론적 검색
- 검색된 문서를 기반으로 한 답변 생성
- 한국어 문법 및 언어 규칙에 대한 질의 응답
- GPU 가속을 통한 빠른 응답 생성
- 사용자 파일 첨부 기능 (TXT, PDF 등 문서 파일 지원)

### 시스템 구조
- `src/ui/`: 웹 인터페이스 관련 코드
  - Flask 기반 웹 서버
  - 사용자 질의 입력 및 응답 표시 인터페이스
  - 파일 업로드 및 관리 기능
- `src/retrieval/`: 문서 검색 및 인덱싱 관련 코드
  - 의미론적 검색 엔진
  - 벡터 데이터베이스 연동
  - 유사도 기반 문서 검색
- `src/generation/`: 텍스트 생성 관련 코드
  - Gemma 2-2B-IT 모델 로딩 및 추론
  - 프롬프트 엔지니어링
  - 응답 생성 및 후처리
- `src/preprocessing/`: 데이터 전처리 관련 코드
  - 문서 파싱 및 청소
  - 청크 분할
  - 임베딩 생성
- `data/`: 검색 대상 문서 저장소

### 사용된 모델 및 기술
- **생성 모델**: Google의 Gemma 2-2B-IT
  - 2B 파라미터 규모의 경량 언어 모델
  - 한국어 및 영어 지원
  - 지시사항 따르기(Instruction Following) 기능 최적화
- **임베딩 모델**: Sentence-Transformers (all-MiniLM-L6-v2)
  - 문장 및 문서의 의미론적 표현을 위한 임베딩 생성
  - 384차원 벡터 출력
  - 다국어 지원
- **벡터 데이터베이스**: FAISS (Facebook AI Similarity Search)
  - 고성능 벡터 유사도 검색
  - 인메모리 인덱싱으로 빠른 검색 속도
  - 코사인 유사도 기반 검색

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

### 파일 첨부 기능 사용법
1. 웹 인터페이스의 "파일 업로드" 버튼을 클릭합니다.
2. 지원되는 파일 형식(TXT, PDF 등)의 문서를 선택합니다.
3. 업로드 후, 시스템이 자동으로 문서를 처리하고 검색 가능한 형태로 변환합니다.
4. 이후 질의 시 업로드한 문서의 내용도 검색 대상에 포함됩니다.

### 요구사항
- Python 3.10 이상
- CUDA 지원 GPU (권장)
- PyTorch 2.3.0 이상 (CUDA 지원 버전)
- 최소 8GB RAM (16GB 이상 권장)
- 최소 10GB 디스크 공간

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
- File attachment functionality (supporting TXT, PDF, and other document formats)

### System Structure
- `src/ui/`: Web interface related code
  - Flask-based web server
  - User query input and response display interface
  - File upload and management functionality
- `src/retrieval/`: Document retrieval and indexing related code
  - Semantic search engine
  - Vector database integration
  - Similarity-based document retrieval
- `src/generation/`: Text generation related code
  - Gemma 2-2B-IT model loading and inference
  - Prompt engineering
  - Response generation and post-processing
- `src/preprocessing/`: Data preprocessing related code
  - Document parsing and cleaning
  - Chunk splitting
  - Embedding generation
- `data/`: Repository for documents to be searched

### Models and Technologies Used
- **Generation Model**: Google's Gemma 2-2B-IT
  - Lightweight language model with 2B parameters
  - Support for Korean and English
  - Optimized for instruction following
- **Embedding Model**: Sentence-Transformers (all-MiniLM-L6-v2)
  - Creates semantic representations for sentences and documents
  - Outputs 384-dimensional vectors
  - Multilingual support
- **Vector Database**: FAISS (Facebook AI Similarity Search)
  - High-performance vector similarity search
  - In-memory indexing for fast search speeds
  - Cosine similarity-based search

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

### How to Use File Attachment Feature
1. Click the "Upload File" button in the web interface.
2. Select a document in a supported format (TXT, PDF, etc.).
3. After uploading, the system automatically processes the document and converts it into a searchable format.
4. Subsequent queries will include the content of the uploaded document in the search results.

### Requirements
- Python 3.10 or higher
- CUDA-supported GPU (recommended)
- PyTorch 2.3.0 or higher (with CUDA support)
- Minimum 8GB RAM (16GB or more recommended)
- Minimum 10GB disk space 
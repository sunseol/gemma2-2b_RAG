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

### 시스템 프로세스 흐름
1. **초기화 단계**
   - 웹 서버(Flask) 시작
   - 임베딩 모델(Sentence-Transformers) 로드
   - 벡터 데이터베이스(FAISS) 인덱스 로드
   - Gemma 2-2B-IT 모델 로드 (GPU 가속 활성화)

2. **문서 처리 단계**
   - 사용자가 문서 파일(TXT, PDF) 업로드
   - 문서 파싱 및 텍스트 추출
   - 텍스트를 적절한 크기의 청크로 분할
   - 각 청크에 대한 임베딩 벡터 생성
   - 생성된 임베딩을 FAISS 인덱스에 추가

3. **질의 처리 단계**
   - 사용자가 웹 UI를 통해 질문 입력
   - 질문 텍스트에 대한 임베딩 벡터 생성
   - 임베딩 벡터를 사용하여 FAISS에서 유사한 문서 청크 검색
   - 유사도 점수에 따라 상위 문서 선택 (기본값: 상위 3개)

4. **답변 생성 단계**
   - 검색된 문서 청크에서 핵심 정보 추출
   - 질문과 검색된 정보를 결합하여 프롬프트 생성
   - Gemma 2-2B-IT 모델에 프롬프트 전달
   - 모델이 답변 생성 (온도 설정: 0.1, 최대 길이: 1024 토큰)
   - 생성된 답변 후처리 및 정제

5. **응답 전달 단계**
   - 생성된 답변을 웹 UI에 표시
   - 참조된 문서 출처 및 유사도 점수 함께 제공
   - 사용자가 필요시 추가 질문 가능

이 전체 프로세스는 사용자의 질문에서 답변 생성까지 평균 2-5초 내에 완료됩니다(GPU 사용 시). 시스템은 사용자가 업로드한 문서뿐만 아니라 기본 제공되는 지식 베이스를 활용하여 다양한 질문에 답변할 수 있습니다.

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

### System Process Flow
1. **Initialization Step**
   - Start the web server (Flask)
   - Load the embedding model (Sentence-Transformers)
   - Load the vector database (FAISS)
   - Load the Gemma 2-2B-IT model (GPU acceleration enabled)

2. **Document Processing Step**
   - User uploads a document file (TXT, PDF)
   - Document parsing and text extraction
   - Text is split into appropriate-sized chunks
   - Embedding vectors are generated for each chunk
   - Generated embeddings are added to the FAISS index

3. **Query Processing Step**
   - User inputs a query through the web UI
   - Embedding vector is generated for the query text
   - Embedding vector is used to search for similar document chunks in FAISS
   - Top documents are selected based on similarity scores (default: top 3)

4. **Answer Generation Step**
   - Extract key information from retrieved document chunks
   - Combine query and retrieved information to generate a prompt
   - Prompt is passed to the Gemma 2-2B-IT model
   - Model generates an answer (temperature: 0.1, maximum length: 1024 tokens)
   - Post-processing and refinement of generated answer

5. **Response Delivery Step**
   - Generated answer is displayed in the web UI
   - Reference document source and similarity score are provided
   - User can ask additional questions if needed

This entire process completes within 2-5 seconds on average (with GPU usage). The system can answer various questions using not only the documents uploaded by the user but also the knowledge base provided by default.

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
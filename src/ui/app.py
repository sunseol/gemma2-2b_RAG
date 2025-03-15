import os
import sys
import time
import traceback
import shutil
import getpass
from flask import Flask, render_template, request, jsonify, redirect, url_for
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# 현재 스크립트의 절대 경로
current_dir = os.path.dirname(os.path.abspath(__file__))
# 프로젝트 루트 디렉토리
root_dir = os.path.dirname(os.path.dirname(current_dir))
# 모델 디렉토리
models_dir = os.path.join(root_dir, "models")
# 데이터 디렉토리
data_dir = os.path.join(root_dir, "data")

# Hugging Face 토큰 설정
hf_token = os.environ.get("HUGGING_FACE_HUB_TOKEN")
if not hf_token:
    print("Hugging Face 토큰이 환경 변수에 설정되어 있지 않습니다.")
    print("Gemma 모델을 사용하려면 Hugging Face 토큰이 필요합니다.")
    print("https://huggingface.co/settings/tokens 에서 토큰을 생성하세요.")
    print("Gemma 3-4B 모델에 접근하려면 Hugging Face에 로그인하고 모델 접근 권한을 요청해야 합니다.")
    print("https://huggingface.co/google/gemma-3-4b-it 에서 모델 접근 권한을 요청하세요.")
    hf_token = getpass.getpass("Hugging Face 토큰을 입력하세요: ")
    os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token
else:
    print(f"Hugging Face 토큰: {hf_token[:4]}...{hf_token[-4:] if hf_token else ''}")

# 상위 디렉토리 추가하여 모듈 import 가능하게 함
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from generation.generate import AnswerGenerator
from preprocessing.preprocess import preprocess_documents
from retrieval.build_index import build_search_index

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 최대 16MB 파일 허용
app.config['UPLOAD_FOLDER'] = data_dir

# 허용된 파일 확장자
ALLOWED_EXTENSIONS = {'txt', 'pdf'}

# 전역 변수로 RAG 시스템 인스턴스 저장
rag_generator = None

def allowed_file(filename):
    """파일 확장자 검사"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """메인 페이지 렌더링"""
    return render_template('index.html')

@app.route('/api/query', methods=['POST'])
def process_query():
    """API 엔드포인트: 질문 처리 및 답변 생성"""
    global rag_generator
    
    # 요청 데이터 가져오기
    data = request.json
    query = data.get('query', '')
    top_k = int(data.get('top_k', 3))
    
    if not query:
        return jsonify({'error': '질문이 비어 있습니다.'}), 400
    
    try:
        start_time = time.time()
        
        # 현재 작업 디렉토리 출력
        print(f"현재 작업 디렉토리: {os.getcwd()}")
        print(f"모델 디렉토리: {models_dir}")
        
        # RAG 시스템이 초기화되지 않은 경우 초기화
        if rag_generator is None:
            rag_generator = AnswerGenerator(model_name="google/gemma-2-2b", models_dir=models_dir)
            print("Gemma 2-2B 모델로 초기화되었습니다.")
        
        # 답변 생성
        answer, docs = rag_generator.process_query(query, top_k)
        
        elapsed_time = time.time() - start_time
        
        # 결과 포맷팅
        sources = []
        for doc in docs:
            sources.append({
                'content': doc['document'][:300] + '...' if len(doc['document']) > 300 else doc['document'],
                'source': doc['source'],
                'score': float(doc['score'])  # float로 변환하여 JSON 직렬화 가능하게 함
            })
        
        return jsonify({
            'answer': answer,
            'sources': sources,
            'elapsed_time': elapsed_time
        })
    
    except Exception as e:
        print(f"오류 발생: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': f'처리 중 오류가 발생했습니다: {str(e)}'}), 500

@app.route('/upload', methods=['POST'])
def upload_file():
    """파일 업로드 처리"""
    if 'file' not in request.files:
        return jsonify({'error': '파일이 제공되지 않았습니다.'}), 400
        
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': '선택된 파일이 없습니다.'}), 400
        
    if file and allowed_file(file.filename):
        try:
            # 파일명 보안 처리
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            # 파일 저장
            file.save(file_path)
            
            # 파일 처리 및 인덱스 재구축
            process_result = process_uploaded_files()
            
            return jsonify({
                'success': True,
                'message': f'파일 "{filename}" 업로드 및 처리 완료',
                'details': process_result
            })
            
        except Exception as e:
            print(f"파일 업로드 오류: {str(e)}")
            print(traceback.format_exc())
            return jsonify({'error': f'파일 처리 중 오류가 발생했습니다: {str(e)}'}), 500
    else:
        return jsonify({'error': f'허용되지 않는 파일 형식입니다. 허용된 형식: {", ".join(ALLOWED_EXTENSIONS)}'}), 400

def process_uploaded_files():
    """업로드된 파일 처리 및 인덱스 재구축"""
    global rag_generator
    
    # 1. 문서 전처리
    doc_count = preprocess_documents(data_dir=data_dir, output_dir=models_dir)
    
    # 2. 인덱스 재구축
    index_info = build_search_index(models_dir=models_dir)
    
    # 3. RAG 시스템 재초기화 (필요한 경우)
    if rag_generator is not None:
        rag_generator = None  # 기존 인스턴스 해제
    
    return {
        'document_count': len(doc_count) if doc_count else 0,
        'index_info': index_info
    }

def create_templates():
    """템플릿 디렉토리 및 파일 생성"""
    templates_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')
    os.makedirs(templates_dir, exist_ok=True)
    
    # index.html 템플릿 생성
    index_path = os.path.join(templates_dir, 'index.html')
    with open(index_path, 'w', encoding='utf-8') as f:
        f.write('''<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RAG 시스템 - Gemma 2-2B</title>
    <style>
        body {
            font-family: 'Noto Sans KR', Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
        }
        h1 {
            color: #2c3e50;
            text-align: center;
            margin-bottom: 30px;
        }
        .container {
            background-color: #f9f9f9;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .query-form, .upload-form {
            margin-bottom: 20px;
            padding-bottom: 20px;
            border-bottom: 1px solid #eee;
        }
        .form-group {
            margin-bottom: 15px;
        }
        label {
            display: block;
            margin-bottom: 5px;
            font-weight: bold;
        }
        textarea, select, input[type="file"] {
            width: 100%;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 4px;
            font-size: 16px;
        }
        button {
            background-color: #3498db;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 16px;
        }
        button:hover {
            background-color: #2980b9;
        }
        button:disabled {
            background-color: #95a5a6;
            cursor: not-allowed;
        }
        .result {
            margin-top: 30px;
            display: none;
        }
        .answer {
            background-color: #fff;
            border-left: 4px solid #3498db;
            padding: 15px;
            margin-bottom: 20px;
            white-space: pre-wrap;
        }
        .sources {
            background-color: #fff;
            border: 1px solid #ddd;
            border-radius: 4px;
            padding: 15px;
        }
        .source-item {
            margin-bottom: 15px;
            padding-bottom: 15px;
            border-bottom: 1px solid #eee;
        }
        .source-item:last-child {
            border-bottom: none;
        }
        .meta {
            color: #7f8c8d;
            font-size: 14px;
            margin-top: 10px;
        }
        .loading {
            text-align: center;
            display: none;
        }
        .error {
            color: #e74c3c;
            background-color: #fadbd8;
            padding: 10px;
            border-radius: 4px;
            margin-top: 10px;
            display: none;
        }
        .success {
            color: #27ae60;
            background-color: #d5f5e3;
            padding: 10px;
            border-radius: 4px;
            margin-top: 10px;
            display: none;
        }
        .model-info {
            text-align: center;
            font-size: 14px;
            color: #7f8c8d;
            margin-bottom: 20px;
        }
        .tabs {
            display: flex;
            margin-bottom: 20px;
        }
        .tab {
            padding: 10px 20px;
            cursor: pointer;
            background-color: #f1f1f1;
            border-radius: 4px 4px 0 0;
            margin-right: 5px;
        }
        .tab.active {
            background-color: #3498db;
            color: white;
        }
        .tab-content {
            display: none;
        }
        .tab-content.active {
            display: block;
        }
    </style>
</head>
<body>
    <h1>RAG 시스템</h1>
    <div class="model-info">Powered by Gemma 2-2B</div>
    
    <div class="container">
        <div class="tabs">
            <div class="tab active" data-tab="query">질문하기</div>
            <div class="tab" data-tab="upload">문서 업로드</div>
        </div>
        
        <div class="tab-content active" id="query-tab">
            <div class="query-form">
                <div class="form-group">
                    <label for="query">질문:</label>
                    <textarea id="query" rows="3" placeholder="질문을 입력하세요..."></textarea>
                </div>
                
                <div class="form-group">
                    <label for="top-k">검색할 문서 수:</label>
                    <select id="top-k">
                        <option value="1">1</option>
                        <option value="3" selected>3</option>
                        <option value="5">5</option>
                    </select>
                </div>
                
                <button id="submit-btn">답변 생성</button>
            </div>
            
            <div class="loading-message" id="loadingMessage" style="display: none;">
                <div class="spinner"></div>
                <p>답변 생성 중... 최대 2분 정도 소요될 수 있습니다.</p>
            </div>
            
            <div class="error" id="query-error"></div>
            
            <div class="result">
                <h2>답변:</h2>
                <div class="answer"></div>
                
                <h3>참고 문서:</h3>
                <div class="sources"></div>
                
                <div class="meta">
                    처리 시간: <span class="elapsed-time"></span>초
                </div>
            </div>
        </div>
        
        <div class="tab-content" id="upload-tab">
            <div class="upload-form">
                <div class="form-group">
                    <label for="file-upload">문서 파일 업로드 (TXT, PDF):</label>
                    <input type="file" id="file-upload" accept=".txt,.pdf">
                </div>
                
                <button id="upload-btn">업로드 및 처리</button>
            </div>
            
            <div class="loading" id="upload-loading">
                <p>파일 처리 중... (파일 크기에 따라 수 분이 소요될 수 있습니다)</p>
            </div>
            
            <div class="error" id="upload-error"></div>
            <div class="success" id="upload-success"></div>
        </div>
    </div>
    
    <script>
        document.addEventListener('DOMContentLoaded', function() {
            // 탭 전환 기능
            const tabs = document.querySelectorAll('.tab');
            const tabContents = document.querySelectorAll('.tab-content');
            
            tabs.forEach(tab => {
                tab.addEventListener('click', function() {
                    const tabId = this.getAttribute('data-tab');
                    
                    // 탭 활성화
                    tabs.forEach(t => t.classList.remove('active'));
                    this.classList.add('active');
                    
                    // 탭 컨텐츠 활성화
                    tabContents.forEach(content => content.classList.remove('active'));
                    document.getElementById(tabId + '-tab').classList.add('active');
                    
                    // 오류 메시지 초기화
                    document.querySelectorAll('.error, .success').forEach(el => el.style.display = 'none');
                });
            });
            
            // 질문 제출 기능
            const submitBtn = document.getElementById('submit-btn');
            const queryInput = document.getElementById('query');
            const topKSelect = document.getElementById('top-k');
            const resultDiv = document.querySelector('.result');
            const answerDiv = document.querySelector('.answer');
            const sourcesDiv = document.querySelector('.sources');
            const elapsedTimeSpan = document.querySelector('.elapsed-time');
            const queryLoadingDiv = document.getElementById('loadingMessage');
            const queryErrorDiv = document.getElementById('query-error');
            
            submitBtn.addEventListener('click', async function() {
                const query = queryInput.value.trim();
                const topK = parseInt(topKSelect.value);
                
                if (!query) {
                    queryErrorDiv.textContent = '질문을 입력해주세요.';
                    queryErrorDiv.style.display = 'block';
                    return;
                }
                
                // UI 초기화
                resultDiv.style.display = 'none';
                queryErrorDiv.style.display = 'none';
                queryLoadingDiv.style.display = 'block';
                submitBtn.disabled = true;
                
                try {
                    const response = await fetch('/api/query', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify({
                            query: query,
                            top_k: topK
                        })
                    });
                    
                    const data = await response.json();
                    
                    if (response.ok) {
                        // 답변 표시
                        answerDiv.textContent = data.answer;
                        
                        // 참고 문서 표시
                        sourcesDiv.innerHTML = '';
                        if (data.sources && data.sources.length > 0) {
                            data.sources.forEach(source => {
                                const sourceItem = document.createElement('div');
                                sourceItem.className = 'source-item';
                                
                                const sourceContent = document.createElement('div');
                                sourceContent.textContent = source.content;
                                
                                const sourceInfo = document.createElement('div');
                                sourceInfo.className = 'meta';
                                sourceInfo.textContent = `출처: ${source.source} (점수: ${source.score.toFixed(4)})`;
                                
                                sourceItem.appendChild(sourceContent);
                                sourceItem.appendChild(sourceInfo);
                                sourcesDiv.appendChild(sourceItem);
                            });
                        } else {
                            sourcesDiv.textContent = '관련 문서가 없습니다.';
                        }
                        
                        // 처리 시간 표시
                        elapsedTimeSpan.textContent = data.elapsed_time.toFixed(2);
                        
                        // 결과 표시
                        resultDiv.style.display = 'block';
                    } else {
                        // 오류 표시
                        queryErrorDiv.textContent = data.error || '서버 오류가 발생했습니다.';
                        queryErrorDiv.style.display = 'block';
                    }
                } catch (error) {
                    console.error('Error:', error);
                    queryErrorDiv.textContent = '요청 처리 중 오류가 발생했습니다.';
                    queryErrorDiv.style.display = 'block';
                } finally {
                    queryLoadingDiv.style.display = 'none';
                    submitBtn.disabled = false;
                }
            });
            
            // 파일 업로드 기능
            const uploadBtn = document.getElementById('upload-btn');
            const fileInput = document.getElementById('file-upload');
            const uploadLoadingDiv = document.getElementById('upload-loading');
            const uploadErrorDiv = document.getElementById('upload-error');
            const uploadSuccessDiv = document.getElementById('upload-success');
            
            uploadBtn.addEventListener('click', async function() {
                if (!fileInput.files || fileInput.files.length === 0) {
                    uploadErrorDiv.textContent = '파일을 선택해주세요.';
                    uploadErrorDiv.style.display = 'block';
                    return;
                }
                
                const file = fileInput.files[0];
                const formData = new FormData();
                formData.append('file', file);
                
                // UI 초기화
                uploadErrorDiv.style.display = 'none';
                uploadSuccessDiv.style.display = 'none';
                uploadLoadingDiv.style.display = 'block';
                uploadBtn.disabled = true;
                
                try {
                    const response = await fetch('/upload', {
                        method: 'POST',
                        body: formData
                    });
                    
                    const data = await response.json();
                    
                    if (response.ok && data.success) {
                        // 성공 메시지 표시
                        uploadSuccessDiv.textContent = data.message;
                        uploadSuccessDiv.style.display = 'block';
                        
                        // 파일 입력 초기화
                        fileInput.value = '';
                    } else {
                        // 오류 표시
                        uploadErrorDiv.textContent = data.error || '파일 업로드 중 오류가 발생했습니다.';
                        uploadErrorDiv.style.display = 'block';
                    }
                } catch (error) {
                    console.error('Error:', error);
                    uploadErrorDiv.textContent = '파일 업로드 중 오류가 발생했습니다.';
                    uploadErrorDiv.style.display = 'block';
                } finally {
                    uploadLoadingDiv.style.display = 'none';
                    uploadBtn.disabled = false;
                }
            });
        });
    </script>
</body>
</html>''')
    
    print(f"템플릿 파일 생성 완료: {index_path}")

if __name__ == "__main__":
    # 필요한 디렉토리 생성
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    
    create_templates()
    print("Flask 웹 서버 시작 (http://localhost:5000)")
    app.run(host="0.0.0.0", debug=True) 
import os
import re
import pickle
from pathlib import Path
import PyPDF2

def preprocess_documents(data_dir="data", output_dir="models"):
    """
    데이터 디렉토리에서 문서를 전처리하고 청크로 분할하여 저장합니다.
    
    Args:
        data_dir (str): 원본 문서가 있는 디렉토리 경로
        output_dir (str): 전처리된 문서를 저장할 디렉토리 경로
        
    Returns:
        list: 생성된 문서 목록
    """
    print("문서 전처리 시작...")
    
    # 절대 경로로 변환
    if not os.path.isabs(data_dir):
        data_dir = os.path.abspath(data_dir)
    if not os.path.isabs(output_dir):
        output_dir = os.path.abspath(output_dir)
    
    print(f"데이터 디렉토리: {data_dir}")
    print(f"출력 디렉토리: {output_dir}")
    
    # 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # 문서 목록 가져오기
    documents = []
    
    # 데이터 디렉토리의 모든 파일 처리
    for file_path in Path(data_dir).glob("*.*"):
        print(f"파일 발견: {file_path}")
        if file_path.suffix.lower() == '.txt':
            # 텍스트 파일 처리
            try:
                # 다양한 인코딩 시도
                encodings = ['utf-8', 'cp949', 'euc-kr', 'latin1']
                text = None
                
                for encoding in encodings:
                    try:
                        with open(file_path, 'r', encoding=encoding) as f:
                            text = f.read()
                            print(f"파일을 {encoding} 인코딩으로 성공적으로 읽었습니다.")
                            break
                    except UnicodeDecodeError:
                        print(f"{encoding} 인코딩으로 읽기 실패, 다음 인코딩 시도...")
                        continue
                
                if text is None:
                    print(f"모든 인코딩 시도 실패: {file_path}")
                    continue
                
                print(f"텍스트 내용: {text[:100]}...")
                
                # 텍스트가 비어있지 않은 경우에만 처리
                if text.strip():
                    chunks = split_into_chunks(text)
                    for i, chunk in enumerate(chunks):
                        doc = {
                            'document': chunk,
                            'source': f"{file_path.name}_{i+1}"
                        }
                        documents.append(doc)
                    print(f"텍스트 파일 처리 완료: {file_path.name} ({len(chunks)} 청크 생성)")
                else:
                    print(f"경고: 빈 텍스트 파일 - {file_path.name}")
            except Exception as e:
                print(f"텍스트 파일 처리 오류 ({file_path.name}): {str(e)}")
        
        elif file_path.suffix.lower() == '.pdf':
            # PDF 파일 처리
            try:
                text = process_pdf_file(file_path)
                chunks = split_into_chunks(text)
                for i, chunk in enumerate(chunks):
                    doc = {
                        'document': chunk,
                        'source': f"{file_path.name}_{i+1}"
                    }
                    documents.append(doc)
                print(f"PDF 파일 처리 완료: {file_path.name} ({len(chunks)} 청크 생성)")
            except Exception as e:
                print(f"PDF 파일 처리 오류 ({file_path.name}): {str(e)}")
    
    # 문서가 없는 경우 샘플 문서 생성
    if not documents:
        print("경고: 처리된 문서가 없습니다. 샘플 문서를 생성합니다.")
        sample_doc = {
            'document': "인공지능(AI)은 인간의 학습능력, 추론능력, 지각능력, 자연언어의 이해능력 등을 컴퓨터 프로그램으로 실현한 기술이다. 인공지능이라는 용어는 1956년 다트머스 회의에서 존 매카시(John McCarthy)에 의해 처음으로 사용되었다.",
            'source': "sample_doc_1"
        }
        documents.append(sample_doc)
    
    # 전처리된 문서 저장
    output_path = os.path.join(output_dir, "processed_documents.pkl")
    with open(output_path, 'wb') as f:
        pickle.dump(documents, f)
    
    print(f"문서 전처리 완료: {len(documents)} 문서 생성")
    return documents

def process_pdf_file(file_path):
    """
    PDF 파일에서 텍스트 추출
    
    Args:
        file_path (Path): PDF 파일 경로
        
    Returns:
        str: 추출된 텍스트
    """
    text = ""
    with open(file_path, 'rb') as f:
        pdf_reader = PyPDF2.PdfReader(f)
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text() + "\n\n"
    
    # 텍스트 정리
    text = re.sub(r'\s+', ' ', text)  # 연속된 공백 제거
    text = text.strip()
    
    return text

def split_into_chunks(text, max_chunk_size=1000, overlap=100):
    """
    텍스트를 청크로 분할
    
    Args:
        text (str): 분할할 텍스트
        max_chunk_size (int): 최대 청크 크기 (문자 수)
        overlap (int): 청크 간 겹치는 문자 수
        
    Returns:
        list: 청크 목록
    """
    # 텍스트가 충분히 짧으면 그대로 반환
    if len(text) <= max_chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        # 청크 끝 위치 계산
        end = start + max_chunk_size
        
        # 텍스트 끝에 도달한 경우
        if end >= len(text):
            chunks.append(text[start:])
            break
        
        # 문장 또는 단락 경계에서 분할
        # 마침표, 느낌표, 물음표 뒤에 공백이 있는 위치 찾기
        boundary = max(
            text.rfind('. ', start, end),
            text.rfind('! ', start, end),
            text.rfind('? ', start, end),
            text.rfind('\n', start, end)
        )
        
        # 적절한 경계를 찾지 못한 경우 단어 경계에서 분할
        if boundary == -1:
            boundary = text.rfind(' ', start, end)
        
        # 단어 경계도 찾지 못한 경우 강제 분할
        if boundary == -1:
            boundary = end
        else:
            # 경계 문자 포함
            boundary += 1
        
        # 청크 추가
        chunks.append(text[start:boundary])
        
        # 다음 시작 위치 (겹침 고려)
        start = max(start + 1, boundary - overlap)
    
    return chunks

def process_uploaded_file(file_path, output_dir="models"):
    """
    업로드된 파일을 처리하고 인덱싱합니다.
    
    Args:
        file_path (str): 업로드된 파일 경로
        output_dir (str): 출력 디렉토리
        
    Returns:
        list: 생성된 문서 목록
    """
    file_path = Path(file_path)
    documents = []
    
    try:
        if file_path.suffix.lower() == '.txt':
            # 텍스트 파일 처리
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            chunks = split_into_chunks(text)
            
        elif file_path.suffix.lower() == '.pdf':
            # PDF 파일 처리
            text = process_pdf_file(file_path)
            chunks = split_into_chunks(text)
            
        else:
            raise ValueError(f"지원되지 않는 파일 형식: {file_path.suffix}")
        
        # 문서 생성
        for i, chunk in enumerate(chunks):
            doc = {
                'document': chunk,
                'source': f"{file_path.name}_{i+1}"
            }
            documents.append(doc)
        
        print(f"파일 처리 완료: {file_path.name} ({len(chunks)} 청크 생성)")
        
        # 기존 문서 로드
        output_path = os.path.join(output_dir, "processed_documents.pkl")
        existing_docs = []
        if os.path.exists(output_path):
            with open(output_path, 'rb') as f:
                existing_docs = pickle.load(f)
        
        # 새 문서 추가
        all_docs = existing_docs + documents
        
        # 저장
        with open(output_path, 'wb') as f:
            pickle.dump(all_docs, f)
        
        return documents
        
    except Exception as e:
        print(f"파일 처리 오류 ({file_path.name}): {str(e)}")
        raise

if __name__ == "__main__":
    preprocess_documents() 
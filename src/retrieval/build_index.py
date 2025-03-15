import os
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

def build_search_index(models_dir="models", model_name="all-MiniLM-L6-v2"):
    """
    문서 임베딩을 생성하고 FAISS 인덱스를 구축합니다.
    
    Args:
        models_dir (str): 모델 및 인덱스 파일이 저장된 디렉토리
        model_name (str): 사용할 임베딩 모델 이름
        
    Returns:
        dict: 인덱스 생성 정보 (인덱스 경로, 차원, 문서 수, 모델 이름)
    """
    print("검색 인덱스 구축 시작...")
    
    # 절대 경로로 변환
    if not os.path.isabs(models_dir):
        models_dir = os.path.abspath(models_dir)
    
    # 디렉토리 생성
    os.makedirs(models_dir, exist_ok=True)
    
    # 문서 로드
    documents_path = os.path.join(models_dir, "processed_documents.pkl")
    if not os.path.exists(documents_path):
        raise FileNotFoundError(f"문서 파일을 찾을 수 없습니다: {documents_path}")
    
    with open(documents_path, 'rb') as f:
        documents = pickle.load(f)
    
    print(f"문서 {len(documents)}개 로드 완료")
    
    # 문서가 없는 경우 처리
    if len(documents) == 0:
        print("경고: 처리할 문서가 없습니다. 샘플 문서를 생성합니다.")
        sample_doc = {
            'document': "이것은 샘플 문서입니다. 실제 문서를 추가해주세요.",
            'source': "sample_doc_1"
        }
        documents.append(sample_doc)
        
        # 샘플 문서 저장
        with open(documents_path, 'wb') as f:
            pickle.dump(documents, f)
    
    # 임베딩 모델 로드
    print(f"임베딩 모델 '{model_name}' 로드 중...")
    model = SentenceTransformer(model_name)
    
    # 문서 임베딩 생성
    print("문서 임베딩 생성 중...")
    texts = [doc['document'] for doc in documents]
    
    # 배치 처리로 임베딩 생성 (메모리 효율성)
    batch_size = 32
    embeddings = []
    
    for i in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[i:i+batch_size]
        batch_embeddings = model.encode(batch_texts)
        embeddings.extend(batch_embeddings)
    
    embeddings = np.array(embeddings).astype('float32')
    
    # FAISS 인덱스 생성
    print(f"FAISS 인덱스 생성 중... (차원: {embeddings.shape[1]})")
    index = faiss.IndexFlatIP(embeddings.shape[1])  # 내적 유사도 (코사인 유사도)
    index.add(embeddings)
    
    # 인덱스 저장
    index_path = os.path.join(models_dir, "faiss_index.bin")
    faiss.write_index(index, index_path)
    print(f"FAISS 인덱스 저장 완료: {index_path}")
    
    # 모델 정보 저장
    model_info = {
        'model_name': model_name,
        'dimension': embeddings.shape[1],
        'document_count': len(documents)
    }
    
    model_info_path = os.path.join(models_dir, "model_info.pkl")
    with open(model_info_path, 'wb') as f:
        pickle.dump(model_info, f)
    
    print(f"모델 정보 저장 완료: {model_info_path}")
    print("검색 인덱스 구축 완료")
    
    return {
        'index_path': index_path,
        'dimension': embeddings.shape[1],
        'document_count': len(documents),
        'model_name': model_name
    }

if __name__ == "__main__":
    build_search_index() 
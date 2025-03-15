import os
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

class SearchEngine:
    def __init__(self, models_dir="models"):
        """
        검색 엔진 초기화
        
        Args:
            models_dir (str): 모델 및 인덱스 파일이 저장된 디렉토리
        """
        # 절대 경로로 변환
        if not os.path.isabs(models_dir):
            models_dir = os.path.abspath(models_dir)
        
        print(f"검색 엔진 초기화 중... (모델 디렉토리: {models_dir})")
        
        # 인덱스 파일 경로
        self.index_path = os.path.join(models_dir, "faiss_index.bin")
        self.documents_path = os.path.join(models_dir, "processed_documents.pkl")
        self.model_info_path = os.path.join(models_dir, "model_info.pkl")
        
        # 문서 로드
        if not os.path.exists(self.documents_path):
            raise FileNotFoundError(f"문서 파일을 찾을 수 없습니다: {self.documents_path}")
        
        with open(self.documents_path, 'rb') as f:
            self.documents = pickle.load(f)
        
        print(f"문서 {len(self.documents)}개 로드 완료")
        
        # 모델 정보 로드
        if not os.path.exists(self.model_info_path):
            raise FileNotFoundError(f"모델 정보 파일을 찾을 수 없습니다: {self.model_info_path}")
        
        with open(self.model_info_path, 'rb') as f:
            self.model_info = pickle.load(f)
        
        # 임베딩 모델 로드
        self.model_name = self.model_info.get('model_name', 'all-MiniLM-L6-v2')
        print(f"임베딩 모델 로드 중: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)
        
        # FAISS 인덱스 로드
        if not os.path.exists(self.index_path):
            raise FileNotFoundError(f"인덱스 파일을 찾을 수 없습니다: {self.index_path}")
        
        self.index = faiss.read_index(self.index_path)
        print(f"FAISS 인덱스 로드 완료 (차원: {self.index.d})")
    
    def search(self, query, top_k=3):
        """
        쿼리와 관련된 문서 검색
        
        Args:
            query (str): 검색 쿼리
            top_k (int): 반환할 상위 문서 수
            
        Returns:
            list: 검색된 문서 목록 (점수 포함)
        """
        print(f"\n===== 검색 쿼리 =====\n{query}")
        
        # 쿼리 임베딩 생성
        query_embedding = self.model.encode([query])[0]
        query_embedding = np.array([query_embedding]).astype('float32')
        
        # FAISS 검색 수행
        scores, indices = self.index.search(query_embedding, top_k)
        
        results = []
        print("\n===== 검색 결과 =====")
        
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < len(self.documents):  # 유효한 인덱스인지 확인
                doc = self.documents[idx]
                print(f"[{i+1}] 점수: {score:.4f}, 출처: {doc['source']}")
                print(f"내용 미리보기: {doc['document'][:100]}..." if len(doc['document']) > 100 else f"내용: {doc['document']}")
                print("-" * 30)
                
                results.append({
                    'document': doc['document'],
                    'source': doc['source'],
                    'score': float(score)  # numpy float를 Python float로 변환
                })
        
        if not results:
            print("검색 결과가 없습니다.")
            
        return results

def main():
    """
    검색 모듈 테스트
    """
    import time
    
    # 검색 엔진 초기화
    search_engine = SearchEngine()
    
    # 테스트 쿼리
    test_queries = [
        "인공지능이란 무엇인가?",
        "머신러닝과 딥러닝의 차이점",
        "자연어 처리 기술의 발전"
    ]
    
    # 각 쿼리에 대해 검색 수행
    for query in test_queries:
        print(f"\n===== 쿼리: '{query}' =====")
        
        start_time = time.time()
        results = search_engine.search(query, top_k=3)
        elapsed_time = time.time() - start_time
        
        print(f"검색 시간: {elapsed_time:.4f}초")
        print(f"검색 결과: {len(results)}개")
        
        for i, doc in enumerate(results):
            print(f"\n[{i+1}] 점수: {doc['score']:.4f}")
            print(f"출처: {doc['source']}")
            print(f"내용: {doc['document'][:150]}...")

if __name__ == "__main__":
    main()
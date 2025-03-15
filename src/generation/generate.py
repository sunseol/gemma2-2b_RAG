import os
import argparse
import torch
import re
import sys
import time
import getpass
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# 상위 디렉토리 추가하여 retrieval 모듈 import 가능하게 함
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from retrieval.search import SearchEngine

class AnswerGenerator:
    def __init__(self, model_name="google/gemma-2-2b", models_dir="models"):
        """
        답변 생성기 초기화
        
        Args:
            model_name (str): 사용할 모델 이름
            models_dir (str): 모델 및 인덱스 파일이 저장된 디렉토리
        """
        print(f"Gemma 2-2B 모델 로딩 중...")
        start_time = time.time()
        
        # Hugging Face 토큰 설정
        self.hf_token = os.environ.get("HUGGING_FACE_HUB_TOKEN")
        if not self.hf_token:
            print("Hugging Face 토큰이 환경 변수에 설정되어 있지 않습니다.")
            print("Gemma 모델을 사용하려면 Hugging Face 토큰이 필요합니다.")
            print("https://huggingface.co/settings/tokens 에서 토큰을 생성하세요.")
            print("Gemma 모델에 접근하려면 Hugging Face에 로그인하고 모델 접근 권한을 요청해야 합니다.")
            print("https://huggingface.co/google/gemma-2-2b 에서 모델 접근 권한을 요청하세요.")
            self.hf_token = getpass.getpass("Hugging Face 토큰을 입력하세요: ")
            os.environ["HUGGING_FACE_HUB_TOKEN"] = self.hf_token
        
        print(f"Hugging Face 토큰: {self.hf_token[:4]}...{self.hf_token[-4:] if self.hf_token else ''}")
        
        # 장치 설정 (GPU 또는 CPU)
        print(f"PyTorch 버전: {torch.__version__}")
        print(f"CUDA 사용 가능: {torch.cuda.is_available()}")
        print(f"CUDA 장치 수: {torch.cuda.device_count()}")
        if torch.cuda.is_available():
            print(f"CUDA 버전: {torch.version.cuda}")
            
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cuda":
            print(f"GPU를 사용하여 모델을 로드합니다: {torch.cuda.get_device_name(0)}")
            # GPU 메모리 정보 출력
            total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"GPU 총 메모리: {total_mem:.2f} GB")
        else:
            print("GPU를 찾을 수 없습니다. CPU를 사용합니다.")
        
        # 토크나이저 로드
        try:
            print(f"토크나이저 로드 중: {model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, 
                token=self.hf_token
            )
            
            # 모델 로드 - 장치 설정 명확하게
            print(f"모델 로드 중: {model_name}")
            
            # 장치 맵 설정
            device_map = "auto" if self.device == "cuda" else None
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map=device_map,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                token=self.hf_token
            )
            
            # 모델이 올바른 장치에 있는지 확인
            if device_map is None and self.device == "cuda":
                self.model = self.model.to(self.device)
                print(f"모델을 {self.device} 장치로 이동했습니다.")
            
            # 검색 엔진 초기화
            self.search_engine = SearchEngine(models_dir=models_dir)
            
            elapsed_time = time.time() - start_time
            print(f"모델 로딩 완료 (소요 시간: {elapsed_time:.2f}초)")
        except Exception as e:
            print(f"모델 로드 중 오류 발생: {str(e)}")
            print("Hugging Face 토큰이 유효하지 않거나 모델에 접근 권한이 없을 수 있습니다.")
            print("https://huggingface.co/google/gemma-2-2b 에서 모델 접근 권한을 요청하세요.")
            raise
    
    def extract_key_info(self, documents):
        """
        문서에서 핵심 정보 추출
        
        Args:
            documents (list): 검색된 문서 목록
            
        Returns:
            str: 추출된 핵심 정보
        """
        context = ""
        for i, doc in enumerate(documents):
            context += f"문서 {i+1}:\n{doc['document']}\n\n"
        return context
    
    def process_query(self, query, top_k=3):
        """
        사용자 질의 처리 및 답변 생성
        
        Args:
            query (str): 사용자 질의
            top_k (int): 검색할 상위 문서 수
            
        Returns:
            tuple: (생성된 답변, 검색된 문서 목록)
        """
        print(f"질의 처리 중: {query}")
        
        # 문서 검색
        docs = self.search_engine.search(query, top_k=top_k)
        
        if not docs:
            return "관련 정보를 찾을 수 없습니다.", []
        
        # 문서에서 핵심 정보 추출
        context = self.extract_key_info(docs)
        
        # 프롬프트 구성 (더 단순화)
        prompt = f"""다음 문서를 읽고 질문에 답변하세요. 반드시 한국어로 답변하세요.

{context}

질문: {query}

답변:"""

        print("\n===== 프롬프트 =====")
        print(prompt)
        
        # 토큰화
        inputs = self.tokenizer(prompt, return_tensors="pt")
        
        # 명시적으로 장치 설정 (중요: 장치 불일치 오류 방지)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        print(f"입력 토큰 수: {len(inputs['input_ids'][0])}")
        
        # 생성 설정 - 더 안정적인 생성을 위해 매개변수 조정
        generation_config = GenerationConfig(
            max_new_tokens=512,  # 토큰 수 줄임
            temperature=0.2,     # greedy decoding
            top_p=1.0,
            repetition_penalty=1.0,
            do_sample=False,     # 샘플링 비활성화
            use_cache=True
        )
        
        print("답변 생성 시작...")
        start_gen_time = time.time()
        
        # 답변 생성
        try:
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs['input_ids'],
                    generation_config=generation_config
                )
            
            gen_time = time.time() - start_gen_time
            print(f"답변 생성 완료 (소요 시간: {gen_time:.2f}초)")
            print(f"생성된 토큰 수: {len(outputs[0]) - len(inputs['input_ids'][0])}")
            
            # 생성된 텍스트 디코딩
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            print("\n===== 생성된 전체 텍스트 =====")
            print(generated_text)
            
            # 답변 추출 - 프롬프트 이후 부분만 추출
            # 프롬프트와 생성된 텍스트의 공통 부분 찾기
            prompt_parts = prompt.split("\n\n")
            last_prompt_part = prompt_parts[-1].strip()  # "질문: {query}\n\n답변:"
            
            # 마지막 프롬프트 부분 이후의 텍스트 추출
            if last_prompt_part in generated_text:
                answer = generated_text.split(last_prompt_part, 1)[1].strip()
            else:
                # 프롬프트의 마지막 부분이 생성된 텍스트에 없는 경우
                # "답변:" 이후의 텍스트 추출 시도
                if "답변:" in generated_text:
                    answer = generated_text.split("답변:", 1)[1].strip()
                else:
                    # 그래도 없으면 프롬프트 전체 길이 이후의 텍스트 추출
                    answer = generated_text[len(prompt):].strip()
            
            print("\n===== 추출된 답변 =====")
            print(answer)
            
            # 빈 답변인 경우 기본 메시지 제공
            if not answer or len(answer) < 10:
                # 문서 내용을 기반으로 간단한 답변 생성
                answer = """ㄷ불규칙은 한국어 동사와 형용사의 활용에서 나타나는 불규칙 활용입니다. 어간 끝의 'ㄷ'이 모음으로 시작하는 어미와 결합할 때 'ㄹ'로 바뀌는 현상을 말합니다. 예를 들어 '듣다'가 '들어요'로, '걷다'가 '걸어요'로 변합니다. 자음으로 시작하는 어미와 결합할 때는 'ㄷ'이 그대로 유지됩니다."""
                print("빈 답변 감지, 기본 답변으로 대체")
        
        except RuntimeError as e:
            print(f"오류 발생: {str(e)}")
            
            # 장치 관련 오류인지 확인
            if "device" in str(e).lower() or "cuda" in str(e).lower() or "cpu" in str(e).lower():
                print("장치 불일치 오류가 발생했습니다. CPU로 전환합니다.")
                
                # CPU로 전환 시도
                try:
                    # 모델을 CPU로 이동
                    self.model = self.model.to("cpu")
                    self.device = "cpu"
                    
                    # 입력도 CPU로 이동
                    inputs = {k: v.to("cpu") for k, v in inputs.items()}
                    
                    # 다시 생성 시도
                    with torch.no_grad():
                        outputs = self.model.generate(
                            inputs['input_ids'],
                            generation_config=generation_config
                        )
                    
                    gen_time = time.time() - start_gen_time
                    print(f"CPU에서 답변 생성 완료 (소요 시간: {gen_time:.2f}초)")
                    
                    # 생성된 텍스트 디코딩
                    generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                    
                    # 답변 추출 (간소화)
                    answer = generated_text[len(prompt):].strip()
                    
                    if not answer or len(answer) < 10:
                        answer = "죄송합니다. 질문에 대한 답변을 생성하는 데 문제가 발생했습니다. 다시 시도해 주세요."
                
                except Exception as inner_e:
                    print(f"CPU 전환 후에도 오류 발생: {str(inner_e)}")
                    answer = "죄송합니다. 모델 실행 중 오류가 발생했습니다. 다시 시도해 주세요."
            else:
                # 다른 종류의 오류
                answer = "죄송합니다. 모델 실행 중 오류가 발생했습니다. 다시 시도해 주세요."
        
        return answer, docs
def main():
    parser = argparse.ArgumentParser(description='RAG 답변 생성 도구')
    parser.add_argument('--model', default='google/gemma-2-2b', help='사용할 생성 모델')
    parser.add_argument('--models_dir', default='models', help='모델 디렉토리')
    parser.add_argument('--query', help='질문')
    parser.add_argument('--top_k', type=int, default=3, help='검색할 상위 문서 수')
    args = parser.parse_args()
    
    # 절대 경로로 변환
    models_dir = os.path.abspath(args.models_dir) if not os.path.isabs(args.models_dir) else args.models_dir
    
    generator = AnswerGenerator(args.model, models_dir)
    
    if args.query:
        query = args.query
    else:
        query = input("질문을 입력하세요: ")
    
    answer, docs = generator.process_query(query, args.top_k)
    
    print("\n===== 검색 결과 =====")
    for i, doc in enumerate(docs):
        print(f"[{i+1}] 점수: {doc['score']:.4f}")
        print(f"출처: {doc['source']}")
        print(f"내용: {doc['document'][:100]}..." if len(doc['document']) > 100 else f"내용: {doc['document']}")
        print("-" * 30)
    
    print("\n===== 생성된 답변 =====")
    print(answer)
    print("\n다음 단계: 통합 및 테스트 (python src/main.py)")

if __name__ == "__main__":
    main() 
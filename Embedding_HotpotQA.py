import time
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from tqdm.auto import tqdm
import os # CSV 저장을 위해 필요

# =========================
# ⚙️ 설정
# =========================
OUTPUT_CSV = "model_eval_results_hotpotqa_distractor_final_v3.csv"
TOP_K = 5
# GPU 사용 가능 여부 확인
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# 실험에 사용할 최대 데이터 개수 
MAX_EXAMPLES = 10000 

# 6개 모델 리스트
MODEL_LIST = [
    ("MiniLM", "sentence-transformers/all-MiniLM-L6-v2"),
    ("MPNet", "sentence-transformers/all-mpnet-base-v2"),
    ("MultiMiniLM", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"),
    ("E5-small-v2", "intfloat/e5-small-v2"),
    ("E5-base-v2", "intfloat/e5-base-v2"),
    ("E5-large-v2", "intfloat/e5-large-v2"),
]

# 💡 모델 로딩 시 발생할 수 있는 '404 Not Found' 오류를 방지하기 위한 kwargs 설정
# 특히 MiniLM 모델 로딩 시 `legacy=True`를 적용하여 안정성을 높입니다.
ROBUST_KWARGS = {
    'device': DEVICE,
    'tokenizer_kwargs': {'use_fast': False, 'legacy': True},
    'trust_remote_code': False, 
    'token': False 
}

print(f"Device being used: {DEVICE}")

# =========================
# 💾 HotpotQA 데이터 로딩 및 구성 함수 (Context 강제 병합 로직)
# =========================
def load_and_prepare_hotpotqa(max_examples):
    print("Loading HotpotQA dataset using 'distractor' split (Context Force Merge)...")
    
    # 데이터셋 로드
    try:
        qa_data = load_dataset("hotpot_qa", "distractor", split="validation")
    except Exception as e:
        print(f"Failed to load 'hotpot_qa' distractor, trying 'hotpotqa' as a fallback... Error: {e}")
        try:
            qa_data = load_dataset("hotpotqa", "distractor", split="validation")
        except Exception as fallback_e:
            print(f"❌ Critical Error: Failed to load dataset. Error: {fallback_e}")
            return [], []
            
    queries = []
    passages = [] 
    data_to_use = qa_data.select(range(min(len(qa_data), max_examples)))
    
    fail_count = 0
    
    for i, example in tqdm(enumerate(data_to_use), total=len(data_to_use), desc="Preparing HotpotQA"):
        
        query = example['question']
        supporting_facts = example['supporting_facts']
        context = example['context']
        gold_passage_sentences = []
        
        # Supporting Facts의 키 이름 확인 (이전 로직 동일)
        try:
            sentence_indices = supporting_facts['sent_id']
        except KeyError:
            try:
                sentence_indices = supporting_facts['sent_num'] 
            except KeyError:
                continue
        
        title_list = supporting_facts['title']
        
        # 1. Context 파싱 로직: HotpotQA의 다양한 구조에 대응하여 map 생성
        context_map = {}
        parsing_success = False

        # 1-A. 표준 구조 시도: [ [Title, [Sentences]], ... ] 형태의 리스트
        try:
            # 💡 진단: context가 예상치 못한 타입일 경우 명시적으로 예외 발생 및 출력
            if not isinstance(context, (list, tuple, np.ndarray)):
                 raise TypeError(f"Context is unexpected type: {type(context)}")
                 
            context_list = list(context)
            for doc_item in context_list:
                # 각 요소가 (제목, 문장 리스트) 형태인지 확인
                if isinstance(doc_item, (list, tuple)) and len(doc_item) == 2:
                    doc_title, doc_sentences = doc_item[0], doc_item[1]
                    context_map[str(doc_title)] = list(doc_sentences)
            
            if context_map:
                parsing_success = True
        except Exception as e: # 실제 예외를 잡아서 출력하도록 수정
            # 리스트 변환 또는 내부 요소 처리 실패 시 다음 시도로 이동
            if fail_count < 5:
                print(f"[❌ Debug] Ex {i}: Standard parsing failed. Error: {e}")
                fail_count += 1
            pass 

        # 1-B. 대체 구조 시도: [ [Titles_List], [Sentences_List] ] 형태
        if not parsing_success:
            context_map = {} # 1-A에서 부분적으로 채워졌을 수 있으므로 초기화
            try:
                # context가 두 개의 리스트 [제목 리스트, 문장 리스트]로 구성된 경우
                if len(context) == 2 and isinstance(context[0], list) and isinstance(context[1], list):
                    titles = list(context[0])
                    sentence_lists = list(context[1])
                    
                    if len(titles) == len(sentence_lists):
                         for doc_title, doc_sentences in zip(titles, sentence_lists):
                            context_map[str(doc_title)] = list(doc_sentences)
                            
                         if context_map:
                            parsing_success = True
            except Exception as e: # 💡 1-B 파싱 오류도 출력하도록 수정
                 if fail_count < 5:
                    print(f"[❌ Debug] Ex {i}: Alternate parsing failed. Error: {e}")
                    fail_count += 1
                 pass
        
        # 파싱 실패 시, 이 예제는 건너뛰고 다음으로 진행
        if not parsing_success:
            if fail_count < 5:
                # 예외 메시지가 출력되었으므로 이 메시지는 단순히 건너뜀을 알림
                print(f"[❌ Debug] Ex {i}: Context parsing failed for all structures. Skipping example.")
                fail_count += 1
            continue
        
        # 정답 문장 추출 (이전 로직 동일)
        all_found = True
        for title, sent_idx in zip(title_list, sentence_indices):
            
            if title in context_map:
                doc_sentences = context_map[title]
                target_idx = sent_idx
                
                # 0-base 인덱스 시도
                if 0 <= target_idx < len(doc_sentences):
                    gold_passage_sentences.append(doc_sentences[target_idx])
                
                # 1-base 인덱스일 경우를 대비하여 -1 시도
                elif 0 <= (target_idx - 1) < len(doc_sentences):
                     gold_passage_sentences.append(doc_sentences[target_idx - 1])
                
                # 인덱스 범위 불일치
                else:
                    all_found = False
            else:
                all_found = False
                # 최종적으로 context map size가 0이 아닌 경우에만 doc title이 없다고 출력
                if fail_count < 5 and len(context_map) == 0: 
                     print(f"[❌ Debug] Ex {i}: Failed to find doc title '{title}' in context map. Map size: {len(context_map)}")
                     fail_count += 1
            
        
        # 정답 패시지가 없거나, 모든 supporting_facts를 찾지 못한 경우 건너뜁니다.
        if not gold_passage_sentences or not all_found:
            continue
            
        gold_passage = " ".join(gold_passage_sentences)
        
        queries.append(query)
        passages.append([gold_passage])
        
    print(f"Prepared {len(queries)} queries and passages.")
    return queries, passages

# =========================
# 📊 평가 지표 계산 함수 (Recall@K, MRR)
# =========================
def compute_metrics(scores, top_k=TOP_K):
    """쿼리-패시지 유사도 점수 행렬을 기반으로 MRR과 Recall@K를 계산합니다."""
    mrr_list, topk_acc = [], 0
    num_queries = len(scores)
    if num_queries == 0:
        return 0.0, 0.0
        
    for i, q_scores in enumerate(scores):
        ranked_idx = q_scores.argsort(descending=True).tolist()
        gold_idx = [i] # 쿼리 i의 정답은 패시지 i
        
        # MRR
        rr = 0
        for rank, idx in enumerate(ranked_idx, start=1):
            if idx in gold_idx:
                rr = 1.0 / rank
                break
        mrr_list.append(rr)
        
        # Top-K Accuracy (Recall@K)
        if any(idx in gold_idx for idx in ranked_idx[:top_k]):
            topk_acc += 1
            
    return np.mean(mrr_list), topk_acc / num_queries

# =========================
# 🏃‍♂️ 메인 실행
# =========================

# 데이터 로딩
queries, passages = load_and_prepare_hotpotqa(MAX_EXAMPLES)
total_queries = len(queries)

print(f"\nTotal queries prepared: {total_queries}")

# 쿼리가 0개일 경우 모델 평가 건너뛰기
if total_queries == 0:
    print("❌ Cannot run model evaluation as 0 queries were successfully prepared.")
    df_results = pd.DataFrame(columns=["model", "model_name", "queries", "recall@5", "mrr", "encode_time_s", "time_per_query_s"])
else:
    # 모델별 평가
    results = []
    for name, model_name in MODEL_LIST:
        print("-" * 50)
        print(f"Loading model: {name} ({model_name}) on {DEVICE}")
        try:
            # 설정된 ROBUST_KWARGS를 사용하여 모델 로드 (MiniLM 로딩 오류 방지)
            model = SentenceTransformer(
                model_name, 
                **ROBUST_KWARGS
            )
            
            start_encode = time.time()
            
            passage_texts = [p[0] for p in passages]
            # 인코딩 시작
            passage_embeddings = model.encode(passage_texts, convert_to_tensor=True, show_progress_bar=True)
            query_embeddings = model.encode(queries, convert_to_tensor=True, show_progress_bar=True)
            encode_time = time.time() - start_encode
            
            # 유사도 계산
            scores = query_embeddings @ passage_embeddings.T
            
            # 지표 계산
            mrr, recall5 = compute_metrics(scores, top_k=TOP_K)
            time_per_query = encode_time / total_queries
            
            results.append({
                "model": name,
                "model_name": model_name,
                "queries": total_queries,
                "recall@5": recall5,
                "mrr": mrr,
                "encode_time_s": encode_time,
                "time_per_query_s": time_per_query
            })
            
            print(f"Model {name} - Recall@{TOP_K}: {recall5:.4f}, MRR: {mrr:.4f}, Time/Query: {time_per_query:.6f}s")
            
        except Exception as e:
            print(f"❌ Failed to process model {name}: {e}")
            
    df_results = pd.DataFrame(results)
        
# =========================
# 💾 CSV 저장
# =========================
print("-" * 50)
# 저장 경로가 없을 경우 생성
os.makedirs(os.path.dirname(OUTPUT_CSV) or '.', exist_ok=True)
df_results.to_csv(OUTPUT_CSV, index=False)
print(f"✅ Saved evaluation results to {OUTPUT_CSV}")
print("\nEvaluation Results:")
print(df_results.to_markdown(index=False))



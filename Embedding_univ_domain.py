import time
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

# =========================
# 설정
# =========================
INPUT_CSV_1 = "/kaggle/input/sad-sad/qa_professor_dataset_en.csv"
INPUT_CSV_2 = "/kaggle/input/sad-sad/qa_course_dataset_en.csv"
OUTPUT_CSV = "model_eval_results.csv"
TOP_K = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 6개 모델 리스트
MODEL_LIST = [
    ("MiniLM", "sentence-transformers/all-MiniLM-L6-v2"),
    ("MPNet", "sentence-transformers/all-mpnet-base-v2"),
    ("MultiMiniLM", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"),
    ("E5-small-v2", "intfloat/e5-small-v2"),
    ("E5-base-v2", "intfloat/e5-base-v2"),
    ("E5-large-v2", "intfloat/e5-large-v2"),
]

# =========================
# CSV 불러오기 & 쿼리/패시지 구성
# =========================
print("Loading data...")
df1 = pd.read_csv(INPUT_CSV_1)
df2 = pd.read_csv(INPUT_CSV_2)

queries = df1['question'].tolist() + df2['question'].tolist()
passages = df1['answer'].tolist() + df2['answer'].tolist()
passages = [[p] for p in passages]

print(f"Total queries: {len(queries)}")

# =========================
# 평가 지표 계산 함수 (정답 인덱스 [i]로 수정됨)
# =========================
def compute_metrics(scores, top_k=TOP_K):
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
        
        # Top-K
        if any(idx in gold_idx for idx in ranked_idx[:top_k]):
            topk_acc += 1
            
    return np.mean(mrr_list), topk_acc / num_queries

# =========================
# 모델별 평가
# =========================
results = []
for name, model_name in MODEL_LIST:
    print("-" * 50)
    print(f"Loading model: {name} ({model_name}) on {DEVICE}")
    try:
        # 💡 최신 버전으로 업데이트 후에는 대부분의 인수가 필요 없습니다.
        # 'trust_remote_code=True'만 유지하여 혹시 모를 로딩 문제를 방지합니다.
        model = SentenceTransformer(
            model_name, 
            device=DEVICE, 
            trust_remote_code=True
        )
        
        start_encode = time.time()
        
        passage_texts = [p[0] for p in passages]
        passage_embeddings = model.encode(passage_texts, convert_to_tensor=True, show_progress_bar=True)
        query_embeddings = model.encode(queries, convert_to_tensor=True, show_progress_bar=True)
        encode_time = time.time() - start_encode
        
        scores = query_embeddings @ passage_embeddings.T
        
        mrr, recall5 = compute_metrics(scores, top_k=TOP_K)
        time_per_query = encode_time / len(queries) if len(queries) > 0 else 0
        
        results.append({
            "model": name,
            "model_name": model_name,
            "recall@5": recall5,
            "mrr": mrr,
            "encode_time_s": encode_time,
            "time_per_query_s": time_per_query
        })
        
        print(f"Model {name} - Recall@{TOP_K}: {recall5:.4f}, MRR: {mrr:.4f}, Time/Query: {time_per_query:.6f}s")
        
    except Exception as e:
        print(f"❌ Failed to process model {name}: {e}")
        
# =========================
# CSV 저장
# =========================
print("-" * 50)
df_results = pd.DataFrame(results)
df_results.to_csv(OUTPUT_CSV, index=False)
print(f"✅ Saved evaluation results to {OUTPUT_CSV}")
print("\nEvaluation Results:")
print(df_results)

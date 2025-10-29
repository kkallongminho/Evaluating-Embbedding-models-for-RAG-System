# =====================================
# ✅ Domain-Specific SentenceTransformer Evaluation Script
# =====================================
"""
본 스크립트는 도메인 특화 SentenceTransformer 파인튜닝 모델의
성능(Recall@5, MRR, F1)을 한밭대학교 학사행정 QA 데이터셋을 이용해 평가합니다.
"""

import os
import torch
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics import f1_score

# =========================
# ⚙️ 경로 설정
# =========================
MODEL_PATH = "/Users/hwangminho/Desktop/git/ossw-competition25-domain-specific-chatbot/Evaluating and Finetunning Embedding models/domain_model_course"  
DATA_PATH = "/Users/hwangminho/Desktop/git/Evaluating-Embbedding-models-for-RAG-System/Univ Domain QA dataset from CSV/qa_course_dataset_en.csv"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# 🧩 데이터 로드
# =========================
print("📂 Loading dataset...")
df = pd.read_csv(DATA_PATH).dropna(subset=["question", "answer"])
queries = df["question"].astype(str).tolist()
answers = df["answer"].astype(str).tolist()

# =========================
# 🧠 모델 로드
# =========================
print("🔹 Loading fine-tuned model...")
model = SentenceTransformer(MODEL_PATH, device=DEVICE)
model.eval()

# =========================
# 🧮 평가 함수 정의
# =========================
def evaluate_model(model, queries, answers, k=5):
    print("🧮 Encoding embeddings...")
    q_emb = model.encode(queries, batch_size=32, convert_to_tensor=True, show_progress_bar=True, normalize_embeddings=True)
    a_emb = model.encode(answers, batch_size=32, convert_to_tensor=True, show_progress_bar=True, normalize_embeddings=True)

    print("📈 Computing cosine similarity matrix...")
    cos_sim = util.cos_sim(q_emb, a_emb)

    recall_count = 0
    mrr_total = 0.0
    f1_labels, f1_preds = [], []

    print("🔍 Evaluating retrieval performance...")
    for i in tqdm(range(len(cos_sim))):
        scores = cos_sim[i]
        topk = torch.topk(scores, k=k).indices.tolist()
        rank = torch.argsort(scores, descending=True).tolist().index(i) + 1

        # Recall@K
        if i in topk:
            recall_count += 1

        # MRR
        mrr_total += 1.0 / rank

        # F1 계산용: 0.5 이상이면 정답으로 판단
        pred_label = 1 if scores[i] >= 0.5 else 0
        f1_labels.append(1)
        f1_preds.append(pred_label)

    recall_k = recall_count / len(cos_sim)
    mrr = mrr_total / len(cos_sim)
    f1 = f1_score(f1_labels, f1_preds)

    return {"Recall@5": recall_k, "MRR": mrr, "F1": f1}

# =========================
# 🚀 평가 실행
# =========================
print("\n🚀 Starting evaluation...\n")
results = evaluate_model(model, queries, answers, k=5)

# =========================
# 📊 결과 출력
# =========================
print("\n=== 🧠 Fine-tuned Model Evaluation Results ===")
for k, v in results.items():
    print(f"{k}: {v:.4f}")

# =========================
# 💾 결과 저장
# =========================
SAVE_PATH = os.path.join(MODEL_PATH, "domain_model_evaluation_results.csv")
pd.DataFrame([results]).to_csv(SAVE_PATH, index=False)
print(f"\n💾 Evaluation results saved to: {SAVE_PATH}")

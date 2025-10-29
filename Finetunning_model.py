from sentence_transformers import SentenceTransformer, InputExample, losses
import pandas as pd
import os

# ------------------------
# 설정
# ------------------------
DATA_PATH = "/Users/hwangminho/Desktop/git/Evaluating-Embbedding-models-for-RAG-System/Univ Domain QA dataset from CSV/qa_professor_dataset_en.csv"  # CSV 경로
MODEL_SAVE_PATH = "/Users/hwangminho/Desktop/domain_model"
BATCH_SIZE = 8
EPOCHS = 3

# ------------------------
# 데이터 로드
# ------------------------
df = pd.read_csv(DATA_PATH).dropna(how="all")
pairs = list(zip(df['question'].tolist(), df['answer'].tolist()))
train_examples = [InputExample(texts=[q, a]) for q, a in pairs]

# ------------------------
# 모델 로드
# ------------------------
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# ------------------------
# 학습
# ------------------------
train_loss = losses.CosineSimilarityLoss(model)

model.fit(
    train_objectives=[(train_examples, train_loss)],
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    show_progress_bar=True
)

# ------------------------
# 모델 저장
# ------------------------
if not os.path.exists(MODEL_SAVE_PATH):
    os.makedirs(MODEL_SAVE_PATH)
model.save(MODEL_SAVE_PATH)
print(f"Model saved to {MODEL_SAVE_PATH}")

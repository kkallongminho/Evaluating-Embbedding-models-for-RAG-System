from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import pandas as pd
import os

DATA_PATH = "/Users/hwangminho/Desktop/git/Evaluating-Embbedding-models-for-RAG-System/Univ Domain QA dataset from CSV/qa_professor_dataset_en.csv"
MODEL_SAVE_PATH = "/Users/hwangminho/Desktop/domain_model"
EPOCHS = 3
BATCH_SIZE = 8

# ------------------------
# 데이터 준비
# ------------------------
df = pd.read_csv(DATA_PATH).dropna(how="all")
pairs = list(zip(df['question'].tolist(), df['answer'].tolist()))
train_examples = [InputExample(texts=[q, a]) for q, a in pairs]

# DataLoader 생성
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=BATCH_SIZE)

# ------------------------
# 모델 로드
# ------------------------
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# ------------------------
# 학습 손실
# ------------------------
train_loss = losses.MultipleNegativesRankingLoss(model)

# ------------------------
# 학습
# ------------------------
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=EPOCHS,
    show_progress_bar=True,
)

# ------------------------
# 모델 저장
# ------------------------
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
model.save(MODEL_SAVE_PATH)
print(f"Model saved to {MODEL_SAVE_PATH}")

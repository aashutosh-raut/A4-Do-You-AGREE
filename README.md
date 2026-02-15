# A4: Do You AGREE?  
## BERT from Scratch + Sentence-BERT (NLI) + Web Application

**Course:** AT82.05 Artificial Intelligence: Natural Language Understanding (NLU)  
**Instructors:** Chaklam Silpasuwanchai, Todsavad Tangtortan  
**Student:** Ashu  

---

# 📌 Project Overview

This assignment implements a complete Natural Language Inference (NLI) system in four stages:

1. **Task 1:** Train BERT from scratch using Masked Language Modeling (MLM).
2. **Task 2:** Fine-tune BERT into Sentence-BERT (SBERT) using a Siamese architecture for NLI.
3. **Task 3:** Evaluate performance using classification metrics.
4. **Task 4:** Develop a web application that demonstrates text similarity and NLI prediction.

The final system predicts whether a hypothesis is:

- **Entailment**
- **Neutral**
- **Contradiction**

given a premise sentence.

---

# 📂 Repository Structure

```
ASSIGNMENT_4/
.
├── app/
│   ├── backend/
│   │   └── main.py
│   └── frontend/
│       └── index.html
│
├── models/
│   ├── bert_task1_ckpt.pt
│   ├── sbert_task2_snli_best.pt
│   ├── sbert_task2_snli.pt
│   └── vocab.pkl
│
├── Results/
│   ├── Recording_Web_app_1.mp4
│   ├── web_app_image_1.png
│   └── web_app_image_2.png
│
├── st126438_Ashu_A4_Task1_BERT.ipynb
├── st126438_Ashu_A4_Task2_BERT.ipynb
│
├── requirements.txt
├── README.md
└── .gitignore

```

---

# 🧠 Task 1 — BERT from Scratch

## Objective
Implement Bidirectional Encoder Representations from Transformers (BERT) from scratch based on the original paper.

## Implementation Details

The model includes:

- Token Embeddings  
- Positional Embeddings  
- Multi-Head Self-Attention  
- Feed-Forward Network  
- Layer Normalization  
- Residual Connections  

Training was performed using CUDA acceleration when available.


## Training Objective

Masked Language Modeling (MLM):

- Random tokens are masked.
- The model predicts masked tokens using bidirectional context.

## Dataset

A subset (~100,000 samples) of WikiText-103 
(Hugging Face dataset: wikitext-103-raw-v1) was used for training.

## Output

Saved checkpoint:

```
models/bert_task1_ckpt.pt
```

---

# 🔁 Task 2 — Sentence-BERT for NLI

## Dataset

- SNLI (Stanford Natural Language Inference)

## Architecture

A Siamese network structure:

- Premise and hypothesis are encoded separately.
- Shared BERT encoder weights.
- Produces fixed-size sentence embeddings.

---

## Sentence Embedding Strategy

We apply **mean pooling** over the last hidden states using the attention mask:

```
embedding = sum(token_embeddings * attention_mask) / sum(attention_mask)
```

This generates semantically meaningful sentence representations.

---

## Classification Objective (SoftmaxLoss)

Given:

- u = premise embedding  
- v = hypothesis embedding  

We concatenate:

- u  
- v  
- |u − v|  

Then apply:

```
o = softmax(W · [u, v, |u − v|])
```

Loss function used:

```
CrossEntropyLoss
```

---

## Model Checkpoint

Best performing model saved as:

```
models/sbert_task2_snli_best.pt
```

---

# 📊 Task 3 — Evaluation and Analysis

Evaluation performed on the SNLI validation/test split.

Metrics reported:

- Precision  
- Recall  
- F1-score  
- Support  
- Accuracy  

The classification report is included in:

```
st126438_Ashu_A4_Task2_BERT.ipynb

```

## Observations

- Cosine similarity reflects semantic similarity.
- The classification head determines the logical relationship between sentences.
- High cosine similarity does not always imply entailment.
- Performance limited by:
  - Dataset subset size  
  - Training epochs  
  - Computational constraints  

### SNLI Test Performance

| Class           | Precision | Recall | F1-score | Support |
|----------------|-----------|--------|----------|---------|
| Entailment     | 0.63      | 0.73   | 0.67     | 3486    |
| Neutral        | 0.62      | 0.58   | 0.60     | 3199    |
| Contradiction  | 0.65      | 0.59   | 0.62     | 3315    |
| **Accuracy**   |           |        | **0.63** | 10000   |


## Potential Improvements

- Larger training dataset  
- Longer training  
- Hyperparameter tuning  
- Full BERT pretraining corpus  
- Contrastive learning objectives  

---

# 🌐 Task 4 — Web Application

A web application was developed to demonstrate the model’s capabilities.

## Technologies Used

- FastAPI (Backend)  
- HTML + JavaScript (Frontend)  
- PyTorch (Inference)  
- Uvicorn (Server)  

---

## Web App Features

- Two input boxes:
  - Premise  
  - Hypothesis  
- Predict NLI label  
- Displays:
  - Predicted label  
  - Cosine similarity  
  - Probability distribution  
- Uses custom-trained Sentence-BERT model  

---

## Example (From Assignment)

**Premise:**
```
A man is playing a guitar on stage.
```

**Hypothesis:**
```
The man is performing music.
```

**Predicted Label:**
```
ENTAILMENT
```

**Cosine Similarity:** 0.73

```

This demonstrates that cosine similarity captures semantic closeness, 
while the classification layer determines the logical inference relationship.

```


---

# 🚀 How to Run

## 1️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 2️⃣ Run Backend

From project root:

```bash
uvicorn app.backend.main:app --reload
```

Access:

Swagger Docs:
```
http://127.0.0.1:8000/docs
```

Health Endpoint:
```
http://127.0.0.1:8000/health
```

---

## 3️⃣ Open Frontend

Open in browser:

```
app/frontend/index.html
```

Click **Predict NLI Label**.

---

# ✅ Assignment Completion Checklist

- [x] BERT implemented from scratch  
- [x] Trained on public dataset subset  
- [x] Sentence-BERT fine-tuned on SNLI  
- [x] Softmax classification objective implemented  
- [x] Evaluation metrics reported  
- [x] Web application developed  
- [x] Custom-trained model used in deployment  

---

**End of Assignment A4**

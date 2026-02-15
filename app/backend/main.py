import os
import re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# -----------------------------
# FastAPI setup
# -----------------------------
app = FastAPI(title="A4 SBERT NLI Demo")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ok for assignment demo
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Load checkpoint
# -----------------------------
CKPT_PATH = os.path.join("models", "sbert_task2_snli_best.pt")
assert os.path.exists(CKPT_PATH), f"Missing checkpoint at: {CKPT_PATH}"

ckpt = torch.load(CKPT_PATH, map_location="cpu")
cfg = ckpt["config"]
word2id = ckpt["vocab"]["word2id"]
# id2word not required for inference
SBERT_MAX_LEN = int(ckpt.get("sbert_max_len", min(cfg["max_len"], 64)))

# special ids
pad_id  = cfg["pad_id"]
cls_id  = cfg["cls_id"]
sep_id  = cfg["sep_id"]
unk_id  = cfg["unk_id"]

# model dims
n_layers   = cfg["n_layers"]
n_heads    = cfg["n_heads"]
d_model    = cfg["d_model"]
d_ff       = cfg["d_ff"]
d_k        = cfg["d_k"]
d_v        = cfg["d_v"]
max_len    = cfg["max_len"]
n_segments = cfg["n_segments"]
vocab_size = cfg["vocab_size"]

LABEL_NAMES = ["entailment", "neutral", "contradiction"]

# -----------------------------
# Model definitions (same as notebook)
# -----------------------------
class Embedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_len, d_model)
        self.seg_embed = nn.Embedding(n_segments, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, seg):
        seq_len = x.size(1)
        pos = torch.arange(seq_len, dtype=torch.long, device=x.device).unsqueeze(0).expand_as(x)
        emb = self.tok_embed(x) + self.pos_embed(pos) + self.seg_embed(seg)
        return self.norm(emb)

def get_attn_pad_mask(seq_q, seq_k):
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    pad_attn_mask = seq_k.data.eq(pad_id).unsqueeze(1)  # (bs, 1, len_k)
    return pad_attn_mask.expand(batch_size, len_q, len_k)

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, Q, K, V, attn_mask):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)
        scores.masked_fill_(attn_mask, -1e9)
        attn = torch.softmax(scores, dim=-1)
        context = torch.matmul(attn, V)
        return context, attn

class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_heads)
        self.W_K = nn.Linear(d_model, d_k * n_heads)
        self.W_V = nn.Linear(d_model, d_v * n_heads)
        self.fc = nn.Linear(n_heads * d_v, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, Q, K, V, attn_mask):
        residual, batch_size = Q, Q.size(0)

        q_s = self.W_Q(Q).view(batch_size, -1, n_heads, d_k).transpose(1, 2)
        k_s = self.W_K(K).view(batch_size, -1, n_heads, d_k).transpose(1, 2)
        v_s = self.W_V(V).view(batch_size, -1, n_heads, d_v).transpose(1, 2)

        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)

        context, attn = ScaledDotProductAttention()(q_s, k_s, v_s, attn_mask)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, n_heads * d_v)

        output = self.fc(context)
        return self.norm(output + residual), attn

class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))

class EncoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc_self_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, enc_inputs, enc_self_attn_mask):
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask)
        enc_outputs = self.pos_ffn(enc_outputs)
        return enc_outputs, attn

class BERT(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = Embedding()
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])

        # Heads exist in your training model; not used here for inference,
        # but kept to match state_dict compatibility.
        self.fc = nn.Linear(d_model, d_model)
        self.activ = nn.Tanh()
        self.linear = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, 2)

        embed_weight = self.embedding.tok_embed.weight
        n_vocab, n_dim = embed_weight.size()
        self.decoder = nn.Linear(n_dim, n_vocab, bias=False)
        self.decoder.weight = embed_weight
        self.decoder_bias = nn.Parameter(torch.zeros(n_vocab))

    def forward(self, input_ids, segment_ids, masked_pos=None, return_sequence_output=False):
        output = self.embedding(input_ids, segment_ids)
        enc_self_attn_mask = get_attn_pad_mask(input_ids, input_ids)
        for layer in self.layers:
            output, _attn = layer(output, enc_self_attn_mask)

        if return_sequence_output:
            attention_mask = (input_ids != pad_id).long()
            return output, attention_mask

        # below is MLM/NSP path (not used in webapp)
        if masked_pos is None:
            raise ValueError("masked_pos must be provided when return_sequence_output=False")

        h_pooled = self.activ(self.fc(output[:, 0]))
        logits_nsp = self.classifier(h_pooled)

        masked_pos = masked_pos[:, :, None].expand(-1, -1, output.size(-1))
        h_masked = torch.gather(output, 1, masked_pos)
        h_masked = self.norm(F.gelu(self.linear(h_masked)))
        logits_lm = self.decoder(h_masked) + self.decoder_bias
        return logits_lm, logits_nsp

def mean_pool(token_embeddings: torch.Tensor, attention_mask: torch.Tensor):
    mask = attention_mask.unsqueeze(-1).type_as(token_embeddings)  # [B,L,1]
    summed = (token_embeddings * mask).sum(dim=1)                  # [B,H]
    counts = mask.sum(dim=1).clamp(min=1e-9)                       # [B,1]
    return summed / counts

class SiameseSBERT(nn.Module):
    def __init__(self, bert_model: nn.Module, hidden_size: int, num_labels: int = 3):
        super().__init__()
        self.bert = bert_model
        self.classifier = nn.Linear(hidden_size * 3, num_labels)

    def encode(self, ids, seg, attn):
        seq_out, _mask = self.bert(ids, seg, return_sequence_output=True)
        return mean_pool(seq_out, attn)

    def forward(self, p_ids, p_seg, p_attn, h_ids, h_seg, h_attn):
        u = self.encode(p_ids, p_seg, p_attn)
        v = self.encode(h_ids, h_seg, h_attn)
        x = torch.cat([u, v, torch.abs(u - v)], dim=1)
        return self.classifier(x)

# -----------------------------
# Build models + load weights
# -----------------------------
bert = BERT()
bert.load_state_dict(ckpt["bert_state_dict"])
bert.to(device)
bert.eval()

sbert = SiameseSBERT(bert, hidden_size=d_model, num_labels=3)
sbert.load_state_dict(ckpt["sbert_state_dict"])
sbert.to(device)
sbert.eval()

# -----------------------------
# Tokenization / encoding (must match notebook behavior)
# -----------------------------
def simple_tokenize(text: str):
    # match your notebook tokenizer: words + punctuation tokens
    return re.findall(r"\w+|[^\w\s]", text.lower().strip())

def encode_sentence(text: str, max_seq_len: int):
    tokens = simple_tokenize(text)
    ids = [cls_id] + [word2id.get(t, unk_id) for t in tokens] + [sep_id]

    if len(ids) > max_seq_len:
        ids = ids[:max_seq_len]
        ids[-1] = sep_id

    attn = [1] * len(ids)
    seg  = [0] * len(ids)

    pad_len = max_seq_len - len(ids)
    if pad_len > 0:
        ids  += [pad_id] * pad_len
        attn += [0] * pad_len
        seg  += [0] * pad_len

    return (
        torch.tensor(ids, dtype=torch.long),
        torch.tensor(attn, dtype=torch.long),
        torch.tensor(seg, dtype=torch.long),
    )

def predict(premise: str, hypothesis: str):
    p_ids, p_attn, p_seg = encode_sentence(premise, SBERT_MAX_LEN)
    h_ids, h_attn, h_seg = encode_sentence(hypothesis, SBERT_MAX_LEN)

    p_ids  = p_ids.unsqueeze(0).to(device)
    p_attn = p_attn.unsqueeze(0).to(device)
    p_seg  = p_seg.unsqueeze(0).to(device)

    h_ids  = h_ids.unsqueeze(0).to(device)
    h_attn = h_attn.unsqueeze(0).to(device)
    h_seg  = h_seg.unsqueeze(0).to(device)

    with torch.no_grad():
        logits = sbert(p_ids, p_seg, p_attn, h_ids, h_seg, h_attn)
        probs_t = torch.softmax(logits, dim=1).squeeze(0)
        pred_id = int(torch.argmax(probs_t).item())

        u = sbert.encode(p_ids, p_seg, p_attn).squeeze(0)
        v = sbert.encode(h_ids, h_seg, h_attn).squeeze(0)
        cos = float(F.cosine_similarity(u.unsqueeze(0), v.unsqueeze(0)).item())

    probs = {LABEL_NAMES[i]: float(probs_t[i].cpu().item()) for i in range(3)}
    return LABEL_NAMES[pred_id], probs, cos

# -----------------------------
# API schema
# -----------------------------
class PredictRequest(BaseModel):
    premise: str
    hypothesis: str

@app.get("/health")
def health():
    return {"status": "ok", "device": str(device), "max_len": SBERT_MAX_LEN}

@app.post("/predict")
def predict_endpoint(req: PredictRequest):
    label, probs, cosine = predict(req.premise, req.hypothesis)
    return {"label": label, "probs": probs, "cosine_similarity": cosine}

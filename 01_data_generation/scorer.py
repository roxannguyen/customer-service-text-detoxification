import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from config import SCORER_MODEL, GENERATION_THRESHOLD, EASY_MAX, MEDIUM_MAX

_tokenizer = None
_model = None
_device = None

def get_scorer():
    global _tokenizer, _model, _device
    if _model is None:
        _device = "cuda" if torch.cuda.is_available() else "cpu"
        _tokenizer = AutoTokenizer.from_pretrained(SCORER_MODEL)
        _model = AutoModelForSequenceClassification.from_pretrained(SCORER_MODEL).to(_device)
        _model.eval()
    return _tokenizer, _model, _device

def score_text(text: str) -> float:
    """Returns toxicity probability in [0, 1]."""
    tokenizer, model, device = get_scorer()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
        prob = torch.sigmoid(logits[0][0]).item()
    return prob

def score_batch(texts: list) -> list:
    """Returns list of toxicity probabilities."""
    tokenizer, model, device = get_scorer()
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.sigmoid(logits[:, 0]).cpu().tolist()
    return probs

def get_band(score: float) -> str:
    """Maps continuous score to difficulty band."""
    if score < GENERATION_THRESHOLD:
        return "below_threshold"
    elif score < EASY_MAX:
        return "easy"
    elif score < MEDIUM_MAX:
        return "medium"
    else:
        return "hard"
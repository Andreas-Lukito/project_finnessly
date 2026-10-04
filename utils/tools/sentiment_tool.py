from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from crewai.tools import tool
import os

# Config
from pathlib import Path

PROJECT_PATH = Path(__file__).resolve().parents[2]

# Load Model Weights
## FINBERT Model
finbert_model_path = PROJECT_PATH / "model" / "finbert"
finbert_tokenizer = AutoTokenizer.from_pretrained(
    "ProsusAI/finbert"
)

finbert_model = AutoModelForSequenceClassification.from_pretrained(
    finbert_model_path
)

## FINBERT Model
bert_model_path = PROJECT_PATH / "model" / "bert"
bert_tokenizer = AutoTokenizer.from_pretrained(
    "google-bert/bert-base-uncased"
)

bert_model = AutoModelForSequenceClassification.from_pretrained(
    bert_model_path
)

# Create Sentiment Analysis Pipelines
finbert_pipeline = pipeline(
    "text-classification",
    model=finbert_model,
    tokenizer=finbert_tokenizer
)

bert_pipeline = pipeline(
    "text-classification",
    model=bert_model,
    tokenizer=bert_tokenizer
)

# Define Label Mapping
BERT_LABEL_MAP = {
    0: "positive",
    1: "negative",
    2: "neutral",
}

FINBERT_LABEL_MAP = {
    0: "positive",
    1: "negative",
    2: "neutral",
}

LABEL_MAPS = {
    "finbert": FINBERT_LABEL_MAP,
    "bert": BERT_LABEL_MAP,
}

# Define the Sentiment Analysis Tool
def analyze_sentiment(text: str, model_type: str = "finbert") -> dict:

    model_type = model_type.lower().strip()

    if model_type == "finbert":
        result = finbert_pipeline(text)[0]

        return {
            "label": result["label"].lower(),
            "score": result["score"]
        }

    elif model_type == "bert":
        result = bert_pipeline(text)[0]

        label_id = int(result["label"].split("_")[-1])

        return {
            "label": BERT_LABEL_MAP[label_id],
            "score": result["score"]
        }

    else:
        raise ValueError(
            "Invalid model_type. Choose 'finbert' or 'bert'."
        )

# Initial Test
if __name__ == "__main__":
    print("BERT:", bert_model.config.id2label)
    print("FinBERT:", finbert_model.config.id2label)
    # print(finbert_pipeline("NVIDIA reported record revenue."))
    # print(bert_pipeline("NVIDIA reported record revenue."))
    print(analyze_sentiment("NVIDIA reported record revenue.", model_type="finbert"))
    print(analyze_sentiment("NVIDIA reported record revenue.", model_type="bert"))
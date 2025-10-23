import logging
import nltk
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

nltk.download('punkt')

class Summarizer:
    def __init__(self, model_path, tokenizer_path):
        logger.info(f"Initializing Summarizer: model={model_path}, tokenizer={tokenizer_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def model_summarize(self, text, max_length, min_length, **gen_kwargs):
        inputs = self.tokenizer("summarize: " + text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        output_ids = self.model.generate(
            **inputs,
            max_length=max_length,
            min_length=min_length,
            **gen_kwargs,
            early_stopping=True
        )
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

    def split_into_sentences(self, text):
        return nltk.sent_tokenize(text)

    def chunk_sentences(self, sentences, max_words=300, min_words=50):
        chunks, current, count = [], [], 0
        for sent in sentences:
            words = len(sent.split())
            if count + words <= max_words:
                current.append(sent); count += words
            else:
                if count >= min_words:
                    chunks.append(" ".join(current))
                current, count = [sent], words
        if count >= min_words:
            chunks.append(" ".join(current))
        return chunks

    def recursive_summarize(self, text, max_length, min_length, threshold):
        if len(text.split()) <= threshold:
            return text

        sentences = self.split_into_sentences(text)
        chunks = self.chunk_sentences(sentences)
        summaries = [
            self.model_summarize(chunk, max_length=max_length, min_length=min_length)
            for chunk in chunks
        ]
        combined = " ".join(summaries)
        if len(nltk.sent_tokenize(combined)) == 1 or len(combined.split()) <= threshold:
            return combined
        return self.recursive_summarize(combined, max_length, min_length, threshold)

    def iterative_summarization(self, text, max_length=150, min_length=30):
        logger.info(f"Summarizing to between {min_length}–{max_length} words")
        return self.recursive_summarize(text, max_length, min_length, threshold=min_length)

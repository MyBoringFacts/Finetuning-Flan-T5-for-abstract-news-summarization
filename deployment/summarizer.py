import logging
import nltk
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Configure logger to print to console
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Download necessary NLTK data (if not already downloaded)
nltk.download('punkt')

class Summarizer:
    def __init__(self, model_path, tokenizer_path):
        """
        Initialize the summarizer with a fine-tuned model and tokenizer.
        Both model and tokenizer are loaded from the same directory.
        """
        logger.info(f"Initializing Summarizer with model_path: {model_path} and tokenizer_path: {tokenizer_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        self.tokenizer_path = tokenizer_path
        
        # Set device to GPU if available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        logger.info(f"Model loaded on device: {self.device}")
    
    def model_summarize(self, text_chunk, 
                        max_length=200, 
                        min_length=30,
                        num_beams=4, 
                        temperature=0.3, 
                        top_k=50, 
                        top_p=0.95):
        """
        Summarizes a text chunk using the fine-tuned model.
        The prompt instructs the model to include explicit noun references.
        """
        logger.info(f"Summarizing text chunk of {len(text_chunk.split())} words.")
        
        # Re-load tokenizer from the given path (as in original code)
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
        input_text = "summarize : " + text_chunk
        inputs = self.tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_length=max_length,
                min_length=30,
                num_beams=num_beams,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                do_sample=True,
                early_stopping=True
            )
        
        summary = self.tokenizer.decode(output_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
        logger.info("Summary generated.")
        return summary

    def split_into_sentences(self, text):
        """
        Splits the text into sentences using NLTK.
        """
        sentences = nltk.sent_tokenize(text)
        logger.info(f"Text split into {len(sentences)} sentences.")
        return sentences
    
    def chunk_sentences(self, sentences):
        """
        Groups sentences into chunks.
        Each chunk contains as many sentences as possible while keeping its total word count below 300.
        Only chunks with at least 50 words are kept; chunks with fewer words are discarded.
        """
        logger.info("Starting sentence chunking.")
        chunks = []
        current_chunk = []
        current_word_count = 0

        for sentence in sentences:
            sentence_word_count = len(sentence.split())
            
            # If adding this sentence keeps the chunk under 300 words, add it.
            if current_word_count + sentence_word_count <= 300:
                current_chunk.append(sentence)
                current_word_count += sentence_word_count
            else:
                # If current chunk meets the minimum word requirement, add it to the chunks list.
                if current_word_count >= 75:
                    chunks.append(" ".join(current_chunk))
                    logger.info(f"Created a chunk with {current_word_count} words.")
                # Start a new chunk with the current sentence.
                current_chunk = [sentence]
                current_word_count = sentence_word_count

        # After the loop, add the last chunk if it meets the minimum requirement.
        if current_word_count >= 75:
            chunks.append(" ".join(current_chunk))
            logger.info(f"Final chunk created with {current_word_count} words.")
        
        logger.info(f"Total chunks created: {len(chunks)}")
        return chunks
    
    def recursive_summarize(self, text, threshold=50):
        """
        Recursively summarizes the text until its word count is below the threshold.
        If the combined summary consists of a single sentence (even if its length is above the threshold),
        the recursion stops.
        """
        logger.info(f"Recursive summarization called on text with {len(text.split())} words.")
        if len(text.split()) <= threshold:
            logger.info("Text is below the threshold; returning original text.")
            return text

        sentences = self.split_into_sentences(text)
        if not sentences:
            logger.warning("No sentences found; returning original text.")
            return text  # Edge case if sentence splitting fails

        chunks = self.chunk_sentences(sentences)
        logger.info("Generating summaries for each chunk.")
        summaries = [self.model_summarize(chunk) for chunk in chunks]
        combined_summary = " ".join(summaries)
        logger.info(f"Combined summary length: {len(combined_summary.split())} words.")

        # Check if the combined summary is a single sentence; if so, stop recursion.
        summary_sentences = self.split_into_sentences(combined_summary)
        if len(summary_sentences) == 1:
            logger.info("Combined summary consists of a single sentence; returning summary without further recursion.")
            return combined_summary

        if len(combined_summary.split()) > threshold:
            logger.info("Combined summary exceeds threshold; recursing further.")
            return self.recursive_summarize(combined_summary, threshold)
        else:
            logger.info("Combined summary meets threshold; summarization complete.")
            return combined_summary

    def iterative_summarization(self, text, threshold=75):
        """
        Alias for recursive_summarize to maintain compatibility with fetch_top_news.py.
        """
        logger.info("Starting iterative summarization.")
        return self.recursive_summarize(text, threshold)

# if __name__ == "__main__":
#     # Example test block to verify functionality.
#     text = """Your test text here."""
#     summarizer = Summarizer("beta./model", "beta./model")
#     final_summary = summarizer.iterative_summarization(text, threshold=50)
#     print(final_summary)

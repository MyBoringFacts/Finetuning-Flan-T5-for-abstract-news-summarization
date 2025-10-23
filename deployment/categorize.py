from transformers import AutoModel, AutoTokenizer
import torch
import joblib
import os
from dotenv import load_dotenv

class NewsCategorizer:
    def __init__(self, model_name, classifier_path, label_encoder_path):
        """
        Initialize the NewsCategorizer with a transformer model and a pre-trained SVM classifier.
        
        Args:
            model_name (str): Name of the Hugging Face model for sentence embeddings.
            classifier_path (str): File path to the saved SVM classifier.
            label_encoder_path (str): File path to the saved label encoder.
        """
        # Load environment variables in case they're needed for paths.
        load_dotenv()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Load the transformer model and tokenizer.
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        # Load the pre-trained SVM classifier and label encoder.
        self.clf = joblib.load(classifier_path)
        self.label_encoder = joblib.load(label_encoder_path)

    def compute_embedding(self, text):
        """
        Compute the sentence embedding for the given text using the transformer model.
        
        Args:
            text (str): The input text to encode.
        
        Returns:
            np.ndarray: The computed embedding as a NumPy array.
        """
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            padding=True, 
            max_length=512
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Average the last hidden states to obtain a fixed-length sentence embedding.
            embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        return embedding

    def predict(self, text):
        """
        Predict the category of the given text.
        
        Args:
            text (str): The text (or summary) of the news article.
        
        Returns:
            str: The predicted news category.
        """
        embedding = self.compute_embedding(text)
        prediction = self.clf.predict(embedding)
        return self.label_encoder.inverse_transform(prediction)[0]

# import streamlit as st
# import re
# import nltk
# nltk.download('punkt')  # Ensure sentence tokenizers are available.
# from summarizer import Summarizer
# from categorize import NewsCategorizer

# # Function to clean text
# def clean_text(text):
#     """
#     Cleans the text by removing extra whitespace.
#     You can expand this function to include additional cleaning as needed.
#     """
#     cleaned = re.sub(r'\s+', ' ', text).strip()
#     return cleaned

# # Initialize summarizer and categorizer
# MODEL_PATH = r"server_side_summarize_news"  # Replace with your model path
# TOKENIZER_PATH = r"server_side_summarize_news"  # Replace with your tokenizer path
# summarizer = Summarizer(MODEL_PATH, TOKENIZER_PATH)

# MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
# CLASSIFIER_PATH ="server_side_categorize_news/svm_model.pkl" # Replace with your classifier path
# LABEL_ENCODER_PATH = "server_side_categorize_news/label_encoder.pkl"  # Replace with your label encoder path
# categorizer = NewsCategorizer(MODEL_NAME, CLASSIFIER_PATH, LABEL_ENCODER_PATH)

# # Set up the Streamlit UI
# st.title("Text Summarizer & Categorizer")
# st.write("Enter your text below to generate a summary and categorize it:")

# # Text area for user input
# input_text = st.text_area("Input Text", height=200)

# # When the 'Process' button is pressed, summarize and categorize the text
# if st.button("Summarize & Categorize"):
#     if not input_text.strip():
#         st.warning("Please enter some text to process.")
#     else:
#         # Clean the input text
#         cleaned_text = clean_text(input_text)
#         try:
#             # Generate the summary
#             summary = summarizer.iterative_summarization(cleaned_text)
#             st.subheader("Summary")
#             st.write(summary)

#             # Categorize the summary
#             category = categorizer.predict(summary)
#             st.subheader("Category")
#             st.write(category)
#         except Exception as e:
#             st.error(f"An error occurred: {e}")
import streamlit as st
import re
import nltk
nltk.download('punkt')  # Ensure sentence tokenizers are available.
from summarizer import Summarizer
from categorize import NewsCategorizer

# Basic cleaning function
def clean_text(text):
    """
    Cleans the text by removing extra whitespace.
    """
    cleaned = re.sub(r'\s+', ' ', text).strip()
    return cleaned

# Enhanced cleaning function (preprocessing)
def clean_junk_text(text):
    """
    Cleans the text so that the final result is well-formed English.
    Steps include:
    - Removing timestamps, media markers, subscription messages, links, emojis, HTML entities, and extra dots.
    - Normalizing whitespace.
    - Tokenizing and filtering tokens to keep only English words (and punctuation).
    - Reassembling and capitalizing sentences.
    """
    # Remove timestamps like "22:14 (UTC +04:00)"
    text = re.sub(r'\d{1,2}:\d{2}\s*\(UTC[^\)]+\)', '', text)
    
    # Remove media markers like (PHOTO) or (PHOTO/VIDEO)
    text = re.sub(r'\(PHOTO(?:/VIDEO)?\)', '', text, flags=re.IGNORECASE)
    
    # Remove subscription/access messages (customizable)
    text = re.sub(r'Access to paid information is limited.*?Subscription to paid content', '', text, flags=re.DOTALL)
    
    # Remove links (URLs)
    text = re.sub(r'http[s]?://\S+', '', text)
    
    # Remove emojis using unicode ranges
    emoji_pattern = re.compile("[" 
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)
    
    # Remove HTML encoded characters
    text = re.sub(r'&[#A-Za-z0-9]+;', '', text)
    
    # Replace multiple dots with a single dot
    text = re.sub(r'\.{2,}', '.', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Tokenize text to preserve punctuation.
    tokens = nltk.word_tokenize(text)
    allowed_tokens = []
    for token in tokens:
        # Allow only English words (with possible apostrophes) or common punctuation.
        if re.fullmatch(r"[a-zA-Z]+(?:'[a-zA-Z]+)?", token):
            allowed_tokens.append(token)
        elif re.fullmatch(r"[.,!?;:'\"()-]+", token):
            allowed_tokens.append(token)
    cleaned_text = " ".join(allowed_tokens)
    
    # Remove space before punctuation.
    cleaned_text = re.sub(r'\s+([.,!?;:\)])', r'\1', cleaned_text)
    
    # Sentence tokenization and capitalize the first letter of each sentence.
    sentences = nltk.sent_tokenize(cleaned_text)
    sentences = [s.capitalize() for s in sentences]
    cleaned_text = " ".join(sentences)
    
    return cleaned_text

# Initialize summarizer and categorizer
MODEL_PATH = r"server_side_summarize_news"  # Replace with your model path
TOKENIZER_PATH = r"server_side_summarize_news"  # Replace with your tokenizer path
summarizer = Summarizer(MODEL_PATH, TOKENIZER_PATH)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CLASSIFIER_PATH = "server_side_categorize_news/svm_model.pkl"  # Replace with your classifier path
LABEL_ENCODER_PATH = "server_side_categorize_news/label_encoder.pkl"  # Replace with your label encoder path
categorizer = NewsCategorizer(MODEL_NAME, CLASSIFIER_PATH, LABEL_ENCODER_PATH)

# Set up the Streamlit UI
st.title("Text Summarizer & Categorizer")
st.write("Enter your text below to generate a summary and categorize it using enhanced preprocessing.")

# Text area for user input
input_text = st.text_area("Input Text", height=200)

# Checkbox to trigger comparison with basic cleaning
compare_mode = st.checkbox("Compare with Basic Cleaning", value=False)

if st.button("Summarize & Categorize"):
    if not input_text.strip():
        st.warning("Please enter some text to process.")
    else:
        # Enhanced processing by default
        preprocessed_text = clean_junk_text(input_text)
        try:
            summary_preprocessed = summarizer.iterative_summarization(preprocessed_text)
            category_preprocessed = categorizer.predict(summary_preprocessed)
        except Exception as e:
            st.error(f"An error occurred in enhanced preprocessing: {e}")
            summary_preprocessed = "Error generating summary."
            category_preprocessed = "N/A"
        
        if not compare_mode:
            # Display enhanced processing results only
            st.subheader("Enhanced Preprocessing")
            # st.write("**Cleaned Text:**")
            # st.write(preprocessed_text)
            st.subheader("Summary")
            st.write(summary_preprocessed)
            st.subheader("Category")
            st.write(category_preprocessed)
        else:
            # Also process with basic cleaning for comparison
            basic_cleaned_text = clean_text(input_text)
            try:
                summary_basic = summarizer.iterative_summarization(basic_cleaned_text)
                category_basic = categorizer.predict(summary_basic)
            except Exception as e:
                st.error(f"An error occurred in basic processing: {e}")
                summary_basic = "Error generating summary."
                category_basic = "N/A"
            
            # Display results side by side for comparison
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Enhanced Preprocessing")
                # st.write("**Cleaned Text:**")
                # st.write(preprocessed_text)
                st.subheader("Summary")
                st.write(summary_preprocessed)
                st.subheader("Category")
                st.write(category_preprocessed)
            with col2:
                st.subheader("Basic Cleaning")
                # st.write("**Cleaned Text:**")
                # st.write(basic_cleaned_text)
                st.subheader("Summary")
                st.write(summary_basic)
                st.subheader("Category")
                st.write(category_basic)

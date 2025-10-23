import os
import re
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv, find_dotenv
import firebase_admin
from firebase_admin import credentials, firestore
import nltk
nltk.download('punkt')  # Ensure sentence and word tokenizers are available.
# Import your summarizer and categorizer classes from their modules.
from summarizer import Summarizer
from categorize import NewsCategorizer

def initialize_environment():
    """
    Loads environment variables from the .env file.
    """
    dotenv_path = find_dotenv()
    load_dotenv(dotenv_path)

def initialize_firebase():
    """
    Initializes Firebase using credentials.
    Returns a Firestore client.
    """
    load_dotenv()
    cred_path = os.getenv("cred_firebase")
    if not firebase_admin._apps:
        cred = credentials.Certificate(cred_path)
        firebase_admin.initialize_app(cred)
    return firestore.client()

def get_current_date():
    """
    Returns the current date in 'YYYY-MM-DD' format.
    """
    return datetime.now().strftime("%Y-%m-%d")

def fetch_top_news(api_key, language="en", date=None, number=30):
    """
    Fetches a few top news articles using the News API.
    """
    date = date or get_current_date()
    base_url = "https://newsapi.org/v2/everything"
    params = {
        "q": "sport",  # Fetch general world news
        "language": language,
        "from": date,
        "sortBy": "popular",
        "pageSize": number,
        "apiKey": api_key
    }
    #breaking news for world categories

    print(f"Making request with parameters: {params}")
    response = requests.get(base_url, params=params)
    print(f"Response status code: {response.status_code}")

    if response.status_code == 200:
        response_json = response.json()
        print(f"Total results: {response_json.get('totalResults', 0)}")
        articles = response_json.get("articles", [])
        if not articles:
            print("No articles returned in the response")
        return [
            {
                "Date": date,
                "title": article.get("title", "No Title"),
                "text": article.get("description", "No Content"),
                "url": article.get("url", "No URL"),
            }
            for article in articles
        ]
    else:
        print(f"Error fetching news: {response.status_code}")
        print(f"Response text: {response.text}")
        return []


def fetch_top_news_world_news(api_key, language="en", date=None, number=20):
    """
    Fetches top news articles using the World News API.
    - Sets the date parameter to the current date if none is provided.
    - Constructs the base URL and parameters for the API call.
    - Sends a GET request to the API.
    - If the request is successful (status code 200), extracts the list of news articles.
    - For each article, constructs a dictionary containing the date, title, text, and URL.
    - Returns a list of these article dictionaries; otherwise, prints an error and returns an empty list.
    """
    date = date   or get_current_date()
    base_url = "https://api.worldnewsapi.com/search-news?text=deepseek-ai&earliest-published-date=2025-02-1&latest-published-date=2025-02-27"
    params = {
        'api-key': api_key,
        'language': language,
        'published_date': date,
        'sort-by': 'relevance',
        'number': number
    }
    response = requests.get(base_url, params=params)
    if response.status_code == 200:
        articles = response.json().get("news", [])
        return [
            {
                "Date": date,
                "title": article.get("title", "No Title"),
                "text": article.get("text", "No Text"),
                "url": article.get("url", "No URL"),
            }
            for article in articles
        ]
    else:
        print(f"Error fetching news: {response.status_code}, {response.text}")
        return []

def clean_junk_text(text):
    """
    Cleans the text so that the final result is a well-formed English text.
    
    Steps:
    - Removes timestamps, media markers, subscription messages, links, emojis, HTML entities,
      extra dots, and normalizes whitespace.
    - Tokenizes text into words and punctuation while filtering out tokens that are not:
         (a) pure English words (allowing for contractions), or 
         (b) common punctuation.
    - Reassembles the text with proper spacing.
    - Uses sentence tokenization to capitalize the first letter of each sentence.
    
    This helps ensure the summarizer and categorizer (LLMs) receive proper, grammatical English.
    """
    # Remove timestamps like "22:14 (UTC +04:00)"
    text = re.sub(r'\d{1,2}:\d{2}\s*\(UTC[^\)]+\)', '', text)
    
    # Remove media markers like (PHOTO) or (PHOTO/VIDEO)
    text = re.sub(r'\(PHOTO(?:/VIDEO)?\)', '', text, flags=re.IGNORECASE)
    
    # Remove subscription/access messages (customize this pattern as needed)
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
    
    # Remove HTML encoded characters (e.g., &#039;)
    text = re.sub(r'&[#A-Za-z0-9]+;', '', text)
    
    # Replace multiple dots with a single dot
    text = re.sub(r'\.{2,}', '.', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Tokenize text using nltk to preserve punctuation.
    tokens = nltk.word_tokenize(text)
    allowed_tokens = []
    for token in tokens:
        # Allow tokens that are:
        #   a) English words (allowing for apostrophes in contractions)
        #   b) punctuation marks
        if re.fullmatch(r"[a-zA-Z]+(?:'[a-zA-Z]+)?", token):
            allowed_tokens.append(token)
        elif re.fullmatch(r"[.,!?;:'\"()-]+", token):
            allowed_tokens.append(token)
        # Skip any token that doesn't match these patterns.
    
    # Reassemble the text.
    cleaned_text = " ".join(allowed_tokens)
    
    # Fix spacing: remove space before punctuation.
    cleaned_text = re.sub(r'\s+([.,!?;:\)])', r'\1', cleaned_text)
    
    # Use sentence tokenization and capitalize the first letter of each sentence.
    sentences = nltk.sent_tokenize(cleaned_text)
    sentences = [s.capitalize() for s in sentences]
    cleaned_text = " ".join(sentences)
    
    return cleaned_text

def is_strictly_english(text, threshold=0.9):
    """
    Determines whether the text is strictly English by calculating the ratio of words 
    that contain only A-Z letters to the total number of words.
    """
    words = text.split()
    if not words:
        return False
    english_words = re.findall(r'\b[a-zA-Z]+\b', text)
    ratio = len(english_words) / len(words)
    return ratio >= threshold

# def process_articles(articles, summarizer, categorizer, seen_titles, seen_texts, seen_urls):
#     """
#     Processes articles by verifying they are strictly English, cleaning the text,
#     summarizing, and categorizing. Additionally, it filters out redundant articles
#     by checking if the title, cleaned text, or URL has already been seen in this run.
#     """
#     processed_articles = []
#     for article in articles:
#         title = article["title"].strip()
#         url = article["url"].strip()
#         title_lower = title.lower()
#         url_lower = url.lower()

#         # Clean the article text using our enhanced cleaning function.
#         cleaned_text = clean_junk_text(article["text"])
#         text_lower = cleaned_text.lower()

#         # Check if any of these have been seen before.
#         if title_lower in seen_titles or text_lower in seen_texts or url_lower in seen_urls:
#             print("Skipping redundant article:", title)
#             continue

#         # Update the seen sets.
#         seen_titles.add(title_lower)
#         seen_texts.add(text_lower)
#         seen_urls.add(url_lower)

#         # Update the article with cleaned text.
#         article["text"] = cleaned_text
        
#         # Verify that the article text is strictly English.
#         if not is_strictly_english(article["text"]):
#             print("Skipping article due to insufficient English content:", title)
#             continue

#         # Summarize the article.
#         summary = summarizer.iterative_summarization(article["text"])
#         article["summary"] = summary

#         # Categorize the summary.
#         # category = categorizer.predict(summary)
#         combined_text = article["title"] + " " + summary
#         category = categorizer.predict(combined_text)
#         article["category"] = category

#         processed_articles.append(article)
#     return processed_articles

def process_articles(articles, summarizer, categorizer, seen_titles, seen_texts):
    """
    Processes articles by cleaning, summarizing, and categorizing.
    An article is considered redundant and skipped only if both its title and cleaned text
    exactly match an article processed earlier.
    """
    processed_articles = []
    for article in articles:
        title = article["title"].strip()
        title_lower = title.lower()

        # Clean the article text.
        cleaned_text = clean_junk_text(article["text"])
        if not cleaned_text or not cleaned_text.strip():
            print("Skipping article due to missing cleaned text:", title)
            continue
        text_lower = cleaned_text.lower()

        # Skip the article only if both title and text have been seen before.
        if title_lower in seen_titles and text_lower in seen_texts:
            print("Skipping redundant article:", title)
            continue

        # Update the seen sets.
        seen_titles.add(title_lower)
        seen_texts.add(text_lower)

        # Update the article with cleaned text.
        article["text"] = cleaned_text

        # Verify that the article text is strictly English.
        if not is_strictly_english(cleaned_text):
            print("Skipping article due to insufficient English content:", title)
            continue

        # Generate a summary.
        summary = summarizer.iterative_summarization(cleaned_text)
        if not summary or not summary.strip():
            print("Skipping article due to missing summary:", title)
            continue
        article["summary"] = summary

        # Categorize using a combination of title and summary.
        combined_text = title + " " + summary
        category = categorizer.predict(combined_text)
        article["category"] = category

        processed_articles.append(article)
    return processed_articles


def upload_to_firestore(news_list, db_client):
    """
    Uploads a list of processed news articles to Firestore using a batch write.
    Only uploads articles that have both non-empty 'text' and 'summary' fields.
    """
    if not news_list:
        print("[Info] No news articles to upload.")
        return

    collection_ref = db_client.collection("ProcessedNews")
    batch = db_client.batch()
    for article in news_list:
        # Final check: ensure article contains non-empty text and summary.
        if not article.get("text", "").strip() or not article.get("summary", "").strip():
            print("Skipping upload for article due to missing text or summary:", article.get("title"))
            continue
        doc_ref = collection_ref.document()
        batch.set(doc_ref, article)
    batch.commit()
    print("[Info] News articles have been successfully uploaded to Firestore.")


def main():
    # Initialize environment variables and Firebase.
    initialize_environment()
    db_client = initialize_firebase()
    
    # Your API key for the news API.
    api_key = os.getenv("WORLD_NEWS_API")
    
    # Set the date range: from Feb 16, 2025 to Feb 10, 2025.
    start_date = datetime.strptime("2025-02-27", "%Y-%m-%d") #start date is today
    end_date = datetime.strptime("2025-2-1", "%Y-%m-%d") # end date is a day in the past. 
    # this is regression
    
    # Instantiate the summarizer and categorizer.
    MODEL_PATH = os.getenv("SUMMARIZER_MODEL_PATH")
    TOKENIZER_PATH = os.getenv("SUMMARIZER_TOKENIZER_PATH")
    summarizer = Summarizer(MODEL_PATH, TOKENIZER_PATH)
    
    MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    CLASSIFIER_PATH = os.getenv("CLASSIFIER_PATH")
    LABEL_ENCODER_PATH = os.getenv("LABEL_ENCODER_PATH")
    categorizer = NewsCategorizer(MODEL_NAME, CLASSIFIER_PATH, LABEL_ENCODER_PATH)
    
    # Global sets to track seen titles, texts, and URLs for the full run.
    seen_titles = set()
    seen_texts = set()
    seen_urls = set()
    
    # Process news for each day in the specified range.
    current_date = start_date
    while current_date >= end_date:
        date_str = current_date.strftime("%Y-%m-%d")
        print(f"Fetching news for {date_str}")
        # daily_articles = fetch_top_news_world_news(api_key=api_key, date=date_str, number=5)
        daily_articles = fetch_top_news(api_key=api_key, date=date_str, number=50)
        
        
        # Process the day's articles using the global seen sets.
        processed_articles = process_articles(
            daily_articles,
            summarizer,
            categorizer,
            seen_titles,
            seen_texts
        )
        
        # Upload the processed articles for the current day to Firestore.
        upload_to_firestore(processed_articles, db_client)
        
        print("\n" * 3)
        print("Current date:", current_date)
        current_date -= timedelta(days=1)

if __name__ == "__main__":
    main()

import os
import firebase_admin
from firebase_admin import credentials, firestore
from dotenv import load_dotenv, find_dotenv
from datetime import date, datetime
import logging
import json
import streamlit as st

# Set up logging
logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)

# --------------- Firebase Initialization Functions ---------------
def initialize_environment():
    dotenv_path = find_dotenv()
    load_dotenv(dotenv_path)
    logging.info(f"Environment variables loaded from {dotenv_path}")

def initialize_firebase():
    if not firebase_admin._apps:
        data = json.loads(os.getenv("firebase_detailed_cred"))
        cred = credentials.Certificate(data)
        firebase_admin.initialize_app(cred)
        logging.info("Firebase initialized using provided credentials.")
    else:
        logging.info("Firebase already initialized.")
    return firestore.client()

# --------------- Firestore Query Function ---------------
from firebase_admin.firestore import FieldFilter

def fetch_news_firestore(start_date: str, end_date: str, categories, keyword, db_client):
    logging.info(f"Querying Firestore for articles from {start_date} to {end_date}.")
    try:
        start_date_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_date_dt = datetime.strptime(end_date, '%Y-%m-%d')
    except ValueError as e:
        logging.error(f"Invalid date format: {e}. Use YYYY-MM-DD.")
        return []

    start_date_str = start_date_dt.strftime('%Y-%m-%d')
    end_date_str = end_date_dt.strftime('%Y-%m-%d')

    query = db_client.collection("ProcessedNews")
    query = query.where(filter=FieldFilter("Date", ">=", start_date_str))
    query = query.where(filter=FieldFilter("Date", "<=", end_date_str))
    query = query.limit(100)

    if categories:
        logging.info(f"Filtering by categories: {categories}")
        query = query.where(filter=FieldFilter("category", "in", categories))

    docs = query.stream()
    results = [doc.to_dict() for doc in docs]
    logging.info(f"Retrieved {len(results)} articles before keyword filtering.")

    if keyword:
        kw = keyword.lower()
        results = [
            article for article in results
            if kw in article.get("title", "").lower() or kw in article.get("summary", "").lower()
        ]
        logging.info(f"{len(results)} articles remain after keyword filtering.")

    unique_results = []
    seen_titles = set()
    for article in results:
        title = article.get("title", "").strip()
        if title and title not in seen_titles:
            seen_titles.add(title)
            unique_results.append(article)
    logging.info(f"{len(unique_results)} unique articles remain after duplicate removal.")
    
    return unique_results

# --------------- Streamlit UI & Styles ---------------
def local_css(css_text):
    st.markdown(f'<style>{css_text}</style>', unsafe_allow_html=True)

light_mode_styles = """
/* Global Styles for Light Mode */
body, .stApp {
    background-color: #f8f9fa !important;
    color: #212529 !important;
    font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
}

/* News Cards */
.news-container {
    display: flex;
    flex-direction: column;
    gap: 2.5rem;
    margin-bottom: 2rem;
}
.news-card {
    background-color: #fff;
    color: #212529;
    border-radius: 8px;
    border-left: 4px solid #FF7E5F;
    box-shadow: 0 4px 10px rgba(0,0,0,0.1);
    padding: 1.5rem;
    margin-bottom: 1rem;
    position: relative;
    transition: transform 0.3s ease;
}
.news-card:hover {
    transform: translateY(-5px);
}
.news-card .meta {
    font-size: 0.9rem;
    color: #6c757d;
    margin-bottom: 0.8rem;
}

/* Modal Overlay for Spotlight */
.modal-overlay {
    position: fixed;
    top: 0; left: 0; width: 100%; height: 100%;
    background: rgba(0,0,0,0.5);
    display: flex;
    justify-content: center;
    align-items: center;
    z-index: 1000;
}
.modal-card {
    background: #fff;
    padding: 2rem;
    border-radius: 8px;
    max-width: 600px;
    width: 90%;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
}
.close-btn {
    margin-top: 1rem;
    padding: 0.5rem 1rem;
    background: #FF7E5F;
    color: #fff;
    border: none;
    border-radius: 4px;
    cursor: pointer;
}
"""

# --------------- Main App ---------------
def main():
    st.set_page_config(page_title="Modern News Feed", layout="centered")
    st.sidebar.title("Refine Your Feed")
    local_css(light_mode_styles)

    # Initialize Firebase and environment variables only once.
    if "db_client" not in st.session_state:
        initialize_environment()
        st.session_state.db_client = initialize_firebase()

    db_client = st.session_state.db_client

    # Initialize session state for selected article if not set
    if "selected_article" not in st.session_state:
        st.session_state.selected_article = None

    # Sidebar filters.
    today_date = date.today()
    default_start = today_date.replace(day=1)
    default_end = today_date

    date_range = st.sidebar.date_input("Select Date Range", [default_start, default_end])
    if len(date_range) != 2:
        st.sidebar.error("Please select both a start and an end date.")
        return

    start_date_obj, end_date_obj = date_range
    start_date_str = start_date_obj.strftime("%Y-%m-%d")
    end_date_str = end_date_obj.strftime("%Y-%m-%d")

    categories_list = ["World", "Sports", "Business", "Sci/Tech", "Politics", "Entertainment", "Others"]
    selected_categories = st.sidebar.multiselect("Select Categories", options=categories_list)
    keyword = st.sidebar.text_input("Search Keywords")

    col1, col2 = st.sidebar.columns(2)
    with col1:
        get_news = st.button("Get News")
    with col2:
        clear_news = st.button("Clear News")

    if get_news:
        st.session_state.news_pressed = True
        with st.spinner("Fetching articles..."):
            articles = fetch_news_firestore(start_date_str, end_date_str, selected_categories, keyword, db_client)
        st.session_state.articles = articles
        logging.info(f"{len(articles)} articles stored after filtering.")

    if clear_news:
        st.session_state.pop("articles", None)
        st.session_state.pop("news_pressed", None)
        st.session_state.selected_article = None

    st.markdown("<div class='container'>", unsafe_allow_html=True)

    # If an article is selected, render the spotlight modal.
    if st.session_state.selected_article:
        article = st.session_state.selected_article
        modal_html = f"""
        <div class="modal-overlay">
            <div class="modal-card">
                <h2>{article.get("title", "No Title")}</h2>
                <p><strong>Date:</strong> {article.get("Date", "Unknown")}</p>
                <p><strong>Category:</strong> {article.get("category", "Uncategorized")}</p>
                <p>{article.get("summary", "No Summary Available")}</p>
                {'<a href="' + article.get("url", "#") + '" target="_blank">Read More</a>' if article.get("url") else ""}
                <br>
                <button class="close-btn" onclick="window.location.reload()">Close Spotlight</button>
            </div>
        </div>
        """
        st.markdown(modal_html, unsafe_allow_html=True)
    else:
        # Render normal view: either news feed or hero section.
        if st.session_state.get("news_pressed", False):
            articles_to_show = st.session_state.get("articles", [])
            if articles_to_show:
                st.markdown("## Explore the Stories That Matter")
                st.success(f"Found {len(articles_to_show)} articles")
                st.markdown("<div class='news-container'>", unsafe_allow_html=True)
                for idx, article in enumerate(articles_to_show):
                    with st.container():
                        st.markdown(
                            f"""
                            <div class="news-card">
                                <h2>{article.get("title", "No Title")}</h2>
                                <div class="meta">
                                    <strong>Date:</strong> {article.get("Date", "Unknown")} &nbsp;&nbsp; 
                                    <strong>Category:</strong> {article.get("category", "Uncategorized")}
                                </div>
                                <p>{article.get("summary", "No Summary Available")}</p>
                                {'<a href="' + article.get("url", "#") + '" target="_blank">Read More</a>' if article.get("url") else ""}
                            </div>
                            """, unsafe_allow_html=True
                        )
                        if st.button("View Details", key=f"view_{idx}"):
                            st.session_state.selected_article = article
                            st.experimental_rerun()  # A normal rerun (not experimental query param) to refresh the view.
                st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.info("No news meet the requirement. Please adjust your filters and try again.")
        else:
            st.markdown("""
                <div class='hero'>
                    <h1>News Spotlight</h1>
                    <p>Discover fresh stories and insights that ignite your curiosity. Your journey into groundbreaking news starts here.</p>
                </div>
            """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()

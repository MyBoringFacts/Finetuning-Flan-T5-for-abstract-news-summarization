# News Summarization and Categorization System  

### Fine-tuning FLAN-T5 for News Domain Summarization  

**Developed by:** Thadoe Hein  
**Advisor:** Dr. Kwankamol Nongpong  
**Institution:** Assumption University of Thailand  

---

##  Overview  

This project, **QuantumQuest**, is an AI-powered news processing system that automatically **summarizes** and **categorizes** online news articles.  
The system fine-tunes **Flan-T5 (250M)** for abstractive summarization and integrates a **transformer-based SVM classifier** for news categorization.

---

##  Project Highlights  

- **Summarization Model:**  
  - Fine-tuned Flan-T5 (250M) on the **CNN/DailyMail dataset** (~312K articles).  
  - Optimized for concise, coherent, and context-aware summaries.  
  - Achieved **ROUGE-1: 0.4412**, **ROUGE-2: 0.2722**, **ROUGE-L: 0.3952**.  

- **Categorization Model:**  
  - Combined **AG News** and **News Category datasets**, filtered to 7 categories.  
  - Used **sentence-transformers/all-MiniLM-L6-v2** for embeddings.  
  - Classified with **SVM**, achieving **85.69% accuracy** and **F1-score: 0.86**.  

- **Deployment:**  
  - Backend built with **FastAPI** for summarization and classification requests.  
  - **Firebase** for storing processed summaries and metadata.  
  - **Streamlit UI** for visualizing categorized news and summaries.  

---

##  System Architecture  

World News API → Preprocessor → Flan-T5 Summarizer
↓
Transformer Embeddings
↓
SVM Classifier
↓
Firebase Database + Streamlit UI

yaml
Copy code

---

##  Technical Stack  

**Languages:** Python  
**Frameworks:** PyTorch, Hugging Face Transformers, Streamlit, FastAPI  
**Libraries:** Scikit-learn, Sentence-Transformers, NLTK, Pandas  
**Cloud & Storage:** Firebase Firestore  
**Model:** Flan-T5 (fine-tuned for summarization), SVM (for categorization)  

---

##  Evaluation Summary  

| Task | Metric | Result |
|------|---------|--------|
| Summarization | ROUGE-1 | 0.4412 |
| Summarization | ROUGE-2 | 0.2722 |
| Summarization | ROUGE-L | 0.3952 |
| Categorization | Accuracy | 85.69% |
| Categorization | F1-score | 0.86 |

---

##  Key Features  

- Automatic summarization of long-form news using fine-tuned Flan-T5  
- Multi-class news categorization into **World, Politics, Business, Sci/Tech, Sports, Entertainment, Others**  
- Clean preprocessing pipeline to remove noise and non-English text  
- Streamlit UI for exploring summarized, categorized news in real time  

---

##  Benchmark Comparison  

| Model | ROUGE-1 | ROUGE-2 | ROUGE-L | Notes |
|--------|----------|----------|----------|-------|
| **Flan-T5 (250M, fine-tuned)** | 44.1 | 27.2 | 39.5 | Efficient and domain-optimized |
| Pegasus | 44.2 | 22.5 | 43.8 | High quality, but costly |
| BART | 44.4 | 21.3 | 41.5 | Fluent but slower |
| T5-11B | 43.5 | 21.5 | 40.6 | Strong abstraction, high cost |

Flan-T5 achieves a balance between **performance and efficiency**, suitable for **real-time summarization** in limited-resource environments.

---

##  Future Improvements  

- Real-time news updates and summarization  
- Model quantization and GPU acceleration  
- Multilingual support (English → Burmese, Thai)  
- Personalized topic filtering and recommendation  

---


⭐ *“AI that reads the news, so you don’t have to.”*

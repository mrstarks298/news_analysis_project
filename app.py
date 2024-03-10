# from flask import Flask, render_template, request, redirect, url_for, session
# import nltk
# from nltk.corpus import stopwords, wordnet
# import requests
# from bs4 import BeautifulSoup
# from textblob import TextBlob
# from wordcloud import WordCloud
# import string
# import base64
# import io
# import matplotlib.pyplot as plt
# import seaborn as sns
# from pattern.en import sentiment as pattern_sentiment
# import spacy
# import psycopg2
# import json
# from collections import Counter
# from sumy.parsers.plaintext import PlaintextParser
# from sumy.nlp.tokenizers import Tokenizer
# from sumy.summarizers.lsa import LsaSummarizer
# from sumy.nlp.stemmers import Stemmer
# from sumy.utils import get_stop_words
# import numpy as np  # Import numpy for numerical operations

# # Load English language model for spaCy
# nlp = spacy.load('en_core_web_sm')

# # Initialize Flask app
# app = Flask(__name__)
# app.secret_key = 'your_secret_key'  # Change this to a random secret key

# # Download NLTK resources (run this only once)
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')

# # Database configuration
# DB_HOST = 'localhost'
# DB_NAME = 'postgres'
# DB_USER = 'postgres'
# DB_PASSWORD = 'Saurabh_Agrahari'

# # Function to establish database connection
# def connect_to_db():
#     conn = psycopg2.connect(host=DB_HOST, database=DB_NAME, user=DB_USER, password=DB_PASSWORD)
#     return conn

# # Function to check if admin is logged in
# def is_admin_logged_in():
#     return session.get('admin_logged_in', False)

# # Function to perform Named Entity Recognition (NER)
# def perform_ner(text):
#     doc = nlp(text)
#     entities = [(ent.text, ent.label_) for ent in doc.ents]
#     return entities

# # Function to perform Keyword Extraction
# def extract_keywords(text):
#     doc = nlp(text)
#     # Get nouns and proper nouns as keywords
#     keywords = [token.text for token in doc if token.pos_ in ['NOUN', 'PROPN']]
#     return keywords

# def clean_text(text):
#     # Remove punctuation except periods
#     text = ''.join([char for char in text if char not in string.punctuation or char == '.'])
#     # Convert text to lowercase
#     text = text.lower()
#     # Remove extra whitespace
#     text = ' '.join(text.split())
#     return text

# # Function to generate sentiment plot
# def generate_sentiment_plot(blob, title):
#     polarity_scores = [sentence.sentiment.polarity for sentence in blob.sentences]
#     plt.figure(figsize=(10, 6))
#     sns.histplot(polarity_scores, bins=20, kde=True, color='skyblue', stat='density', linewidth=0)
#     plt.xlabel('Sentence Polarity', fontsize=14)
#     plt.ylabel('Density', fontsize=14)
#     plt.title(title, fontsize=16)
#     plt.grid(True, linestyle='--', alpha=0.7)
#     plt.legend(['Sentiment Distribution'], loc='upper right', fontsize=12)
#     # Enhance readability of x-axis tick labels
#     plt.xticks(np.arange(-1, 1.1, 0.2), fontsize=12)
#     plt.yticks(fontsize=12)
#     plt.tight_layout()
#     img_data = io.BytesIO()
#     plt.savefig(img_data, format='png')
#     plt.close()
#     img_data.seek(0)
#     encoded_img_data = base64.b64encode(img_data.read()).decode()
#     return encoded_img_data

# # Function to generate pattern sentiment analysis plot
# def generate_pattern_sentiment_plot(text):
#     scores = [pattern_sentiment(sentence)[0] for sentence in nltk.sent_tokenize(text)]
#     plt.figure(figsize=(10, 6))
#     sns.histplot(scores, bins=20, kde=True, color='salmon', stat='density', linewidth=0)
#     plt.xlabel('Sentence Polarity (Pattern)', fontsize=14)
#     plt.ylabel('Density', fontsize=14)
#     plt.title('Pattern Sentiment Analysis', fontsize=16)
#     plt.grid(True, linestyle='--', alpha=0.7)
#     plt.legend(['Sentiment Distribution'], loc='upper right', fontsize=12)
#     # Enhance readability of x-axis tick labels
#     plt.xticks(np.arange(-1, 1.1, 0.2), fontsize=12)
#     plt.yticks(fontsize=12)
#     plt.tight_layout()
#     img_data = io.BytesIO()
#     plt.savefig(img_data, format='png')
#     plt.close()
#     img_data.seek(0)
#     encoded_img_data = base64.b64encode(img_data.read()).decode()
#     return encoded_img_data

# # Function to perform text summarization using Sumy
# def generate_summary(text):
#     LANGUAGE = "english"
#     SENTENCES_COUNT = 3

#     parser = PlaintextParser.from_string(text, Tokenizer(LANGUAGE))
#     stemmer = Stemmer(LANGUAGE)

#     summarizer = LsaSummarizer(stemmer)
#     summarizer.stop_words = get_stop_words(LANGUAGE)

#     summary = summarizer(parser.document, SENTENCES_COUNT)
    
#     # Capitalize the first letter of each sentence
#     formatted_summary = []
#     for sentence in summary:
#         formatted_summary.append(str(sentence).capitalize())

#     return ' '.join(formatted_summary)


# # Route to display the input form
# @app.route('/')
# def index():
#     return render_template('index.html')

# # Function to get synonyms of a word
# def get_synonyms(word):
#     synonyms = set()
#     for synset in wordnet.synsets(word):
#         for lemma in synset.lemmas():
#             synonyms.add(lemma.name())
#     return list(synonyms)

# # Route to handle form submission and display analysis results
# @app.route('/analyze', methods=['POST'])
# def analyze_data():
#     try:
#         # Get user input from form
#         url = request.form['url']

#         # Fetch data from the provided website
#         response = requests.get(url)
#         response.raise_for_status()  # Raise an exception for HTTP errors
#         html = response.text

#         # Parse HTML
#         soup = BeautifulSoup(html, 'html.parser')
#         paragraphs = [paragraph.get_text() for paragraph in soup.find_all('p')]
#         text = ' '.join(paragraphs)

#         # Clean the text
#         cleaned_text = clean_text(text)

#         # Tokenize cleaned text
#         tokens = nltk.word_tokenize(cleaned_text)

#         # Perform part-of-speech tagging
#         pos_tags = nltk.pos_tag(tokens)

#         # Count number of words
#         num_words = len(tokens)

#         # Count number of sentences without removing periods
#         num_sentences = len(nltk.sent_tokenize(text))

#         # Sentiment analysis using TextBlob
#         blob = TextBlob(cleaned_text)
#         sentiment_score = blob.sentiment.polarity

#         # Word cloud generation
#         wordcloud = WordCloud(width=800, height=400, background_color='white').generate(cleaned_text)

#         # TextBlob sentiment analysis
#         textblob_sentiment_plot = generate_sentiment_plot(blob, 'TextBlob Sentiment Analysis')

#         # Pattern sentiment analysis
#         pattern_sentiment_plot = generate_pattern_sentiment_plot(cleaned_text)

#         # Count POS tags
#         pos_counts = {}
#         for _, tag in pos_tags:
#             pos_counts[tag] = pos_counts.get(tag, 0) + 1

#         # Perform Named Entity Recognition (NER)
#         ner_results = perform_ner(cleaned_text)

#         # Extract keywords
#         keywords = extract_keywords(cleaned_text)

#         # Word frequency distribution
#         word_freq_table = Counter(tokens)

#         # Get top 10 most frequent words
#         top_10_words = dict(word_freq_table.most_common(10))

#         # Perform text summarization
#         summary = generate_summary(cleaned_text)

#         # Get synonyms for keywords
#         keyword_synonyms = {}
#         for keyword in keywords:
#             synonyms = get_synonyms(keyword)
#             keyword_synonyms[keyword] = synonyms

#         # Connect to the database
#         conn = connect_to_db()
#         cur = conn.cursor()

#         # Convert pos_counts dictionary to JSON string
#         pos_tags_json = json.dumps(pos_counts)

#         # Insert analysis results into the database
#         cur.execute("INSERT INTO new_table(url, paragraph, num_words, num_sentences, sentiment_score, pos_tags) VALUES (%s, %s, %s, %s, %s, %s)",
#                     (url, text, num_words, num_sentences, sentiment_score, pos_tags_json))

#         conn.commit()

#         # Close database connection
#         cur.close()
#         conn.close()

#         # Convert Word Cloud image to base64
#         img_data = io.BytesIO()
#         wordcloud.to_image().save(img_data, format='PNG')
#         img_data.seek(0)
#         encoded_img_data = base64.b64encode(img_data.read()).decode()

#         # Render template with analysis results
#         return render_template('results.html', pos_counts=pos_counts, num_words=num_words, num_sentences=num_sentences,
#                                sentiment_score=sentiment_score,
#                                paragraphs=text,
#                                wordcloud_img=encoded_img_data,
#                                textblob_sentiment_plot=textblob_sentiment_plot,
#                                pattern_sentiment_plot=pattern_sentiment_plot,
#                                ner_results=ner_results,
#                                keywords=keywords,
#                                synonyms=synonyms,    # Pass synonyms data to the template
#                                word_freq_table=top_10_words,
#                                summary=summary
#                                )

#     except requests.exceptions.RequestException as e:
# Import necessary libraries
# from flask import Flask, render_template, request, redirect, url_for, session
# from bs4 import BeautifulSoup
# from textblob import TextBlob
# from wordcloud import WordCloud
# import requests
# import nltk
# from nltk.corpus import wordnet
# import string
# import base64
# import io
# import matplotlib.pyplot as plt
# import seaborn as sns
# from pattern.en import sentiment as pattern_sentiment
# import spacy
# import psycopg2
# from collections import Counter
# from sumy.parsers.plaintext import PlaintextParser
# from sumy.nlp.tokenizers import Tokenizer
# from sumy.summarizers.lsa import LsaSummarizer
# from sumy.nlp.stemmers import Stemmer
# from sumy.utils import get_stop_words
# import numpy as np

# # Load English language model for spaCy
# nlp = spacy.load('en_core_web_sm')

# # Initialize Flask app
# app = Flask(__name__)
# app.secret_key = 'your_secret_key'

# # Download NLTK resources (run this only once)
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')

# # Database configuration
# DB_HOST = 'localhost'
# DB_NAME = 'postgres'
# DB_USER = 'postgres'
# DB_PASSWORD = 'Saurabh_Agrahari'

# # Function to establish database connection
# def connect_to_db():
#     conn = psycopg2.connect(host=DB_HOST, database=DB_NAME, user=DB_USER, password=DB_PASSWORD)
#     return conn

# # Function to check if admin is logged in
# def is_admin_logged_in():
#     return session.get('admin_logged_in', False)

# # Function to perform Named Entity Recognition (NER)
# def perform_ner(text):
#     doc = nlp(text)
#     entities = [(ent.text, ent.label_) for ent in doc.ents]
#     return entities

# # Function to perform Keyword Extraction
# def extract_keywords(text):
#     doc = nlp(text)
#     keywords = [token.text for token in doc if token.pos_ in ['NOUN', 'PROPN']]
#     return keywords

# def clean_text(text):
#     text = ''.join([char for char in text if char not in string.punctuation or char == '.'])
#     text = text.lower()
#     text = ' '.join(text.split())
#     return text

# # Function to generate sentiment plot
# def generate_sentiment_plot(blob, title):
#     polarity_scores = [sentence.sentiment.polarity for sentence in blob.sentences]
#     plt.figure(figsize=(10, 6))
#     sns.histplot(polarity_scores, bins=20, kde=True, color='skyblue', stat='density', linewidth=0)
#     plt.xlabel('Sentence Polarity', fontsize=14)
#     plt.ylabel('Density', fontsize=14)
#     plt.title(title, fontsize=16)
#     plt.grid(True, linestyle='--', alpha=0.7)
#     plt.legend(['Sentiment Distribution'], loc='upper right', fontsize=12)
#     plt.xticks(np.arange(-1, 1.1, 0.2), fontsize=12)
#     plt.yticks(fontsize=12)
#     plt.tight_layout()
#     img_data = io.BytesIO()
#     plt.savefig(img_data, format='png')
#     plt.close()
#     img_data.seek(0)
#     encoded_img_data = base64.b64encode(img_data.read()).decode()
#     return encoded_img_data

# # Function to generate pattern sentiment analysis plot
# def generate_pattern_sentiment_plot(text):
#     scores = [pattern_sentiment(sentence)[0] for sentence in nltk.sent_tokenize(text)]
#     plt.figure(figsize=(10, 6))
#     sns.histplot(scores, bins=20, kde=True, color='salmon', stat='density', linewidth=0)
#     plt.xlabel('Sentence Polarity (Pattern)', fontsize=14)
#     plt.ylabel('Density', fontsize=14)
#     plt.title('Pattern Sentiment Analysis', fontsize=16)
#     plt.grid(True, linestyle='--', alpha=0.7)
#     plt.legend(['Sentiment Distribution'], loc='upper right', fontsize=12)
#     plt.xticks(np.arange(-1, 1.1, 0.2), fontsize=12)
#     plt.yticks(fontsize=12)
#     plt.tight_layout()
#     img_data = io.BytesIO()
#     plt.savefig(img_data, format='png')
#     plt.close()
#     img_data.seek(0)
#     encoded_img_data = base64.b64encode(img_data.read()).decode()
#     return encoded_img_data

# # Function to perform text summarization using Sumy
# def generate_summary(text):
#     LANGUAGE = "english"
#     SENTENCES_COUNT = 3

#     parser = PlaintextParser.from_string(text, Tokenizer(LANGUAGE))
#     stemmer = Stemmer(LANGUAGE)

#     summarizer = LsaSummarizer(stemmer)
#     summarizer.stop_words = get_stop_words(LANGUAGE)

#     summary = summarizer(parser.document, SENTENCES_COUNT)
    
#     formatted_summary = []
#     for sentence in summary:
#         formatted_summary.append(str(sentence).capitalize())

#     return ' '.join(formatted_summary)

# # Function to get synonyms of a word
# def get_synonyms(word):
#     synonyms = set()
#     for synset in wordnet.synsets(word):
#         for lemma in synset.lemmas():
#             synonyms.add(lemma.name())
#     return list(synonyms)

# # Route to display the input form
# @app.route('/')
# def index():
#     return render_template('index.html')

# # Function to fetch data from a URL
# def fetch_url_data(url):
#     try:
#         response = requests.get(url)
#         response.raise_for_status()
#         return response.text
#     except requests.exceptions.RequestException as e:
#         print(f"Error: Failed to fetch data from the provided URL. {e}")
#         return None

# # Route to handle form submission and display analysis results
# @app.route('/analyze', methods=['POST'])
# def analyze_data():
#     try:
#         url = request.form['url']
#         html = fetch_url_data(url)
#         if html is None:
#             return "Error: Failed to fetch data from the provided URL."

#         soup = BeautifulSoup(html, 'html.parser')
#         paragraphs = [paragraph.get_text() for paragraph in soup.find_all('p')]
#         text = ' '.join(paragraphs)

#         cleaned_text = clean_text(text)
#         tokens = nltk.word_tokenize(cleaned_text)
#         pos_tags = nltk.pos_tag(tokens)
#         num_words = len(tokens)
#         num_sentences = len(nltk.sent_tokenize(text))

#         blob = TextBlob(cleaned_text)
#         sentiment_score = blob.sentiment.polarity

#         wordcloud = WordCloud(width=800, height=400, background_color='white').generate(cleaned_text)
#         wordcloud_img = base64.b64encode(wordcloud.to_image()).decode()

#         textblob_sentiment_plot = generate_sentiment_plot(blob, 'TextBlob Sentiment Analysis')
#         pattern_sentiment_plot = generate_pattern_sentiment_plot(cleaned_text)

#         pos_counts = {}
#         for _, tag in pos_tags:
#             pos_counts[tag] = pos_counts.get(tag, 0) + 1

#         ner_results = perform_ner(cleaned_text)
#         keywords = extract_keywords(cleaned_text)
#         word_freq_table = Counter(tokens)
#         top_10_words = dict(word_freq_table.most_common(10))
#         summary = generate_summary(cleaned_text)
#         keyword_synonyms = {}
#         for keyword in keywords:
#             synonyms = get_synonyms(keyword)
#             keyword_synonyms[keyword] = synonyms

#         return render_template('results.html', pos_counts=pos_counts, num_words=num_words, num_sentences=num_sentences,
#                                sentiment_score=sentiment_score,
#                                paragraphs=text,
#                                wordcloud_img=wordcloud_img,
#                                textblob_sentiment_plot=textblob_sentiment_plot,
#                                pattern_sentiment_plot=pattern_sentiment_plot,
#                                ner_results=ner_results,
#                                keywords=keywords,
#                                synonyms=keyword_synonyms,
#                                word_freq_table=top_10_words,
#                                summary=summary
#                                )

#     except Exception as e:
#         return f"An error occurred while processing the data: {e}"

# # Route to display admin login form
# @app.route('/admin-login', methods=['GET', 'POST'])
# def admin_login():
#     if request.method == 'POST':
#         username = request.form['username']
#         password = request.form['password']
#         if username == 'admin' and password == 'admin_password':
#             session['admin_logged_in'] = True
#             return redirect(url_for('admin_page'))
#         else:
#             return render_template('admin_login.html', error='Invalid username or password')
#     return render_template('admin_login.html')

# # Route to display admin page with history results
# @app.route('/admin-page')
# def admin_page():
#     if not is_admin_logged_in():
#         return redirect(url_for('admin_login'))
#     try:
#         conn = connect_to_db()
#         cur = conn.cursor()
#         cur.execute("SELECT url, paragraph, num_words, num_sentences, sentiment_score FROM new_table")
#         history_results = cur.fetchall()
#         cur.close()
#         conn.close()
#         return render_template('admin_page.html', history_results=history_results)
#     except Exception as e:
#         return f"An error occurred while fetching history: {e}"

# # Route to handle admin logout
# @app.route('/admin-logout')
# def admin_logout():
#     session.pop('admin_logged_in', None)
#     return redirect(url_for('admin_login'))

# if __name__ == '__main__':
#     app.run(debug=True)
# from flask import Flask, render_template, request, redirect, url_for, session
# from bs4 import BeautifulSoup
# from textblob import TextBlob
# from wordcloud import WordCloud
# import base64
# import requests
# import nltk
# import string
# import io
# import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt
# from pattern.en import sentiment as pattern_sentiment
# import spacy
# from collections import Counter
# from sumy.parsers.plaintext import PlaintextParser
# from sumy.nlp.tokenizers import Tokenizer
# from sumy.summarizers.lsa import LsaSummarizer
# from sumy.nlp.stemmers import Stemmer
# from sumy.utils import get_stop_words
# from nltk.corpus import wordnet
# import psycopg2

# # Initialize Flask app
# app = Flask(__name__)
# app.secret_key = 'your_secret_key'  # Change this to a random secret key

# # Load English language model for spaCy
# nlp = spacy.load('en_core_web_sm')

# # Download NLTK resources (run this only once)
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')

# # Database configuration
# DB_HOST = 'localhost'
# DB_NAME = 'postgres'
# DB_USER = 'postgres'
# DB_PASSWORD = 'Saurabh_Agrahari'

# # Function to establish database connection
# def connect_to_db():
#     conn = psycopg2.connect(host=DB_HOST, database=DB_NAME, user=DB_USER, password=DB_PASSWORD)
#     return conn

# # Function to check if admin is logged in
# def is_admin_logged_in():
#     return session.get('admin_logged_in', False)

# # Function to perform Named Entity Recognition (NER)
# def perform_ner(text):
#     doc = nlp(text)
#     entities = [(ent.text, ent.label_) for ent in doc.ents]
#     return entities

# # Function to perform Keyword Extraction
# def extract_keywords(text):
#     doc = nlp(text)
#     keywords = [token.text for token in doc if token.pos_ in ['NOUN', 'PROPN']]
#     return keywords

# # Function to clean text
# def clean_text(text):
#     text = ''.join([char for char in text if char not in string.punctuation or char == '.'])
#     text = text.lower()
#     text = ' '.join(text.split())
#     return text

# # Function to generate sentiment plot
# def generate_sentiment_plot(blob, title):
#     polarity_scores = [sentence.sentiment.polarity for sentence in blob.sentences]
#     plt.figure(figsize=(10, 6))
#     sns.histplot(polarity_scores, bins=20, kde=True, color='skyblue', stat='density', linewidth=0)
#     plt.xlabel('Sentence Polarity', fontsize=14)
#     plt.ylabel('Density', fontsize=14)
#     plt.title(title, fontsize=16)
#     plt.grid(True, linestyle='--', alpha=0.7)
#     plt.legend(['Sentiment Distribution'], loc='upper right', fontsize=12)
#     plt.xticks(np.arange(-1, 1.1, 0.2), fontsize=12)
#     plt.yticks(fontsize=12)
#     plt.tight_layout()
#     img_data = io.BytesIO()
#     plt.savefig(img_data, format='png')
#     plt.close()
#     img_data.seek(0)
#     encoded_img_data = base64.b64encode(img_data.read()).decode()
#     return encoded_img_data

# # Function to generate pattern sentiment analysis plot
# def generate_pattern_sentiment_plot(text):
#     scores = [pattern_sentiment(sentence)[0] for sentence in nltk.sent_tokenize(text)]
#     plt.figure(figsize=(10, 6))
#     sns.histplot(scores, bins=20, kde=True, color='salmon', stat='density', linewidth=0)
#     plt.xlabel('Sentence Polarity (Pattern)', fontsize=14)
#     plt.ylabel('Density', fontsize=14)
#     plt.title('Pattern Sentiment Analysis', fontsize=16)
#     plt.grid(True, linestyle='--', alpha=0.7)
#     plt.legend(['Sentiment Distribution'], loc='upper right', fontsize=12)
#     plt.xticks(np.arange(-1, 1.1, 0.2), fontsize=12)
#     plt.yticks(fontsize=12)
#     plt.tight_layout()
#     img_data = io.BytesIO()
#     plt.savefig(img_data, format='png')
#     plt.close()
#     img_data.seek(0)
#     encoded_img_data = base64.b64encode(img_data.read()).decode()
#     return encoded_img_data

# # Function to perform text summarization using Sumy
# def generate_summary(text):
#     LANGUAGE = "english"
#     SENTENCES_COUNT = 3

#     parser = PlaintextParser.from_string(text, Tokenizer(LANGUAGE))
#     stemmer = Stemmer(LANGUAGE)

#     summarizer = LsaSummarizer(stemmer)
#     summarizer.stop_words = get_stop_words(LANGUAGE)

#     summary = summarizer(parser.document, SENTENCES_COUNT)
    
#     formatted_summary = []
#     for sentence in summary:
#         formatted_summary.append(str(sentence).capitalize())

#     return ' '.join(formatted_summary)

# # Function to get synonyms of a word
# def get_synonyms(word):
#     synonyms = set()
#     for synset in wordnet.synsets(word):
#         for lemma in synset.lemmas():
#             synonyms.add(lemma.name())
#     return list(synonyms)

# # Route to display the input form
# @app.route('/')
# def index():
#     return render_template('index.html')

# # Function to fetch data from a URL
# def fetch_url_data(url):
#     try:
#         response = requests.get(url)
#         response.raise_for_status()
#         return response.text
#     except requests.exceptions.RequestException as e:
#         print(f"Error: Failed to fetch data from the provided URL. {e}")
#         return None

# # Route to handle form submission and display analysis results
# @app.route('/analyze', methods=['POST'])
# def analyze_data():
#     try:
#         url = request.form['url']
#         html = fetch_url_data(url)
#         if html is None:
#             return "Error: Failed to fetch data from the provided URL."

#         soup = BeautifulSoup(html, 'html.parser')
#         paragraphs = [paragraph.get_text() for paragraph in soup.find_all('p')]
#         text = ' '.join(paragraphs)

#         cleaned_text = clean_text(text)
#         tokens = nltk.word_tokenize(cleaned_text)
#         pos_tags = nltk.pos_tag(tokens)
#         num_words = len(tokens)
#         num_sentences = len(nltk.sent_tokenize(text))

#         blob = TextBlob(cleaned_text)
#         sentiment_score = blob.sentiment.polarity
#         wordcloud = WordCloud(width=800, height=400, background_color='white').generate(cleaned_text)
#         wordcloud_img = wordcloud.to_image()
#         img_data = io.BytesIO()
#         wordcloud_img.save(img_data, format='PNG')
#         img_data.seek(0)
#         encoded_img_data = base64.b64encode(img_data.read()).decode()


#         pattern_sentiment_plot = generate_pattern_sentiment_plot(text)  # Pass original text to Pattern sentiment analysis function
#         textblob_sentiment_plot = generate_sentiment_plot(blob, 'TextBlob Sentiment Analysis')  # Pass TextBlob blob object to TextBlob sentiment analysis function

#         pos_counts = {}
#         for _, tag in pos_tags:
#             if tag in pos_counts:
#                 pos_counts[tag] += 1
#             else:
#                 pos_counts[tag] = 1

#         sorted_pos_counts = dict(sorted(pos_counts.items(), key=lambda item: item[1], reverse=True))

#         named_entities = perform_ner(text)
#         keywords = extract_keywords(text)
#         summary = generate_summary(text)

#         return render_template('results.html', num_words=num_words, num_sentences=num_sentences,
#                                sentiment_score=sentiment_score, wordcloud=encoded_img_data,
#                                pattern_sentiment_plot=pattern_sentiment_plot,
#                                textblob_sentiment_plot=textblob_sentiment_plot,
#                                pos_counts=sorted_pos_counts, named_entities=named_entities,
#                                keywords=keywords, summary=summary)
#     except Exception as e:
#         print(f"An error occurred: {e}")
#         return "Error: An unexpected error occurred."

# # Route to display login form
# @app.route('/login')
# def login():
#     if is_admin_logged_in():
#         return redirect(url_for('admin_dashboard'))
#     return render_template('login.html')

# # Route to handle login form submission
# @app.route('/login', methods=['POST'])
# def login_submit():
#     username = request.form['username']
#     password = request.form['password']
#     if username == 'admin' and password == 'admin_password':
#         session['admin_logged_in'] = True
#         return redirect(url_for('admin_dashboard'))
#     else:
#         return render_template('login.html', error=True)

# # Route to display admin dashboard
# @app.route('/admin/dashboard')
# def admin_dashboard():
#     if not is_admin_logged_in():
#         return redirect(url_for('login'))
#     return render_template('admin_dashboard.html')

# # Route to handle admin logout
# @app.route('/admin/logout')
# def admin_logout():
#     session.pop('admin_logged_in', None)
#     return redirect(url_for('login'))

# if __name__ == '__main__':
#     app.run(debug=True)
# from flask import Flask, render_template, request, redirect, url_for, session
# import nltk
# from nltk.corpus import stopwords, wordnet
# import requests
# from bs4 import BeautifulSoup
# from textblob import TextBlob
# from wordcloud import WordCloud
# import string
# import base64
# import io
# import matplotlib.pyplot as plt
# import seaborn as sns
# from pattern.en import sentiment as pattern_sentiment
# import spacy
# import psycopg2
# import json
# from collections import Counter
# from sumy.parsers.plaintext import PlaintextParser
# from sumy.nlp.tokenizers import Tokenizer
# from sumy.summarizers.lsa import LsaSummarizer
# from sumy.nlp.stemmers import Stemmer
# from sumy.utils import get_stop_words
# import numpy as np  # Import numpy for numerical operations

# # Load English language model for spaCy
# nlp = spacy.load('en_core_web_sm')

# # Initialize Flask app
# app = Flask(__name__)
# app.secret_key = 'your_secret_key'  # Change this to a random secret key

# # Download NLTK resources (run this only once)
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')

# # Database configuration
# DB_HOST = 'localhost'
# DB_NAME = 'postgres'
# DB_USER = 'postgres'
# DB_PASSWORD = 'Saurabh_Agrahari'

# # Function to establish database connection
# def connect_to_db():
#     conn = psycopg2.connect(host=DB_HOST, database=DB_NAME, user=DB_USER, password=DB_PASSWORD)
#     return conn

# # Function to check if admin is logged in
# def is_admin_logged_in():
#     return session.get('admin_logged_in', False)

# # Function to perform Named Entity Recognition (NER)
# def perform_ner(text):
#     doc = nlp(text)
#     entities = [(ent.text, ent.label_) for ent in doc.ents]
#     return entities

# # Function to perform Keyword Extraction
# def extract_keywords(text):
#     doc = nlp(text)
#     # Get nouns and proper nouns as keywords
#     keywords = [token.text for token in doc if token.pos_ in ['NOUN', 'PROPN']]
#     return keywords

# def clean_text(text):
#     # Remove punctuation except periods
#     text = ''.join([char for char in text if char not in string.punctuation or char == '.'])
#     # Convert text to lowercase
#     text = text.lower()
#     # Remove extra whitespace
#     text = ' '.join(text.split())
#     return text

# # Function to generate sentiment plot
# def generate_sentiment_plot(blob, title):
#     polarity_scores = [sentence.sentiment.polarity for sentence in blob.sentences]
#     plt.figure(figsize=(10, 6))
#     sns.histplot(polarity_scores, bins=20, kde=True, color='skyblue', stat='density', linewidth=0)
#     plt.xlabel('Sentence Polarity', fontsize=14)
#     plt.ylabel('Density', fontsize=14)
#     plt.title(title, fontsize=16)
#     plt.grid(True, linestyle='--', alpha=0.7)
#     plt.legend(['Sentiment Distribution'], loc='upper right', fontsize=12)
#     # Enhance readability of x-axis tick labels
#     plt.xticks(np.arange(-1, 1.1, 0.2), fontsize=12)
#     plt.yticks(fontsize=12)
#     plt.tight_layout()
#     img_data = io.BytesIO()
#     plt.savefig(img_data, format='png')
#     plt.close()
#     img_data.seek(0)
#     encoded_img_data = base64.b64encode(img_data.read()).decode()
#     return encoded_img_data

# # Function to generate pattern sentiment analysis plot
# def generate_pattern_sentiment_plot(text):
#     scores = [pattern_sentiment(sentence)[0] for sentence in nltk.sent_tokenize(text)]
#     plt.figure(figsize=(10, 6))
#     sns.histplot(scores, bins=20, kde=True, color='salmon', stat='density', linewidth=0)
#     plt.xlabel('Sentence Polarity (Pattern)', fontsize=14)
#     plt.ylabel('Density', fontsize=14)
#     plt.title('Pattern Sentiment Analysis', fontsize=16)
#     plt.grid(True, linestyle='--', alpha=0.7)
#     plt.legend(['Sentiment Distribution'], loc='upper right', fontsize=12)
#     # Enhance readability of x-axis tick labels
#     plt.xticks(np.arange(-1, 1.1, 0.2), fontsize=12)
#     plt.yticks(fontsize=12)
#     plt.tight_layout()
#     img_data = io.BytesIO()
#     plt.savefig(img_data, format='png')
#     plt.close()
#     img_data.seek(0)
#     encoded_img_data = base64.b64encode(img_data.read()).decode()
#     return encoded_img_data

# # Function to perform text summarization using Sumy
# def generate_summary(text):
#     LANGUAGE = "english"
#     SENTENCES_COUNT = 3

#     parser = PlaintextParser.from_string(text, Tokenizer(LANGUAGE))
#     stemmer = Stemmer(LANGUAGE)

#     summarizer = LsaSummarizer(stemmer)
#     summarizer.stop_words = get_stop_words(LANGUAGE)

#     summary = summarizer(parser.document, SENTENCES_COUNT)
    
#     # Capitalize the first letter of each sentence
#     formatted_summary = []
#     for sentence in summary:
#         formatted_summary.append(str(sentence).capitalize())

#     return ' '.join(formatted_summary)


# # Route to display the input form
# @app.route('/')
# def index():
#     return render_template('index.html')

# # Function to get synonyms of a word
# def get_synonyms(word):
#     synonyms = set()
#     for synset in wordnet.synsets(word):
#         for lemma in synset.lemmas():
#             synonyms.add(lemma.name())
#     return list(synonyms)

# # Route to handle form submission and display analysis results
# @app.route('/analyze', methods=['POST'])
# def analyze_data():
#     try:
#         # Get user input from form
#         url = request.form['url']

#         # Fetch data from the provided website
#         response = requests.get(url)
#         response.raise_for_status()  # Raise an exception for HTTP errors
#         html = response.text

#         # Parse HTML
#         soup = BeautifulSoup(html, 'html.parser')
#         paragraphs = [paragraph.get_text() for paragraph in soup.find_all('p')]
#         text = ' '.join(paragraphs)

#         # Clean the text
#         cleaned_text = clean_text(text)

#         # Tokenize cleaned text
#         tokens = nltk.word_tokenize(cleaned_text)

#         # Perform part-of-speech tagging
#         pos_tags = nltk.pos_tag(tokens)

#         # Count number of words
#         num_words = len(tokens)

#         # Count number of sentences without removing periods
#         num_sentences = len(nltk.sent_tokenize(text))

#         # Sentiment analysis using TextBlob
#         blob = TextBlob(cleaned_text)
#         sentiment_score = blob.sentiment.polarity

#         # Word cloud generation
#         wordcloud = WordCloud(width=800, height=400, background_color='white').generate(cleaned_text)

#         # TextBlob sentiment analysis
#         textblob_sentiment_plot = generate_sentiment_plot(blob, 'TextBlob Sentiment Analysis')

#         # Pattern sentiment analysis
#         pattern_sentiment_plot = generate_pattern_sentiment_plot(cleaned_text)

#         # Count POS tags
#         pos_counts = {}
#         for _, tag in pos_tags:
#             pos_counts[tag] = pos_counts.get(tag, 0) + 1

#         # Perform Named Entity Recognition (NER)
#         ner_results = perform_ner(cleaned_text)

#         # Extract keywords
#         keywords = extract_keywords(cleaned_text)

#         # Word frequency distribution
#         word_freq_table = Counter(tokens)

#         # Get top 10 most frequent words
#         top_10_words = dict(word_freq_table.most_common(10))

#         # Perform text summarization
#         summary = generate_summary(cleaned_text)

#         # Get synonyms for keywords
#         keyword_synonyms = {}
#         for keyword in keywords:
#             synonyms = get_synonyms(keyword)
#             keyword_synonyms[keyword] = synonyms

#         # Connect to the database
#         conn = connect_to_db()
#         cur = conn.cursor()

#         # Convert pos_counts dictionary to JSON string
#         pos_tags_json = json.dumps(pos_counts)

#         # Insert analysis results into the database
#         cur.execute("INSERT INTO new_table(url, paragraph, num_words, num_sentences, sentiment_score, pos_tags) VALUES (%s, %s, %s, %s, %s, %s)",
#                     (url, text, num_words, num_sentences, sentiment_score, pos_tags_json))

#         conn.commit()

#         # Close database connection
#         cur.close()
#         conn.close()

#         # Convert Word Cloud image to base64
#         img_data = io.BytesIO()
#         wordcloud.to_image().save(img_data, format='PNG')
#         img_data.seek(0)
#         encoded_img_data = base64.b64encode(img_data.read()).decode()

#         # Render template with analysis results
#         return render_template('results.html', pos_counts=pos_counts, num_words=num_words, num_sentences=num_sentences,
#                                sentiment_score=sentiment_score,
#                                paragraphs=text,
#                                wordcloud_img=encoded_img_data,
#                                textblob_sentiment_plot=textblob_sentiment_plot,
#                                pattern_sentiment_plot=pattern_sentiment_plot,
#                                ner_results=ner_results,
#                                keywords=keywords,
#                                synonyms=synonyms,    # Pass synonyms data to the template
#                                word_freq_table=top_10_words,
#                                summary=summary
#                                )

#     except requests.exceptions.RequestException as e:
#         return f"Error: Failed to fetch data from the provided URL. {e}"
#     except Exception as e:
#         return f"An error occurred while processing the data: {e}"

# # Route to display admin login form
# @app.route('/admin-login', methods=['GET', 'POST'])
# def admin_login():
#     if request.method == 'POST':
#         username = request.form['username']
#         password = request.form['password']
#         # Validate admin credentials (replace with your authentication mechanism)
#         if username == 'admin' and password == 'admin_password':
#             session['admin_logged_in'] = True
#             return redirect(url_for('admin_page'))
#         else:
#             return render_template('admin_login.html', error='Invalid username or password')
#     return render_template('admin_login.html')

# # Route to display admin page with history results
# @app.route('/admin-page')
# def admin_page():
#     if not is_admin_logged_in():
#         return redirect(url_for('admin_login'))
#     try:
#         # Connect to the database
#         conn = connect_to_db()
#         cur = conn.cursor()

#         # Fetch history results from the database
#         cur.execute("SELECT url, paragraph, num_words, num_sentences, sentiment_score FROM new_table")
#         history_results = cur.fetchall()

#         # Close database connection
#         cur.close()
#         conn.close()

#         return render_template('admin_page.html', history_results=history_results)
#     except Exception as e:
#         return f"An error occurred while fetching history: {e}"

# # Route to handle admin logout
# @app.route('/admin-logout')
# def admin_logout():
#     session.pop('admin_logged_in', None)
#     return redirect(url_for('admin_login'))

# if __name__ == '__main__':
#     app.run(debug=True)
# from flask import Flask, render_template, request, redirect, url_for, session
# import nltk
# from nltk.corpus import stopwords, wordnet
# import requests
# from bs4 import BeautifulSoup
# from textblob import TextBlob
# from wordcloud import WordCloud
# import string
# import base64
# import io
# import matplotlib.pyplot as plt
# import seaborn as sns
# from pattern.en import sentiment as pattern_sentiment
# import spacy
# import psycopg2
# import json
# from collections import Counter
# from sumy.parsers.plaintext import PlaintextParser
# from sumy.nlp.tokenizers import Tokenizer
# from sumy.summarizers.lsa import LsaSummarizer
# from sumy.nlp.stemmers import Stemmer
# from sumy.utils import get_stop_words
# import numpy as np  # Import numpy for numerical operations

# # Load English language model for spaCy
# nlp = spacy.load('en_core_web_sm')

# # Initialize Flask app
# app = Flask(__name__)
# app.secret_key = 'your_secret_key'  # Change this to a random secret key

# # Download NLTK resources (run this only once)
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')

# # Database configuration
# DB_HOST = 'localhost'
# DB_NAME = 'postgres'
# DB_USER = 'postgres'
# DB_PASSWORD = 'Saurabh_Agrahari'

# # Function to establish database connection
# def connect_to_db():
#     conn = psycopg2.connect(host=DB_HOST, database=DB_NAME, user=DB_USER, password=DB_PASSWORD)
#     return conn

# # Function to check if admin is logged in
# def is_admin_logged_in():
#     return session.get('admin_logged_in', False)

# # Function to perform Named Entity Recognition (NER)
# def perform_ner(text):
#     doc = nlp(text)
#     entities = [(ent.text, ent.label_) for ent in doc.ents]
#     return entities

# # Function to perform Keyword Extraction
# def extract_keywords(text):
#     doc = nlp(text)
#     # Get nouns and proper nouns as keywords
#     keywords = [token.text for token in doc if token.pos_ in ['NOUN', 'PROPN']]
#     return keywords

# def clean_text(text):
#     # Remove punctuation except periods
#     text = ''.join([char for char in text if char not in string.punctuation or char == '.'])
#     # Convert text to lowercase
#     text = text.lower()
#     # Remove extra whitespace
#     text = ' '.join(text.split())
#     return text

# # Function to generate sentiment plot
# def generate_sentiment_plot(blob, title):
#     polarity_scores = [sentence.sentiment.polarity for sentence in blob.sentences]
#     plt.figure(figsize=(10, 6))
#     sns.histplot(polarity_scores, bins=20, kde=True, color='skyblue', stat='density', linewidth=0)
#     plt.xlabel('Sentence Polarity', fontsize=14)
#     plt.ylabel('Density', fontsize=14)
#     plt.title(title, fontsize=16)
#     plt.grid(True, linestyle='--', alpha=0.7)
#     plt.legend(['Sentiment Distribution'], loc='upper right', fontsize=12)
#     # Enhance readability of x-axis tick labels
#     plt.xticks(np.arange(-1, 1.1, 0.2), fontsize=12)
#     plt.yticks(fontsize=12)
#     plt.tight_layout()
#     img_data = io.BytesIO()
#     plt.savefig(img_data, format='png')
#     plt.close()
#     img_data.seek(0)
#     encoded_img_data = base64.b64encode(img_data.read()).decode()
#     return encoded_img_data

# # Function to generate pattern sentiment analysis plot
# def generate_pattern_sentiment_plot(text):
#     scores = [pattern_sentiment(sentence)[0] for sentence in nltk.sent_tokenize(text)]
#     plt.figure(figsize=(10, 6))
#     sns.histplot(scores, bins=20, kde=True, color='salmon', stat='density', linewidth=0)
#     plt.xlabel('Sentence Polarity (Pattern)', fontsize=14)
#     plt.ylabel('Density', fontsize=14)
#     plt.title('Pattern Sentiment Analysis', fontsize=16)
#     plt.grid(True, linestyle='--', alpha=0.7)
#     plt.legend(['Sentiment Distribution'], loc='upper right', fontsize=12)
#     # Enhance readability of x-axis tick labels
#     plt.xticks(np.arange(-1, 1.1, 0.2), fontsize=12)
#     plt.yticks(fontsize=12)
#     plt.tight_layout()
#     img_data = io.BytesIO()
#     plt.savefig(img_data, format='png')
#     plt.close()
#     img_data.seek(0)
#     encoded_img_data = base64.b64encode(img_data.read()).decode()
#     return encoded_img_data

# # Function to perform text summarization using Sumy
# def generate_summary(text):
#     LANGUAGE = "english"
#     SENTENCES_COUNT = 3

#     parser = PlaintextParser.from_string(text, Tokenizer(LANGUAGE))
#     stemmer = Stemmer(LANGUAGE)

#     summarizer = LsaSummarizer(stemmer)
#     summarizer.stop_words = get_stop_words(LANGUAGE)

#     summary = summarizer(parser.document, SENTENCES_COUNT)
    
#     # Capitalize the first letter of each sentence
#     formatted_summary = []
#     for sentence in summary:
#         formatted_summary.append(str(sentence).capitalize())

#     return ' '.join(formatted_summary)

# # Function to get synonyms of a word
# def get_synonyms(word):
#     synonyms = set()
#     for synset in wordnet.synsets(word):
#         for lemma in synset.lemmas():
#             synonyms.add(lemma.name())
#     return list(synonyms)

# # Route to display the input form
# @app.route('/')
# def index():
#     return render_template('index.html')

# # Function to fetch data from a URL
# def fetch_url_data(url):
#     try:
#         response = requests.get(url)
#         response.raise_for_status()
#         return response.text
#     except requests.exceptions.RequestException as e:
#         print(f"Error: Failed to fetch data from the provided URL. {e}")
#         return None

# # Route to handle form submission and display analysis results
# @app.route('/analyze', methods=['POST'])
# def analyze_data():
#     try:
#         # Get user input from form
#         url = request.form['url']

#         # Fetch data from the provided website
#         html = fetch_url_data(url)
#         if html is None:
#             return "Error: Failed to fetch data from the provided URL."

#         # Parse HTML
#         soup = BeautifulSoup(html, 'html.parser')
#         paragraphs = [paragraph.get_text() for paragraph in soup.find_all('p')]
#         text = ' '.join(paragraphs)

#         # Clean the text
#         cleaned_text = clean_text(text)

#         # Tokenize cleaned text
#         tokens = nltk.word_tokenize(cleaned_text)

#         # Perform part-of-speech tagging
#         pos_tags = nltk.pos_tag(tokens)

#         # Count number of words
#         num_words = len(tokens)

#         # Count number of sentences without removing periods
#         num_sentences = len(nltk.sent_tokenize(text))

#         # Sentiment analysis using TextBlob
#         blob = TextBlob(cleaned_text)
#         sentiment_score = blob.sentiment.polarity

#         # Word cloud generation
#         wordcloud = WordCloud(width=800, height=400, background_color='white').generate(cleaned_text)

#         # TextBlob sentiment analysis
#         textblob_sentiment_plot = generate_sentiment_plot(blob, 'TextBlob Sentiment Analysis')

#         # Pattern sentiment analysis
#         pattern_sentiment_plot = generate_pattern_sentiment_plot(cleaned_text)

#         # Count POS tags
#         pos_counts = {}
#         for _, tag in pos_tags:
#             pos_counts[tag] = pos_counts.get(tag, 0) + 1

#         # Perform Named Entity Recognition (NER)
#         ner_results = perform_ner(cleaned_text)

#         # Extract keywords
#         keywords = extract_keywords(cleaned_text)

#         # Word frequency distribution
#         word_freq_table = Counter(tokens)

#         # Get top 10 most frequent words
#         top_10_words = dict(word_freq_table.most_common(10))

#         # Perform text summarization
#         summary = generate_summary(cleaned_text)

#         # Get synonyms for keywords
#         keyword_synonyms = {}
#         for keyword in keywords:
#             synonyms = get_synonyms(keyword)
#             keyword_synonyms[keyword] = synonyms

#         # Generate the sentiment plot image data
#         encoded_img_data = generate_sentiment_plot(blob, 'Sentiment Analysis')

#         # Render template with analysis results
#         return render_template('results.html', pos_counts=pos_counts, num_words=num_words, num_sentences=num_sentences,
#                                sentiment_score=sentiment_score,
#                                paragraphs=text,
#                                wordcloud_img=encoded_img_data,
#                                textblob_sentiment_plot=textblob_sentiment_plot,
#                                pattern_sentiment_plot=pattern_sentiment_plot,
#                                ner_results=ner_results,
#                                keywords=keywords,
#                                synonyms=keyword_synonyms,
#                                word_freq_table=top_10_words,
#                                summary=summary
#                                )

#     except Exception as e:
#         return f"An error occurred while processing the data: {e}"

# # Route to display admin login form
# @app.route('/admin-login', methods=['GET', 'POST'])
# def admin_login():
#     if request.method == 'POST':
#         username = request.form['username']
#         password = request.form['password']
#         # Validate admin credentials (replace with your authentication mechanism)
#         if username == 'admin' and password == 'admin_password':
#             session['admin_logged_in'] = True
#             return redirect(url_for('admin_page'))
#         else:
#             return render_template('admin_login.html', error='Invalid username or password')
#     return render_template('admin_login.html')

# # Route to display admin page with history results
# @app.route('/admin-page')
# def admin_page():
#     if not is_admin_logged_in():
#         return redirect(url_for('admin_login'))
#     try:
#         # Connect to the database
#         conn = connect_to_db()
#         cur = conn.cursor()

#         # Fetch history results from the database
#         cur.execute("SELECT url, paragraph, num_words, num_sentences, sentiment_score FROM new_table")
#         history_results = cur.fetchall()

#         # Close database connection
#         cur.close()
#         conn.close()

#         return render_template('admin_page.html', history_results=history_results)
#     except Exception as e:
#         return f"An error occurred while fetching history: {e}"

# # Route to handle admin logout
# @app.route('/admin-logout')
# def admin_logout():
#     session.pop('admin_logged_in', None)
#     return redirect(url_for('admin_login'))

# if __name__ == '__main__':
#     app.run(debug=True)
# from flask import Flask, render_template, request, redirect, url_for, session
# import nltk
# from nltk.corpus import stopwords
# import requests
# from bs4 import BeautifulSoup
# from textblob import TextBlob
# from wordcloud import WordCloud
# import string
# import base64
# import io
# import matplotlib.pyplot as plt
# import seaborn as sns
# from pattern.en import sentiment as pattern_sentiment
# import spacy
# import psycopg2
# import json
# from collections import Counter
# from sumy.parsers.plaintext import PlaintextParser
# from sumy.nlp.tokenizers import Tokenizer
# from sumy.summarizers.lsa import LsaSummarizer
# from sumy.nlp.stemmers import Stemmer
# from sumy.utils import get_stop_words

# # Load English language model for spaCy
# nlp = spacy.load('en_core_web_sm')

# # Initialize Flask app
# app = Flask(__name__)
# app.secret_key = 'your_secret_key'  # Change this to a random secret key

# # Download NLTK resources (run this only once)
# nltk.download('punkt')
# nltk.download('stopwords')

# # Database configuration
# DB_HOST = 'localhost'
# DB_NAME = 'postgres'
# DB_USER = 'postgres'
# DB_PASSWORD = 'Saurabh_Agrahari'

# # Function to establish database connection
# def connect_to_db():
#     conn = psycopg2.connect(host=DB_HOST, database=DB_NAME, user=DB_USER, password=DB_PASSWORD)
#     return conn

# # Function to check if admin is logged in
# def is_admin_logged_in():
#     return session.get('admin_logged_in', False)

# # Function to perform Named Entity Recognition (NER)
# def perform_ner(text):
#     doc = nlp(text)
#     entities = [(ent.text, ent.label_) for ent in doc.ents]
#     return entities

# # Function to perform Keyword Extraction
# def extract_keywords(text):
#     doc = nlp(text)
#     # Get nouns and proper nouns as keywords
#     keywords = [token.text for token in doc if token.pos_ in ['NOUN', 'PROPN']]
#     return keywords

# def clean_text(text):
#     # Remove punctuation except periods
#     text = ''.join([char for char in text if char not in string.punctuation or char == '.'])
#     # Convert text to lowercase
#     text = text.lower()
#     # Remove extra whitespace
#     text = ' '.join(text.split())
#     return text

# # Function to generate sentiment plot
# def generate_sentiment_plot(sentences, title):
#     polarity_scores = [TextBlob(s).sentiment.polarity for s in sentences]
#     plt.figure(figsize=(10, 6))
#     sns.histplot(polarity_scores, bins=20, kde=True)
#     plt.xlabel('Sentence Polarity')
#     plt.ylabel('Frequency')
#     plt.title(title)
#     plt.tight_layout()
#     img_data = io.BytesIO()
#     plt.savefig(img_data, format='png')
#     plt.close()
#     img_data.seek(0)
#     encoded_img_data = base64.b64encode(img_data.getvalue()).decode()
#     return encoded_img_data

# # Function to generate pattern sentiment analysis plot
# def generate_pattern_sentiment_plot(text):
#     scores = [pattern_sentiment(sentence)[0] for sentence in nltk.sent_tokenize(text)]
#     plt.figure(figsize=(10, 6))
#     sns.histplot(scores, bins=20, kde=True)
#     plt.xlabel('Sentence Polarity (Pattern)')
#     plt.ylabel('Frequency')
#     plt.title('Pattern Sentiment Analysis')
#     plt.tight_layout()
#     img_data = io.BytesIO()
#     plt.savefig(img_data, format='png')
#     plt.close()
#     img_data.seek(0)
#     encoded_img_data = base64.b64encode(img_data.getvalue()).decode()
#     return encoded_img_data

# # Function to perform text summarization using NLTK
# def generate_summary(text):
#     # Tokenize text into sentences
#     sentences = nltk.sent_tokenize(text)

#     # Tokenize words, remove punctuation, and convert to lowercase
#     tokenizer = nltk.RegexpTokenizer(r'\w+')
#     words = tokenizer.tokenize(text.lower())

#     # Remove stopwords
#     stop_words = set(stopwords.words('english'))
#     filtered_words = [word for word in words if word not in stop_words]

#     # Calculate word frequencies
#     word_freq = Counter(filtered_words)

#     # Calculate sentence scores based on word frequencies
#     sentence_scores = {}
#     for sentence in sentences:
#         for word in nltk.word_tokenize(sentence.lower()):
#             if word in word_freq:
#                 if len(sentence.split(' ')) < 30:  # Ensure sentences are not too long
#                     if sentence not in sentence_scores:
#                         sentence_scores[sentence] = word_freq[word]
#                     else:
#                         sentence_scores[sentence] += word_freq[word]

#     # Sort sentences by score
#     sorted_sentences = sorted(sentence_scores, key=sentence_scores.get, reverse=True)

#     # Generate summary by selecting top sentences
#     summary_sentences = sorted_sentences[:3]  # Select top 3 sentences as summary
#     summary = ' '.join(summary_sentences)

#     return summary

# # Route to display the input form
# @app.route('/')
# def index():
#     return render_template('index.html')

# # Route to handle form submission and display analysis results
# @app.route('/analyze', methods=['POST'])
# def analyze_data():
#     try:
#         # Get user input from form
#         url = request.form['url']

#         # Fetch data from the provided website
#         response = requests.get(url)
#         response.raise_for_status()  # Raise an exception for HTTP errors
#         html = response.text

#         # Parse HTML
#         soup = BeautifulSoup(html, 'html.parser')
#         paragraphs = [paragraph.get_text() for paragraph in soup.find_all('p')]
#         text = ' '.join(paragraphs)

#         # Clean the text
#         cleaned_text = clean_text(text)

#         # Tokenize cleaned text
#         tokens = nltk.word_tokenize(cleaned_text)

#         # Perform part-of-speech tagging
#         pos_tags = nltk.pos_tag(tokens)

#         # Count number of words
#         num_words = len(tokens)

#         # Count number of sentences without removing periods
#         num_sentences = len(nltk.sent_tokenize(text))

#         # Sentiment analysis using TextBlob
#         blob = TextBlob(cleaned_text)
#         sentiment_score = blob.sentiment.polarity

#         # Word cloud generation
#         wordcloud = WordCloud(width=800, height=400, background_color='white').generate(cleaned_text)

#         # TextBlob sentiment analysis
#         textblob_sentiment_plot = generate_sentiment_plot(blob, 'TextBlob Sentiment Analysis')

#         # Pattern sentiment analysis
#         pattern_sentiment_plot = generate_pattern_sentiment_plot(cleaned_text)

#         # Count POS tags
#         pos_counts = {}
#         for _, tag in pos_tags:
#             pos_counts[tag] = pos_counts.get(tag, 0) + 1

#         # Perform Named Entity Recognition (NER)
#         ner_results = perform_ner(cleaned_text)

#         # Extract keywords
#         keywords = extract_keywords(cleaned_text)

#         # Word frequency distribution
#         word_freq_table = Counter(tokens)

#         # Get top 10 most frequent words
#         top_10_words = dict(word_freq_table.most_common(10))

#         # Perform text summarization
#         summary = generate_summary(cleaned_text)

#         # Convert Word Cloud image to base64
#         img_data = io.BytesIO()
#         wordcloud.to_image().save(img_data, format='PNG')
#         img_data.seek(0)
#         encoded_img_data = base64.b64encode(img_data.getvalue()).decode()

#         # Render template with analysis results
#         return render_template('results.html', pos_counts=pos_counts, num_words=num_words, num_sentences=num_sentences,
#                                sentiment_score=sentiment_score,
#                                paragraphs=text,
#                                wordcloud_img=encoded_img_data,
#                                textblob_sentiment_plot=textblob_sentiment_plot,
#                                pattern_sentiment_plot=pattern_sentiment_plot,
#                                ner_results=ner_results,
#                                keywords=keywords,
#                                word_freq_table=top_10_words,
#                                summary=summary
#                                )

#     except requests.exceptions.RequestException as e:
#         return f"Error: Failed to fetch data from the provided URL. {e}"
#     except Exception as e:
#         return f"An error occurred while processing the data: {e}"


# # Route to display admin login form
# @app.route('/admin-login', methods=['GET', 'POST'])
# def admin_login():
#     if request.method == 'POST':
#         username = request.form['username']
#         password = request.form['password']
#         # Validate admin credentials (replace with your authentication mechanism)
#         if username == 'admin' and password == 'admin_password':
#             session['admin_logged_in'] = True
#             return redirect(url_for('admin_page'))
#         else:
#             return render_template('admin_login.html', error='Invalid username or password')
#     return render_template('admin_login.html')

# # Route to display admin page with history results
# @app.route('/admin-page')
# def admin_page():
#     if not is_admin_logged_in():
#         return redirect(url_for('admin_login'))
#     try:
#         # Connect to the database
#         conn = connect_to_db()
#         cur = conn.cursor()

#         # Fetch history results from the database
#         cur.execute("SELECT url, paragraph, num_words, num_sentences, sentiment_score FROM new_table")
#         history_results = cur.fetchall()

#         # Close database connection
#         cur.close()
#         conn.close()

#         return render_template('admin_page.html', history_results=history_results)
#     except Exception as e:
#         return f"An error occurred while fetching history: {e}"

# # Route to handle admin logout
# @app.route('/admin-logout')
# def admin_logout():
#     session.pop('admin_logged_in', None)
#     return redirect(url_for('admin_login'))

# if __name__ == '__main__':
#     app.run(debug=True)


# from flask import Flask, render_template, request, redirect, url_for, session, jsonify
# import nltk
# from nltk.corpus import stopwords
# import requests
# from bs4 import BeautifulSoup
# from textblob import TextBlob
# from wordcloud import WordCloud
# import string
# import base64
# import io
# import matplotlib.pyplot as plt
# import seaborn as sns
# from pattern.en import sentiment as pattern_sentiment
# import spacy
# import psycopg2
# import json
# from collections import Counter
# from sumy.parsers.plaintext import PlaintextParser
# from sumy.nlp.tokenizers import Tokenizer
# from sumy.summarizers.lsa import LsaSummarizer
# from sumy.nlp.stemmers import Stemmer
# from sumy.utils import get_stop_words
# import plotly.graph_objs as go

# # Load English language model for spaCy
# nlp = spacy.load('en_core_web_sm')

# # Initialize Flask app
# app = Flask(__name__)
# app.secret_key = 'your_secret_key'  # Change this to a random secret key

# # Download NLTK resources (run this only once)
# nltk.download('punkt')
# nltk.download('stopwords')

# # Database configuration
# DB_HOST = 'localhost'
# DB_NAME = 'postgres'
# DB_USER = 'postgres'
# DB_PASSWORD = 'Saurabh_Agrahari'

# # Function to establish database connection
# def connect_to_db():
#     conn = psycopg2.connect(host=DB_HOST, database=DB_NAME, user=DB_USER, password=DB_PASSWORD)
#     return conn

# # Function to check if admin is logged in
# def is_admin_logged_in():
#     return session.get('admin_logged_in', False)

# # Function to perform Named Entity Recognition (NER)
# def perform_ner(text):
#     doc = nlp(text)
#     entities = [(ent.text, ent.label_) for ent in doc.ents]
#     return entities

# # Function to perform Keyword Extraction
# def extract_keywords(text):
#     doc = nlp(text)
#     # Get nouns and proper nouns as keywords
#     keywords = [token.text for token in doc if token.pos_ in ['NOUN', 'PROPN']]
#     return keywords

# def clean_text(text):
#     # Remove punctuation except periods
#     text = ''.join([char for char in text if char not in string.punctuation or char == '.'])
#     # Convert text to lowercase
#     text = text.lower()
#     # Remove extra whitespace
#     text = ' '.join(text.split())
#     return text

# # Function to generate sentiment plot using Plotly
# def generate_sentiment_plot(sentences, title):
#     polarity_scores = [TextBlob(s).sentiment.polarity for s in sentences]
#     fig = go.Figure()
#     fig.add_trace(go.Histogram(x=polarity_scores, histnorm='probability density'))
#     fig.update_layout(title=title, xaxis_title="Sentence Polarity", yaxis_title="Probability Density")
#     plot_div = fig.to_html(full_html=False)
#     return plot_div

# # Function to generate pattern sentiment analysis plot using Plotly
# def generate_pattern_sentiment_plot(text):
#     scores = [pattern_sentiment(sentence)[0] for sentence in nltk.sent_tokenize(text)]
#     fig = go.Figure()
#     fig.add_trace(go.Histogram(x=scores, histnorm='probability density'))
#     fig.update_layout(title="Pattern Sentiment Analysis", xaxis_title="Sentence Polarity (Pattern)", yaxis_title="Probability Density")
#     plot_div = fig.to_html(full_html=False)
#     return plot_div

# # Function to perform text summarization using Sumy
# def generate_summary(text):
#     LANGUAGE = "english"
#     SENTENCES_COUNT = 3

#     parser = PlaintextParser.from_string(text, Tokenizer(LANGUAGE))
#     stemmer = Stemmer(LANGUAGE)

#     summarizer = LsaSummarizer(stemmer)
#     summarizer.stop_words = get_stop_words(LANGUAGE)

#     summary = summarizer(parser.document, SENTENCES_COUNT)
    
#     # Capitalize the first letter of each sentence
#     formatted_summary = []
#     for sentence in summary:
#         formatted_summary.append(str(sentence).capitalize())

#     return ' '.join(formatted_summary)

# # Route to display the input form
# @app.route('/')
# def index():
#     return render_template('index.html')

# # Route to handle form submission and display analysis results
# @app.route('/analyze', methods=['POST'])
# def analyze_data():
#     try:
#         # Get user input from form
#         url = request.form['url']

#         # Fetch data from the provided website
#         response = requests.get(url)
#         response.raise_for_status()  # Raise an exception for HTTP errors
#         html = response.text

#         # Parse HTML
#         soup = BeautifulSoup(html, 'html.parser')
#         paragraphs = [paragraph.get_text() for paragraph in soup.find_all('p')]
#         text = ' '.join(paragraphs)

#         # Clean the text
#         cleaned_text = clean_text(text)

#         # Tokenize cleaned text
#         tokens = nltk.word_tokenize(cleaned_text)

#         # Perform part-of-speech tagging
#         pos_tags = nltk.pos_tag(tokens)

#         # Count number of words
#         num_words = len(tokens)

#         # Count number of sentences without removing periods
#         num_sentences = len(nltk.sent_tokenize(text))

#         # Sentiment analysis using TextBlob
#         blob = TextBlob(cleaned_text)
#         sentiment_score = blob.sentiment.polarity

#         # Word cloud generation
#         wordcloud = WordCloud(width=800, height=400, background_color='white').generate(cleaned_text)

#         # TextBlob sentiment analysis plot
#         textblob_sentiment_plot = generate_sentiment_plot(nltk.sent_tokenize(cleaned_text), 'TextBlob Sentiment Analysis')

#         # Pattern sentiment analysis plot
#         pattern_sentiment_plot = generate_pattern_sentiment_plot(cleaned_text)

#         # Count POS tags
#         pos_counts = {}
#         for _, tag in pos_tags:
#             pos_counts[tag] = pos_counts.get(tag, 0) + 1

#         # Perform Named Entity Recognition (NER)
#         ner_results = perform_ner(cleaned_text)

#         # Extract keywords
#         keywords = extract_keywords(cleaned_text)

#         # Word frequency distribution
#         word_freq_table = Counter(tokens)

#         # Get top 10 most frequent words
#         top_10_words = dict(word_freq_table.most_common(10))

#         # Perform text summarization
#         summary = generate_summary(cleaned_text)

#         # Connect to the database
#         conn = connect_to_db()
#         cur = conn.cursor()

#         # Convert pos_counts dictionary to JSON string
#         pos_tags_json = json.dumps(pos_counts)

#         # Insert analysis results into the database
#         cur.execute("INSERT INTO new_table(url, paragraph, num_words, num_sentences, sentiment_score, pos_tags) VALUES (%s, %s, %s, %s, %s, %s)",
#                     (url, text, num_words, num_sentences, sentiment_score, pos_tags_json))

#         conn.commit()

#         # Close database connection
#         cur.close()
#         conn.close()

#         # Convert Word Cloud image to base64
#         img_data = io.BytesIO()
#         wordcloud.to_image().save(img_data, format='PNG')
#         img_data.seek(0)
#         encoded_img_data = base64.b64encode(img_data.read()).decode()

#         # Render template with analysis results
#         return render_template('results.html', pos_counts=pos_counts, num_words=num_words, num_sentences=num_sentences,
#                                sentiment_score=sentiment_score,
#                                paragraphs=text,
#                                wordcloud_img=encoded_img_data,
#                                textblob_sentiment_plot=textblob_sentiment_plot,
#                                pattern_sentiment_plot=pattern_sentiment_plot,
#                                ner_results=ner_results,
#                                keywords=keywords,
#                                word_freq_table=top_10_words,
#                                summary=summary
#                                )

#     except requests.exceptions.RequestException as e:
#         return f"Error: Failed to fetch data from the provided URL. {e}"
#     except Exception as e:
#         return f"An error occurred while processing the data: {e}"

# # Route to handle synonym retrieval
# @app.route('/get_synonyms', methods=['POST'])
# def get_synonyms():
#     try:
#         # Get the word entered by the user from the request
#         word = request.form['word']
        
#         # Make a request to a synonym API (e.g., Datamuse API)
#         response = requests.get(f'https://api.datamuse.com/words?rel_syn={word}')
#         data = response.json()
        
#         # Extract synonyms from the API response
#         synonyms = [item['word'] for item in data]
        
#         # Return the synonyms to the frontend
#         return jsonify({'synonyms': synonyms})
#     except Exception as e:
#         return jsonify({'error': str(e)})

# if __name__ == '__main__':
#     app.run(debug=True)
# from flask import Flask, render_template, request, jsonify
# import nltk
# from nltk.corpus import stopwords
# import requests
# from bs4 import BeautifulSoup
# from textblob import TextBlob
# from wordcloud import WordCloud
# import string
# import base64
# import io
# from pattern.en import sentiment as pattern_sentiment
# import spacy
# import psycopg2
# import json
# from collections import Counter
# from sumy.parsers.plaintext import PlaintextParser
# from sumy.nlp.tokenizers import Tokenizer
# from sumy.summarizers.lsa import LsaSummarizer
# from sumy.nlp.stemmers import Stemmer
# from sumy.utils import get_stop_words
# import plotly.graph_objs as go

# # Load English language model for spaCy
# nlp = spacy.load('en_core_web_sm')

# # Initialize Flask app
# app = Flask(__name__)
# app.secret_key = 'your_secret_key'  # Change this to a random secret key

# # Download NLTK resources (run this only once)
# nltk.download('punkt')
# nltk.download('stopwords')

# # Database configuration
# DB_HOST = 'localhost'
# DB_NAME = 'postgres'
# DB_USER = 'postgres'
# DB_PASSWORD = 'Saurabh_Agrahari'

# # Function to establish database connection
# def connect_to_db():
#     conn = psycopg2.connect(host=DB_HOST, database=DB_NAME, user=DB_USER, password=DB_PASSWORD)
#     return conn

# # Function to check if admin is logged in
# def is_admin_logged_in():
#     return session.get('admin_logged_in', False)

# # Function to perform Named Entity Recognition (NER)
# def perform_ner(text):
#     doc = nlp(text)
#     entities = [(ent.text, ent.label_) for ent in doc.ents]
#     return entities

# # Function to perform Keyword Extraction
# def extract_keywords(text):
#     doc = nlp(text)
#     # Get nouns and proper nouns as keywords
#     keywords = [token.text for token in doc if token.pos_ in ['NOUN', 'PROPN']]
#     return keywords

# def clean_text(text):
#     # Remove punctuation except periods
#     text = ''.join([char for char in text if char not in string.punctuation or char == '.'])
#     # Convert text to lowercase
#     text = text.lower()
#     # Remove extra whitespace
#     text = ' '.join(text.split())
#     return text

# # Function to generate sentiment plot using Plotly
# def generate_sentiment_plot(sentences, title):
#     polarity_scores = [TextBlob(s).sentiment.polarity for s in sentences]
#     fig = go.Figure()
#     fig.add_trace(go.Histogram(x=polarity_scores, histnorm='probability density'))
#     fig.update_layout(title=title, xaxis_title="Sentence Polarity", yaxis_title="Probability Density")
#     plot_div = fig.to_html(full_html=False)
#     return plot_div

# # Function to generate pattern sentiment analysis plot using Plotly
# def generate_pattern_sentiment_plot(text):
#     scores = [pattern_sentiment(sentence)[0] for sentence in nltk.sent_tokenize(text)]
#     fig = go.Figure()
#     fig.add_trace(go.Histogram(x=scores, histnorm='probability density'))
#     fig.update_layout(title="Pattern Sentiment Analysis", xaxis_title="Sentence Polarity (Pattern)", yaxis_title="Probability Density")
#     plot_div = fig.to_html(full_html=False)
#     return plot_div

# # Function to perform text summarization using Sumy
# def generate_summary(text):
#     LANGUAGE = "english"
#     SENTENCES_COUNT = 3

#     parser = PlaintextParser.from_string(text, Tokenizer(LANGUAGE))
#     stemmer = Stemmer(LANGUAGE)

#     summarizer = LsaSummarizer(stemmer)
#     summarizer.stop_words = get_stop_words(LANGUAGE)

#     summary = summarizer(parser.document, SENTENCES_COUNT)
    
#     # Capitalize the first letter of each sentence
#     formatted_summary = []
#     for sentence in summary:
#         formatted_summary.append(str(sentence).capitalize())

#     return ' '.join(formatted_summary)

# # Route to display the input form
# @app.route('/')
# def index():
#     return render_template('index.html')

# # Route to handle form submission and display analysis results
# @app.route('/analyze', methods=['POST'])
# def analyze_data():
#     try:
#         # Get user input from form
#         url = request.form['url']

#         # Fetch data from the provided website
#         response = requests.get(url)
#         response.raise_for_status()  # Raise an exception for HTTP errors
#         html = response.text

#         # Parse HTML
#         soup = BeautifulSoup(html, 'html.parser')
#         paragraphs = [paragraph.get_text() for paragraph in soup.find_all('p')]
#         text = ' '.join(paragraphs)

#         # Clean the text
#         cleaned_text = clean_text(text)

#         # Tokenize cleaned text
#         tokens = nltk.word_tokenize(cleaned_text)

#         # Perform part-of-speech tagging
#         pos_tags = nltk.pos_tag(tokens)

#         # Count number of words
#         num_words = len(tokens)

#         # Count number of sentences without removing periods
#         num_sentences = len(nltk.sent_tokenize(text))

#         # Sentiment analysis using TextBlob
#         blob = TextBlob(cleaned_text)
#         sentiment_score = blob.sentiment.polarity

#         # Word cloud generation
#         wordcloud = WordCloud(width=800, height=400, background_color='white').generate(cleaned_text)

#         # TextBlob sentiment analysis plot
#         textblob_sentiment_plot = generate_sentiment_plot(nltk.sent_tokenize(cleaned_text), 'TextBlob Sentiment Analysis')

#         # Pattern sentiment analysis plot
#         pattern_sentiment_plot = generate_pattern_sentiment_plot(cleaned_text)

#         # Count POS tags
#         pos_counts = {}
#         for _, tag in pos_tags:
#             pos_counts[tag] = pos_counts.get(tag, 0) + 1

#         # Perform Named Entity Recognition (NER)
#         ner_results = perform_ner(cleaned_text)

#         # Extract keywords
#         keywords = extract_keywords(cleaned_text)

#         # Word frequency distribution
#         word_freq_table = Counter(tokens)

#         # Get top 10 most frequent words
#         top_10_words = dict(word_freq_table.most_common(10))

#         # Perform text summarization
#         summary = generate_summary(cleaned_text)

#         # Connect to the database
#         conn = connect_to_db()
#         cur = conn.cursor()

#         # Convert pos_counts dictionary to JSON string
#         pos_tags_json = json.dumps(pos_counts)

#         # Insert analysis results into the database
#         cur.execute("INSERT INTO new_table(url, paragraph, num_words, num_sentences, sentiment_score, pos_tags) VALUES (%s, %s, %s, %s, %s, %s)",
#                     (url, text, num_words, num_sentences, sentiment_score, pos_tags_json))

#         conn.commit()

#         # Close database connection
#         cur.close()
#         conn.close()

#         # Convert Word Cloud image to base64
#         img_data = io.BytesIO()
#         wordcloud.to_image().save(img_data, format='PNG')
#         img_data.seek(0)
#         encoded_img_data = base64.b64encode(img_data.read()).decode()

#         # Render template with analysis results
#         return render_template('results.html', pos_counts=pos_counts, num_words=num_words, num_sentences=num_sentences,
#                                sentiment_score=sentiment_score,
#                                paragraphs=text,
#                                wordcloud_img=encoded_img_data,
#                                textblob_sentiment_plot=textblob_sentiment_plot,
#                                pattern_sentiment_plot=pattern_sentiment_plot,
#                                ner_results=ner_results,
#                                keywords=keywords,
#                                word_freq_table=top_10_words,
#                                summary=summary
#                                )

#     except requests.exceptions.RequestException as e:
#         return f"Error: Failed to fetch data from the provided URL. {e}"
#     except Exception as e:
#         return f"An error occurred while processing the data: {e}"

# # Route to handle synonym retrieval
# @app.route('/get_synonyms', methods=['POST'])
# def get_synonyms():
#     try:
#         # Get the word entered by the user from the request
#         word = request.form['word']
        
#         # Make a request to a synonym API (e.g., Datamuse API)
#         response = requests.get(f'https://api.datamuse.com/words?rel_syn={word}')
#         data = response.json()
        
#         # Extract synonyms from the API response
#         synonyms = [item['word'] for item in data]
        
#         # Return the synonyms to the frontend
#         return jsonify({'synonyms': synonyms})
#     except Exception as e:
#         return jsonify({'error': str(e)})

# if __name__ == '__main__':
#     app.run(debug=True)

from flask import Flask, render_template, request, redirect, url_for, session
from authlib.integrations.flask_client import OAuth
import nltk
from nltk.corpus import stopwords
import requests
from bs4 import BeautifulSoup
from textblob import TextBlob
from wordcloud import WordCloud
import string
import base64
import io
import matplotlib.pyplot as plt
import seaborn as sns
import spacy
import psycopg2
from collections import Counter
import json

# Load English language model for spaCy
nlp = spacy.load('en_core_web_sm')

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Change this to a random secret key
oauth=OAuth(app)
app.config['SECRET_KEY'] = "Saurabh"
app.config['GITHUB_CLIENT_ID'] = "7fea963be225cab0f44f"
app.config['GITHUB_CLIENT_SECRET'] = "9b9d4e994a94c55e1dd1fa651da32476c2e309f7"

github = oauth.register(
    name='github',
    client_id=app.config["GITHUB_CLIENT_ID"],
    client_secret=app.config["GITHUB_CLIENT_SECRET"],
    access_token_url='https://github.com/login/oauth/access_token',
    access_token_params=None,
    authorize_url='https://github.com/login/oauth/authorize',
    authorize_params=None,
    api_base_url='https://api.github.com/',
    client_kwargs={'scope': 'user:email'},
)



# Download NLTK resources (run this only once)
nltk.download('punkt')
nltk.download('stopwords')

# Database configuration
DB_HOST = 'localhost'
DB_NAME = 'postgres'
DB_USER = 'postgres'
DB_PASSWORD = 'Saurabh_Agrahari'

# Function to establish database connection
def connect_to_db():
    conn = psycopg2.connect(host=DB_HOST, database=DB_NAME, user=DB_USER, password=DB_PASSWORD)
    return conn

# Function to check if admin is logged in
def is_admin_logged_in():
    return session.get('admin_logged_in', False)

# Function to perform Named Entity Recognition (NER)
def perform_ner(text):
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    # Remove duplicate entities
    unique_entities = list(set(entities))
    return unique_entities

# def perform_ner(text):
#     doc = nlp(text)
#     entities = [(ent.text, ent.label_) for ent in doc.ents]
#     return entities

# Function to perform Keyword Extraction
def extract_keywords(text, min_freq=2, min_word_length=3):
    doc = nlp(text)
    # Get nouns and proper nouns as keywords
    keywords = [token.text for token in doc if token.pos_ in ['NOUN', 'PROPN','ADJ', 'VERB'] and len(token.text) >= min_word_length]
    # Count word frequencies
    word_freq = Counter(keywords)
    # Filter keywords based on frequency threshold
    keywords = [word for word, freq in word_freq.items() if freq >= min_freq]
    return keywords

# def extract_keywords(text):
#     doc = nlp(text)
#     # Get nouns and proper nouns as keywords
#     keywords = [token.text for token in doc if token.pos_ in ['NOUN', 'PROPN']]
#     return keywords

def clean_text(text):
    # Remove punctuation except periods
    text = ''.join([char for char in text if char not in string.punctuation or char == '.' or char == '"' or char == "'"])
    # Convert text to lowercase
   
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text
    # Convert text to lowercase
  
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text

# Function to generate sentiment plot
def generate_sentiment_plot(sentences, title):
    polarity_scores = [TextBlob(s).sentiment.polarity for s in sentences]
    plt.figure(figsize=(10, 6))
    sns.histplot(polarity_scores, bins=20, kde=True)
    plt.xlabel('Sentence Polarity')
    plt.ylabel('Frequency')
    plt.title(title)
    plt.tight_layout()
    img_data = io.BytesIO()
    plt.savefig(img_data, format='png')
    plt.close()
    img_data.seek(0)
    encoded_img_data = base64.b64encode(img_data.getvalue()).decode()
    return encoded_img_data

# Function to generate polarity distribution plot
def generate_polarity_plot(text):
    polarity_scores = [TextBlob(s).sentiment.polarity for s in nltk.sent_tokenize(text)]
    plt.figure(figsize=(10, 6))
    sns.histplot(polarity_scores, bins=20, kde=True)
    plt.xlabel('Sentence Polarity')
    plt.ylabel('Frequency')
    plt.title('Polarity Distribution')
    plt.tight_layout()
    img_data = io.BytesIO()
    plt.savefig(img_data, format='png')
    plt.close()
    img_data.seek(0)
    encoded_img_data = base64.b64encode(img_data.getvalue()).decode()
    return encoded_img_data

# Function to perform text summarization using NLTK
def generate_summary(text):
    # Tokenize text into sentences
    sentences = nltk.sent_tokenize(text)

    # Tokenize words, remove punctuation, and convert to lowercase
    tokenizer = nltk.RegexpTokenizer(r'\w+')
    words = tokenizer.tokenize(text.lower())

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word not in stop_words]

    # Calculate word frequencies
    word_freq = Counter(filtered_words)

    # Calculate sentence scores based on word frequencies
    sentence_scores = {}
    for sentence in sentences:
        for word in nltk.word_tokenize(sentence.lower()):
            if word in word_freq:
                if len(sentence.split(' ')) < 40:  # Ensure sentences are not too long
                    if sentence not in sentence_scores:
                        sentence_scores[sentence] = word_freq[word]
                    else:
                        sentence_scores[sentence] += word_freq[word]

    # Sort sentences by score
    sorted_sentences = sorted(sentence_scores, key=sentence_scores.get, reverse=True)

    # Generate summary by selecting top sentences and capitalize first letter after period
    summary_sentences = []
    for sentence in sorted_sentences[:10]:  # Select top 10 sentences as summary
        sentence = ' '.join([s.capitalize() if s.endswith('.') else s for s in sentence.split()])
        summary_sentences.append(sentence)

    summary = ' '.join(summary_sentences)

    return summary

# Route to display the input form
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle form submission and display analysis results
@app.route('/analyze', methods=['POST'])
def analyze_data():
    try:
        # Get user input from form
        url = request.form['url']

        # Fetch data from the provided website
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for HTTP errors
        html = response.text

        # Parse HTML
        
        # soup = BeautifulSoup(html, 'html.parser')
        # heading_container = soup.select('.headline.PyI5Q')  # Use select for multiple elements
        # heading = [heading.get_text() for heading in heading_container.find('bdi')]  # Use find for the first matching element
        soup = BeautifulSoup(html, 'html.parser')
        heading_container = soup.select('.headline.PyI5Q')
        headings =[ heading.find('bdi').get_text() for heading in heading_container]
        headings_text = ' '.join(headings)


        
        


        paragraphs_container = soup.select_one('.story-element.story-element-text')


        paragraphs = [paragraph.get_text() for paragraph in paragraphs_container.find_all('p')]
    
        text = ' '.join(paragraphs)

        # Clean the text
        cleaned_text = clean_text(text)

        # Tokenize cleaned text
        tokens = nltk.word_tokenize(cleaned_text)

        # Perform part-of-speech tagging
        pos_tags = nltk.pos_tag(tokens)

        # Count number of words
        num_words = len(tokens)

        # Count number of sentences without removing periods
        num_sentences = len(nltk.sent_tokenize(text))

        # Sentiment analysis using TextBlob
        blob = TextBlob(cleaned_text)
        sentiment_score = blob.sentiment.polarity

        # Word cloud generation
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(cleaned_text)

        # TextBlob sentiment analysis
        textblob_sentiment_plot = generate_sentiment_plot(blob, 'TextBlob Sentiment Analysis')

        # Polarity distribution plot
        polarity_plot = generate_polarity_plot(cleaned_text)

        # Count POS tags
        pos_counts = {'Noun': 0, 'Pronoun': 0, 'Verb': 0, 'Adverb': 0, 'Adjective': 0}
        for _, tag in pos_tags:
            if tag.startswith('NN'):  # Noun
                pos_counts['Noun'] += 1
            elif tag.startswith('PR'):  # Pronoun
                pos_counts['Pronoun'] += 1
            elif tag.startswith('VB'):  # Verb
                pos_counts['Verb'] += 1
            elif tag.startswith('RB'):  # Adverb
                pos_counts['Adverb'] += 1
            elif tag.startswith('JJ'):  # Adjective
                pos_counts['Adjective'] += 1
        # pos_counts = {}
        # for _, tag in pos_tags:
        #     pos_counts[tag] = pos_counts.get(tag, 0) + 1

        # Perform Named Entity Recognition (NER)
        ner_results = perform_ner(cleaned_text)

        # Extract keywords
        keywords = extract_keywords(cleaned_text)

        # Word frequency distribution
         # Calculate word frequency distribution excluding dots and commas
        word_freq_table = Counter(word for word in tokens if word not in ['.', ',','"'," ' ",","])

# Get top 10 most frequent words
        top_10_words = dict(word_freq_table.most_common(10))

        

        # Perform text summarization
        summary = generate_summary(cleaned_text)

        # Convert Word Cloud image to base64
        img_data = io.BytesIO()
        wordcloud.to_image().save(img_data, format='PNG')
        img_data.seek(0)
        encoded_img_data = base64.b64encode(img_data.getvalue()).decode()

        # Insert data into the database
        conn = connect_to_db()
        cur = conn.cursor()

        # Replace placeholders with actual variables containing data
        cur.execute("INSERT INTO new_table (url, paragraph, num_words, num_sentences, sentiment_score,pos_tags) VALUES (%s, %s, %s, %s, %s,%s)",
                    (url, text, num_words, num_sentences, sentiment_score,json.dumps(pos_counts)))

        # Commit the transaction
        conn.commit()

        # Close database connection
        cur.close()
        conn.close()

        # Render template with analysis results
        return render_template('results.html', pos_counts=pos_counts, num_words=num_words, num_sentences=num_sentences,
                               sentiment_score=sentiment_score,
                               paragraphs=text,
                               wordcloud_img=encoded_img_data,
                               textblob_sentiment_plot=textblob_sentiment_plot,
                               polarity_plot=polarity_plot,
                               ner_results=ner_results,
                               keywords=keywords,
                               word_freq_table=top_10_words,
                               summary=summary,
                               headings_text=headings_text
                               )

    except requests.exceptions.RequestException as e:
        return f"Error: Failed to fetch data from the provided URL. {e}"
    except Exception as e:
        return f"An error occurred while processing the data: {e}"



# Route to display admin login form
@app.route('/admin-login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        # Validate admin credentials (replace with your authentication mechanism)
        if username == 'admin' and password == 'admin_password':
            session['admin_logged_in'] = True
            return redirect(url_for('admin_page'))
        else:
            return render_template('admin_login.html', error='Invalid username or password')
    return render_template('admin_login.html')

# Route to display admin page with history results
@app.route('/admin-page')
def admin_page():
    if not is_admin_logged_in():
        return redirect(url_for('admin_login'))
    try:
        # Connect to the database
        conn = connect_to_db()
        cur = conn.cursor()

        # Fetch history results from the database
        cur.execute("SELECT url, paragraph, num_words, num_sentences, sentiment_score,pos_tags FROM new_table")
        history_results = cur.fetchall()

        # Close database connection
        cur.close()
        conn.close()

        return render_template('admin_page.html', history_results=history_results)
    except Exception as e:
        return f"An error occurred while fetching history: {e}"

# Route to handle admin logout
@app.route('/admin-logout')
def admin_logout():
    session.pop('admin_logged_in', None)
    return redirect(url_for('index'))


@app.route('/login1')
def login1():
    return render_template('login_page.html')
# Github login route
@app.route('/login/github')
def github_login():
    github = oauth.create_client('github')
    redirect_uri = url_for('github_authorize', _external=True)
    return github.authorize_redirect(redirect_uri)

# Github authorize route
@app.route('/login/github/authorize',methods =['GET'])
def github_authorize():
    github = oauth.create_client('github')
    token = github.authorize_access_token()
    session['github_token'] = token
    resp = github.get('user').json()
    print(f"\n{resp}\n")
    
    conn = connect_to_db()
    cur = conn.cursor()

    # Fetch history results from the database
        
 
        

    cur.execute("SELECT url, paragraph, num_words, num_sentences, sentiment_score,pos_tags FROM new_table")
    history_results = cur.fetchall()
    cur.close()
    conn.close()
    return render_template('admin_page.html', history_results=history_results)

    

    
# Logout route for GitHub
@app.route('/logout/github')
def github_logout():
    session.pop('github_token', None)
    session.pop('admin_logged_in', None)  # Add this line to clear the admin login session
    return redirect(url_for('index'))





if __name__ == '__main__':
    app.run(debug=True)
# from flask import Flask, render_template, request, redirect, url_for, session
# import nltk
# from nltk.corpus import stopwords
# import requests
# from bs4 import BeautifulSoup
# from textblob import TextBlob
# from wordcloud import WordCloud
# import string
# import base64
# import io
# import matplotlib.pyplot as plt
# import seaborn as sns
# import spacy
# import psycopg2
# from collections import Counter
# import json


# # Load English language model for spaCy
# nlp = spacy.load('en_core_web_sm')

# # Initialize Flask app
# app = Flask(__name__)
# app.secret_key = 'your_secret_key'  # Change this to a random secret key

# # Download NLTK resources (run this only once)
# nltk.download('punkt')
# nltk.download('stopwords')

# # Database configuration
# DB_HOST = 'localhost'
# DB_NAME = 'postgres'
# DB_USER = 'postgres'
# DB_PASSWORD = 'Saurabh_Agrahari'

# # Function to establish database connection
# def connect_to_db():
#     conn = psycopg2.connect(host=DB_HOST, database=DB_NAME, user=DB_USER, password=DB_PASSWORD)
#     return conn

# # Function to check if admin is logged in
# def is_admin_logged_in():
#     return session.get('admin_logged_in', False)

# # Function to perform Named Entity Recognition (NER)
# def perform_ner(text):
#     doc = nlp(text)
#     entities = [(ent.text, ent.label_) for ent in doc.ents]
#     return entities

# # Function to perform Keyword Extraction
# def extract_keywords(text):
#     doc = nlp(text)
#     # Get nouns and proper nouns as keywords
#     keywords = [token.text for token in doc if token.pos_ in ['NOUN', 'PROPN']]
#     return keywords

# def clean_text(text):
#     # Remove punctuation except periods
#     text = ''.join([char for char in text if char not in string.punctuation or char == '.'])
#     # Convert text to lowercase
#     text = text.lower()
#     # Remove extra whitespace
#     text = ' '.join(text.split())
#     return text

# # Function to generate sentiment plot
# def generate_sentiment_plot(sentences, title):
#     polarity_scores = [TextBlob(s).sentiment.polarity for s in sentences]
#     plt.figure(figsize=(10, 6))
#     sns.histplot(polarity_scores, bins=20, kde=True)
#     plt.xlabel('Sentence Polarity')
#     plt.ylabel('Frequency')
#     plt.title(title)
#     plt.tight_layout()
#     img_data = io.BytesIO()
#     plt.savefig(img_data, format='png')
#     plt.close()
#     img_data.seek(0)
#     encoded_img_data = base64.b64encode(img_data.getvalue()).decode()
#     return encoded_img_data

# # Function to generate polarity distribution plot
# def generate_polarity_plot(text):
#     polarity_scores = [TextBlob(s).sentiment.polarity for s in nltk.sent_tokenize(text)]
#     plt.figure(figsize=(10, 6))
#     sns.histplot(polarity_scores, bins=20, kde=True)
#     plt.xlabel('Sentence Polarity')
#     plt.ylabel('Frequency')
#     plt.title('Polarity Distribution')
#     plt.tight_layout()
#     img_data = io.BytesIO()
#     plt.savefig(img_data, format='png')
#     plt.close()
#     img_data.seek(0)
#     encoded_img_data = base64.b64encode(img_data.getvalue()).decode()
#     return encoded_img_data

# # Function to perform text summarization using NLTK
# def generate_summary(text):
#     # Tokenize text into sentences
#     sentences = nltk.sent_tokenize(text)

#     # Tokenize words, remove punctuation, and convert to lowercase
#     tokenizer = nltk.RegexpTokenizer(r'\w+')
#     words = tokenizer.tokenize(text.lower())

#     # Remove stopwords
#     stop_words = set(stopwords.words('english'))
#     filtered_words = [word for word in words if word not in stop_words]

#     # Calculate word frequencies
#     word_freq = Counter(filtered_words)

#     # Calculate sentence scores based on word frequencies
#     sentence_scores = {}
#     for sentence in sentences:
#         for word in nltk.word_tokenize(sentence.lower()):
#             if word in word_freq:
#                 if len(sentence.split(' ')) < 30:  # Ensure sentences are not too long
#                     if sentence not in sentence_scores:
#                         sentence_scores[sentence] = word_freq[word]
#                     else:
#                         sentence_scores[sentence] += word_freq[word]

#     # Sort sentences by score
#     sorted_sentences = sorted(sentence_scores, key=sentence_scores.get, reverse=True)

#     # Generate summary by selecting top sentences and capitalize first letter after period
#     summary_sentences = []
#     for sentence in sorted_sentences[:10]:  # Select top 10 sentences as summary
#         sentence = ' '.join([s.capitalize() if s.endswith('.') else s for s in sentence.split()])
#         summary_sentences.append(sentence)

#     summary = ' '.join(summary_sentences)

#     return summary

# # Route to display the input form
# @app.route('/')
# def index():
#     return render_template('index.html')

# # Route to handle form submission and display analysis results
# @app.route('/analyze', methods=['POST'])
# def analyze_data():
#     try:
#         # Get user input from form
#         url = request.form['url']

#         # Fetch data from the provided website
#         response = requests.get(url)
#         response.raise_for_status()  # Raise an exception for HTTP errors
#         html = response.text

#         # Parse HTML
#         soup = BeautifulSoup(html, 'html.parser')
#         paragraphs = [paragraph.get_text() for paragraph in soup.find_all('p')]
#         text = ' '.join(paragraphs)

#         # Clean the text
#         cleaned_text = clean_text(text)

#         # Tokenize cleaned text
#         tokens = nltk.word_tokenize(cleaned_text)

#         # Perform part-of-speech tagging
#         pos_tags = nltk.pos_tag(tokens)

#         # Count number of words
#         num_words = len(tokens)

#         # Count number of sentences without removing periods
#         num_sentences = len(nltk.sent_tokenize(text))

#         # Sentiment analysis using TextBlob
#         blob = TextBlob(cleaned_text)
#         sentiment_score = blob.sentiment.polarity

#         # Word cloud generation
#         wordcloud = WordCloud(width=800, height=400, background_color='white').generate(cleaned_text)

#         # TextBlob sentiment analysis
#         textblob_sentiment_plot = generate_sentiment_plot(blob, 'TextBlob Sentiment Analysis')

#         # Polarity distribution plot
#         polarity_plot = generate_polarity_plot(cleaned_text)

#         # Count POS tags
#         pos_counts = {}
#         for _, tag in pos_tags:
#             pos_counts[tag] = pos_counts.get(tag, 0) + 1

#         # Perform Named Entity Recognition (NER)
#         ner_results = perform_ner(cleaned_text)

#         # Extract keywords
#         keywords = extract_keywords(cleaned_text)

#         # Word frequency distribution
#         word_freq_table = Counter(tokens)

#         # Get top 10 most frequent words
#         top_10_words = dict(word_freq_table.most_common(10))

#         # Perform text summarization
#         summary = generate_summary(cleaned_text)

#         # Convert Word Cloud image to base64
#         img_data = io.BytesIO()
#         wordcloud.to_image().save(img_data, format='PNG')
#         img_data.seek(0)
#         encoded_img_data = base64.b64encode(img_data.getvalue()).decode()

#         # Insert analysis results into the database
#         conn = connect_to_db()
#         cur = conn.cursor()
#         cur.execute("""
#             INSERT INTO new_table (url, paragraph, num_words, num_sentences, sentiment_score,pos_tags)
#             VALUES (%s, %s, %s, %s, %s,%s)
#         """, (url, text, num_words, num_sentences, sentiment_score,json.dumps(pos_tags)))
#         conn.commit()
#         cur.close()
#         conn.close()

#         # Render template with analysis results
#         return render_template('results.html', pos_counts=pos_counts, num_words=num_words, num_sentences=num_sentences,
#                                sentiment_score=sentiment_score,
#                                paragraphs=text,
#                                wordcloud_img=encoded_img_data,
#                                textblob_sentiment_plot=textblob_sentiment_plot,
#                                polarity_plot=polarity_plot,
#                                ner_results=ner_results,
#                                keywords=keywords,
#                                word_freq_table=top_10_words,
#                                summary=summary
#                                )

#     except requests.exceptions.RequestException as e:
#         return f"Error: Failed to fetch data from the provided URL. {e}"
#     except Exception as e:
#         return f"An error occurred while processing the data: {e}"

# # Route to display admin login form
# @app.route('/admin-login', methods=['GET', 'POST'])
# def admin_login():
#     if request.method == 'POST':
#         username = request.form['username']
#         password = request.form['password']
#         # Validate admin credentials (replace with your authentication mechanism)
#         if username == 'admin' and password == 'admin_password':
#             session['admin_logged_in'] = True
#             return redirect(url_for('admin_page'))
#         else:
#             return render_template('admin_login.html', error='Invalid username or password')
#     return render_template('admin_login.html')

# # Route to display admin page with history results
# @app.route('/admin-page')
# def admin_page():
#     if not is_admin_logged_in():
#         return redirect(url_for('admin_login'))
#     try:
#         # Connect to the database
#         conn = connect_to_db()
#         cur = conn.cursor()

#         # Fetch history results from the database
#         cur.execute("SELECT url, paragraph, num_words, num_sentences, sentiment_score FROM new_table")
#         history_results = cur.fetchall()

#         # Close database connection
#         cur.close()
#         conn.close()

#         return render_template('admin_page.html', history_results=history_results)
#     except Exception as e:
#         return f"An error occurred while fetching history: {e}"

# # Route to handle admin logout
# @app.route('/admin-logout')
# def admin_logout():
#     session.pop('admin_logged_in', None)
#     return redirect(url_for('admin_login'))

# if __name__ == '__main__':
#     app.run(debug=True)

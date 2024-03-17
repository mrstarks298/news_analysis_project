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
import psycopg2
from collections import Counter
import json
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')

create_table_query = '''CREATE TABLE IF NOT EXISTS new_table (
    id SERIAL PRIMARY KEY,
    url TEXT,
    paragraph TEXT,
    num_words INTEGER,
    num_sentences INTEGER,
    sentiment_score FLOAT,
    pos_tags JSON
);'''

# Database configuration
DB_HOST = 'dpg-cnr8jnmn7f5s738b3b50-a'
DB_NAME = 'sitare'
DB_USER = 'saurabh'
DB_PASSWORD = 'IFfcZguN27jCfKQDn42wBIv9fGZ6LhQT'

# Function to establish database connection and create the table if it doesn't exist
def connect_to_db():
    conn = psycopg2.connect(host=DB_HOST, database=DB_NAME, user=DB_USER, password=DB_PASSWORD)
    cur = conn.cursor()
    cur.execute(create_table_query)
    conn.commit()
    return conn

# Function to perform Named Entity Recognition (NER)
def perform_ner(text):
    words = nltk.word_tokenize(text)
    pos_tags = nltk.pos_tag(words)
    return nltk.ne_chunk(pos_tags)

# Function to perform Keyword Extraction
def extract_keywords(text, min_freq=2, min_word_length=3):
    tokens = nltk.word_tokenize(text)
    # Get nouns, adjectives, verbs, and adverbs as keywords
    keywords = [word for (word, pos) in nltk.pos_tag(tokens) if pos.startswith(('NN', 'JJ', 'VB', 'RB')) and len(word) >= min_word_length]
    # Count word frequencies
    word_freq = Counter(keywords)
    # Filter keywords based on frequency threshold
    keywords = [word for word, freq in word_freq.items() if freq >= min_freq]
    return keywords

# Function to clean HTML text
def clean_text(html_text):
    soup = BeautifulSoup(html_text, 'html.parser')
    text = soup.get_text()
    text = ''.join([char for char in text if char not in string.punctuation or char == '.' or char == '"' or char == "'"])
    text = text.lower()
    text = ' '.join(text.split())
    return text

# Function to generate sentiment plot
def generate_sentiment_plot(sentences, title):
    sentences = [str(s) for s in sentences]
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

# Function to perform text summarization
def generate_summary(text):
    sentences = nltk.sent_tokenize(text)
    tokenizer = nltk.RegexpTokenizer(r'\w+')
    words = tokenizer.tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word not in stop_words]
    word_freq = Counter(filtered_words)
    sentence_scores = {}
    for sentence in sentences:
        for word in nltk.word_tokenize(sentence.lower()):
            if word in word_freq and len(sentence.split(' ')) < 40:
                if sentence not in sentence_scores:
                    sentence_scores[sentence] = word_freq[word]
                else:
                    sentence_scores[sentence] += word_freq[word]
    sorted_sentences = sorted(sentence_scores, key=sentence_scores.get, reverse=True)
    summary_sentences = []
    for sentence in sorted_sentences[:10]:
        sentence = ' '.join([s.capitalize() if s.endswith('.') else s for s in sentence.split()])
        summary_sentences.append(sentence)
    summary = ' '.join(summary_sentences)
    return summary

app = Flask(__name__)
app.secret_key = 'your_secret_key'

oauth = OAuth(app)
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

# Route to display the input form
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle form submission and display analysis results
@app.route('/analyze', methods=['POST'])
def analyze_data():
    try:
        url = request.form['url']
        response = requests.get(url)
        response.raise_for_status()
        html = response.text

        soup = BeautifulSoup(html, 'html.parser')
        heading_container = soup.select('.headline.PyI5Q')
        headings = [heading.find('bdi').get_text() for heading in heading_container]
        headings_text = ' '.join(headings)

        paragraphs_container = soup.select_one('.story-element.story-element-text')
        paragraphs = [paragraph.get_text() for paragraph in paragraphs_container.find_all('p')]
        text = ' '.join(paragraphs)
        cleaned_text = clean_text(text)
        tokens = nltk.word_tokenize(cleaned_text)
        num_words = len(tokens)
        num_sentences = len(nltk.sent_tokenize(text))

        blob = TextBlob(str(cleaned_text))
        sentiment_score = blob.sentiment.polarity

        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(cleaned_text)
        textblob_sentiment_plot = generate_sentiment_plot(blob.sentences, 'TextBlob Sentiment Analysis')
        polarity_plot = generate_polarity_plot(cleaned_text)
        keywords = extract_keywords(cleaned_text)
        summary = generate_summary(cleaned_text)

        img_data = io.BytesIO()
        wordcloud.to_image().save(img_data, format='PNG')
        img_data.seek(0)
        encoded_img_data = base64.b64encode(img_data.getvalue()).decode()

        conn = connect_to_db()
        cur = conn.cursor()

        cur.execute("INSERT INTO new_table (url, paragraph, num_words, num_sentences, sentiment_score) VALUES (%s, %s, %s, %s, %s)",
                    (url, str(text), num_words, num_sentences, sentiment_score))
        conn.commit()

        cur.close()
        conn.close()

        pos_counts = None  # Initialize pos_counts here
        return render_template('results.html', num_words=num_words, num_sentences=num_sentences,
                               sentiment_score=sentiment_score, paragraphs=text,
                               wordcloud_img=encoded_img_data,
                               textblob_sentiment_plot=textblob_sentiment_plot,
                               polarity_plot=polarity_plot,
                               keywords=keywords,
                               summary=summary,
                               headings_text=headings_text,
                               pos_counts=pos_counts)  # Pass pos_counts to template

    except requests.exceptions.RequestException as e:
        return f"Error: Failed to fetch data from the provided URL. {e}"
    except Exception as e:
        return f"An error occurred while processing the data: {e}"

# Routes for admin login, page, and logout
@app.route('/admin-login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username == 'admin' and password == 'admin_password':
            session['admin_logged_in'] = True
            return redirect(url_for('admin_page'))
        else:
            return render_template('admin_login.html', error='Invalid username or password')
    return render_template('admin_login.html')

@app.route('/admin-page')
def admin_page():
    if not session.get('admin_logged_in', False):
        return redirect(url_for('admin_login'))
    try:
        conn = connect_to_db()
        cur = conn.cursor()
        cur.execute("SELECT url, paragraph, num_words, num_sentences, sentiment_score FROM new_table")
        history_results = cur.fetchall()
        cur.close()
        conn.close()
        return render_template('admin_page.html', history_results=history_results)
    except Exception as e:
        return f"An error occurred while fetching history: {e}"

@app.route('/admin-logout')
def admin_logout():
    session.pop('admin_logged_in', None)
    return redirect(url_for('index'))

# GitHub login routes
@app.route('/login1')
def login1():
    return render_template('login_page.html')

@app.route('/login/github')
def github_login():
    github = oauth.create_client('github')
    redirect_uri = url_for('github_authorize', _external=True)
    return github.authorize_redirect(redirect_uri)

@app.route('/login/github/authorize',methods =['GET'])
def github_authorize():
    github = oauth.create_client('github')
    token = github.authorize_access_token()
    session['github_token'] = token
    resp = github.get('user').json()
    print(f"\n{resp}\n")
    
    conn = connect_to_db()
    cur = conn.cursor()

    cur.execute("SELECT url, paragraph, num_words, num_sentences, sentiment_score FROM new_table")
    history_results = cur.fetchall()
    cur.close()
    conn.close()
    return render_template('admin_page.html', history_results=history_results)

@app.route('/logout/github')
def github_logout():
    session.pop('github_token', None)
    session.pop('admin_logged_in', None)
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)

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
nltk.download  ('averaged_perceptron_tagger')
nltk.download("stopwords")
nltk.download("punkt")
nltk.download('universal_tagset')
create_table_query = '''CREATE TABLE IF NOT EXISTS new_table (
    id SERIAL PRIMARY KEY,
    url TEXT,
    paragraph TEXT,
    num_words INTEGER,
    num_sentences INTEGER,
    sentiment_score FLOAT,
    pos_tags JSON
);'''
nlp = nltk.download('maxent_ne_chunker')
nltk.download('maxent_ne_chunker')


# Database configuration
DB_HOST = 'dpg-cnr8jnmn7f5s738b3b50-a'

DB_NAME = 'sitare'
DB_USER = 'saurabh'
DB_PASSWORD ='IFfcZguN27jCfKQDn42wBIv9fGZ6LhQT'
# Function to establish database connection and create the table if it doesn't exist
def connect_to_db():
    conn = psycopg2.connect(host=DB_HOST, database=DB_NAME, user=DB_USER, password=DB_PASSWORD)
    cur = conn.cursor()
    cur.execute(create_table_query)  # 
    conn.commit()
    return conn
def perform_ner(text):
    words = nltk.word_tokenize(text)
    pos_tags = nltk.pos_tag(words)
    return nltk.ne_chunk(pos_tags)


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



# 


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


def clean_text(html_text):
    # Parse HTML content
    soup = BeautifulSoup(html_text, 'html.parser')
    # Extract text from HTML
    text = soup.get_text()
    # Remove punctuation except periods
    text = ''.join([char for char in text if char not in string.punctuation or char == '.' or char == '"' or char == "'"])
    # Convert text to lowercase
    text = text.lower()
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text

   
    

# Function to generate sentiment plot
def generate_sentiment_plot(sentences, title):
    # Ensure that each element in 'sentences' is a string
    sentences = [str(s) for s in sentences]
    
    # Calculate polarity scores for each sentence
    polarity_scores = [TextBlob(s).sentiment.polarity for s in sentences]
    
    # Generate histogram plot
    plt.figure(figsize=(10, 6))
    sns.histplot(polarity_scores, bins=20, kde=True)
    plt.xlabel('Sentence Polarity')
    plt.ylabel('Frequency')
    plt.title(title)
    plt.tight_layout()
    
    # Convert plot to base64-encoded image
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
    url=''
    try:
        # Get user input from form
        url = request.form['url']

        # Fetch data from the provided website
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for HTTP errors
        html = response.text

        # Parse HTML
        soup = BeautifulSoup(html, 'html.parser')
        heading_container = soup.select('.headline.PyI5Q')
        headings = [heading.find('bdi').get_text() for heading in heading_container]
        headings_text = ' '.join(headings)

        paragraphs_container = soup.select_one('.story-element.story-element-text')
        paragraphs = [paragraph.get_text() for paragraph in paragraphs_container.find_all('p')]
        text = ' '.join(paragraphs)

        # Clean the text
        cleaned_text = str(clean_text(text))

        # Debug message to verify the data type of cleaned_text
        print(f"Data type of cleaned_text: {type(cleaned_text)}")  # Add this line to print the type of 'cleaned_text'



        # Tokenize cleaned text
        tokens = nltk.word_tokenize(cleaned_text)

        # Perform part-of-speech tagging
        pos_tags = nltk.pos_tag(tokens)

        # Count number of words
        num_words = len(tokens)

        # Count number of sentences without removing periods
        num_sentences = len(nltk.sent_tokenize(text))
  
        cleaned_text = str(clean_text(text))

        

        # Debug message to verify the data type of cleaned_text
        print(f"Data type of cleaned_text: {type(cleaned_text)}")

        

        # # Debug message after converting to string
        print("Cleaned text after converting to string:", cleaned_text)

        # # Convert cleaned_text to UTF-8 string
        

        # # Create a TextBlob object
        print(f"Type of 'text' variable: {type(text)}")

      
        blob = TextBlob(str(cleaned_text))


        # # Perform sentiment analysisprint(f"Type of 'text' variable: {type(text)}")

        sentiment_score = blob.sentiment.polarity

        



        # Word cloud generation
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(cleaned_text)

        #TextBlob sentiment analysis
        # Perform sentiment analysis
        sentiment_score = blob.sentiment.polarity

        textblob_sentiment_plot = generate_sentiment_plot(blob.sentences, 'TextBlob Sentiment Analysis')


        # Polarity distribution plot
        polarity_plot = generate_polarity_plot(cleaned_text)

        # Count POS tags
        pos_counts = {'Noun': 0, 'Pronoun': 0, 'Verb': 0, 'Adverb': 0, 'Adjective': 0}
        print(pos_tags)
        print(type(pos_tags))
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
                    (url, str(text), num_words, num_sentences,sentiment_score ,json.dumps(pos_counts)))

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


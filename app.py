import os
import pdfplumber
from flask import Flask, request, render_template, redirect, url_for
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
import re

# Initialize the Flask app
app = Flask(__name__)

# Configure the upload folder
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure 'uploads' directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load English stop words from NLTK
stop_words = set(stopwords.words('english'))

def clean_text(text):
    """Function to clean text (lowercase, remove punctuation, stop words)"""
    text = text.lower()  # Lowercase
    text = re.sub(r'\[.*?\]', '', text)  # Remove text in brackets
    text = re.sub(r'\W', ' ', text)  # Remove non-word characters (punctuation)
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    text = ' '.join(word for word in text.split() if word not in stop_words) # Remove stop words
    return text

def read_pdf(file_path):
    """Function to extract text from a PDF file"""
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text

def calculate_similarity(resume_text, jd_text):
    """Function to calculate similarity between resume and JD"""
    # Clean the text
    cleaned_resume = clean_text(resume_text)
    cleaned_jd = clean_text(jd_text)
    
    # Put text into a list for the vectorizer
    text_corpus = [cleaned_resume, cleaned_jd]
    
    # TF-IDF Vectorizer
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(text_corpus)
    
    # Calculate Cosine Similarity
    # tfidf_matrix[0:1] = resume vector
    # tfidf_matrix[1:2] = jd vector
    similarity_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    
    # Convert the score to a percentage
    return similarity_score[0][0] * 100

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get the Job Description text from the form
        jd_text = request.form.get('jd')
        
        # Check for the resume file
        if 'resume' not in request.files:
            return redirect(request.url) # Reload page if no file
        
        resume_file = request.files['resume']
        
        # If the user did not select a file
        if resume_file.filename == '':
            return redirect(request.url)
        
        if resume_file and resume_file.filename.endswith('.pdf'):
            # Save the file temporarily to the 'uploads' folder
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], resume_file.filename)
            resume_file.save(file_path)
            
            # Extract text from the PDF
            resume_text = read_pdf(file_path)
            
            # Delete the file after extracting text
            os.remove(file_path)
            
            # Calculate the similarity
            match_score = calculate_similarity(resume_text, jd_text)
            
            # Show the 'index.html' page again, but this time with the score
            return render_template('index.html', score=match_score, jd=jd_text)
            
        else:
            # If the file is not a .pdf
            return render_template('index.html', error="Please upload a .pdf file only.")

    # For GET request (when the page first loads)
    return render_template('index.html', score=None)

if __name__ == '__main__':
    app.run(debug=True) # debug=True shows errors during development
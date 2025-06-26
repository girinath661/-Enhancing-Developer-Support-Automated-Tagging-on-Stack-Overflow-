# 🧠 Enhancing-Developer-Support-Automated-Tagging-on-Stack-Overflow
This project is a web-based application that predicts relevant StackOverflow tags for a given programming-related question. It includes data scraping, preprocessing, model training, and a Streamlit interface for real-time predictions.

---

## 📂 Project Structure

.
├── app.ipynb # Web scraping notebook
├── app1.ipynb # Data cleaning and preprocessing notebook
├── best.ipynb # Model training and evaluation notebook
├── best.py # Streamlit web application
├── cleaned_stackoverflow.csv # Cleaned dataset (generated from notebooks)
└── README.md # Project documentation

## ⚙️ Setup Instructions (Windows with Conda)

### 1. Clone the repository
       git clone https://github.com/girinath661/-Enhancing-Developer-Support-Automated-Tagging-on-Stack-Overflow-.git
       cd -Enhancing-Developer-Support-Automated-Tagging-on-Stack-Overflow



### 2. Create and activate a Conda environment
       conda create --name so-tag-predictor python=3.10 -y
       conda activate so-tag-predictor

### 3. Install dependencies
       pip install -r requirements.txt

### Create a requirements.txt file with the following contents:
    streamlit
    pandas
    scikit-learn
    beautifulsoup4
    requests
### 🚀 Workflow
### 🕸️ 1. Web Scraping
  - Open app.ipynb

  - Scrape StackOverflow questions and tags using requests and BeautifulSoup

  - Save the raw data (optional)

### 🧹 2. Data Cleaning & Preprocessing
  - Open app1.ipynb

  - Clean and preprocess the scraped data:

  - Handle missing values

  - Combine question title and text
    
  - Format tags into list format
    
  - Save the output as cleaned_stackoverflow.csv

### 🤖 3. Model Training
  - Run best.ipynb
  
  - Train a multi-label classification model using:
  
  - TfidfVectorizer
  
  - OneVsRestClassifier with LogisticRegression
  
  - Evaluate accuracy and save pipeline if needed

### 🌐 4. Launch Streamlit App
Once the cleaned dataset is ready:
     streamlit run best.py

The app:

  - Loads the cleaned dataset
  
  - Trains or loads the model
  
  - Accepts a programming question
  
  - Predicts the most relevant tags from top 50 frequent tags

# Streamlit file

import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from collections import Counter

# Load and preprocess data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("cleaned_stackoverflow.csv")
        df = df[df['tags'].apply(lambda x: isinstance(x, str))]
        df['text'] = df['text'].fillna('')
        df['question'] = df['question'].fillna('')
        df['combined_text'] = df['question'] + ' ' + df['text']
        df['tags'] = df['tags'].apply(lambda x: [tag.strip() for tag in x.split(',')])
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Train model (cached to avoid retraining)
@st.cache_resource
def train_model(_df):  # Note the underscore prefix for cached functions
    try:
        # Top 50 tags
        top_tags = [tag for tag, _ in Counter([t for tags in _df['tags'] for t in tags]).most_common(50)]
        _df['filtered_tags'] = _df['tags'].apply(lambda tags: [t for t in tags if t in top_tags])
        _df = _df[_df['filtered_tags'].map(len) > 0]
        
        # Prepare data
        mlb = MultiLabelBinarizer()
        Y = mlb.fit_transform(_df['filtered_tags'])
        X = _df['combined_text']
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
        
        # Train model
        model = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=10000, ngram_range=(1,2), stop_words='english')),
            ('clf', OneVsRestClassifier(LogisticRegression(solver='liblinear', max_iter=2000, C=10)))
        ])
        model.fit(X_train, Y_train)
        return model, mlb
    except Exception as e:
        st.error(f"Error training model: {e}")
        return None, None

# Main app
def main():
    st.title("StackOverflow Tag Predictor")
    st.write("Enter a programming question to predict relevant tags:")
    
    # Load data
    df = load_data()
    if df is None:
        st.stop()
    
    # Train model
    model, mlb = train_model(df)
    if model is None or mlb is None:
        st.stop()
    
    # Text input
    user_input = st.text_area("Input your question here:", 
                            "How to read a CSV file using pandas in Python?")
    
    # Prediction button
    if st.button("Predict Tags"):
        if user_input.strip():
            try:
                # Predict
                Y_pred = model.predict([user_input])
                tags = mlb.inverse_transform(Y_pred)
                
                # Display results
                st.subheader("Predicted Tags:")
                if tags[0]:
                    cols = st.columns(3)
                    for i, tag in enumerate(tags[0]):
                        cols[i%3].success(f"🏷️ {tag}")
                else:
                    st.warning("No relevant tags predicted.")
            except Exception as e:
                st.error(f"Prediction error: {e}")
        else:
            st.error("Please enter some text first!")

if __name__ == "__main__":
    main()

    
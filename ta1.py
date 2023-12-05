import json
import spacy
import string
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Path to the JSON file
gift_card_file = "/content/drive/MyDrive/AMPBA 2024S Group Projects/Text Analytics/Gift_Cards.json"

# Load the English model for spaCy
nlp = spacy.load("en_core_web_sm")

# List to store the overall rating and review text
ans = []

# Open and read the JSON file
with open(gift_card_file, 'r') as json_file:
    for line in json_file:
        data = json.loads(line)
        # Check if the keys 'overall' and 'reviewText' exist in the data
        if "overall" in data and "reviewText" in data:
            # Append the overall rating and review text to the list
            ans.append([data["overall"], data["reviewText"]])

# Function to process the text
def f(text):
    # Create a Doc object
    doc = nlp(text)
    # Return the processed text
    return " ".join([token.text.lower() for token in doc if token.text not in string.punctuation])

# List to store the processed text
subset = []

# Process the first 10000 and last 10000 reviews
subset = [f(i[1]) for i in ans[:10000]] + [f(i[1]) for i in ans[-10000:]]

# Create a TfidfVectorizer object
tfidfVectorizer = TfidfVectorizer()

# Transform the processed text to a TF-IDF matrix
tfidf_matrix = tfidfVectorizer.fit_transform(subset)

# Get the feature names
feature_names = tfidfVectorizer.get_feature_names_out()

# Create a DataFrame with the words and their IDF values
idf_df = pd.DataFrame({'word': feature_names, 'idf': tfidfVectorizer.idf_})

# Sort the DataFrame by the IDF values
idf_df_sorted = idf_df.sort_values(by='idf')

# Print the top 30 words based on IDF
print("Top 30 words based on IDF:")
print(idf_df_sorted.head(30))

# Print the bottom 30 words based on IDF
print("Bottom 30 words based on IDF:")
print(idf_df_sorted.tail(30))
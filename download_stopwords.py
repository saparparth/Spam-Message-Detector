import streamlit as st
import nltk
from nltk.corpus import stopwords

# Set the nltk_data path to a local folder inside the app
nltk_data_path = './nltk_data'
nltk.data.path.append(nltk_data_path)

# Check if stopwords are downloaded, if not, download them
try:
    stopwords.words('english')
except LookupError:
    st.info("Downloading NLTK stopwords...")
    nltk.download('stopwords', download_dir=nltk_data_path)

# Your normal Streamlit code continues here
st.write("Stopwords loaded successfully!")

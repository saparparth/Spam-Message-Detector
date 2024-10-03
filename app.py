import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.data.path.append('venv/lib/python3.11/site-packages/nltk')
ps = PorterStemmer()


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text, preserve_line=True)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


tfidf = pickle.load(open('vectorizer2.pkl', 'rb'))
model = pickle.load(open('mode_a.pkl', 'rb'))

st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):

    # 1. preprocess
    transformed_sms = transform_text(input_sms)
    # 2. vectorize
    # mnb.fit(X_train, y_train)

    vector_input = tfidf.transform([transformed_sms])
    # 3. predict
    result = model.predict(vector_input)[0]
    # 4. Display
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")

st.title("Message Examples To try")

# Create a list of spam message examples
spam_examples = [
    "Go until jurong point, crazy.. Available only in bugis n great world la e buffet... Cine there got amore wat...",
    "Ok lar... Joking wif u oni..."
    "Urgent! Your account has been compromised. Click here to verify.",
    "Did you catch the bus ? Are you frying an egg ? Did you make a tea? Are you eating your mom's left over dinner ? Do you feel my Love ?",
    "Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C's apply 08452810075over18's",
    "This is not a scam! Please send your bank details to claim your prize.",
    " Fair enough, anything going on?",
    "Yeah hopefully, if tyler can't do it I could maybe ask around a bit U don't know how stubborn I am. I didn't even want to go to the hospital. I kept telling Mark I'm not a weak sucker.",
    "Hospitals are for weak suckers.",
    "What you thinked about me. First time you saw me in class",
    "A gram usually runs like  &lt;#&gt; , a half eighth is smarter though and gets you almost a whole second gram for  &lt;#&gt;",
    "K fyi x has a ride early tomorrow morning but he's crashing at our place tonight"
]

# Display the spam examples in a box
# st.subheader("Example Spam Mess")
for message in spam_examples:
    st.markdown(f"- {message}")

# Add your name and GitHub link
st.markdown("---")
st.write("Created by: **Parth Sapar*")
st.write("[GitHub Repository](https://github.com/saparparth/Spam-Message-Detector)")

# Optionally, you can add some styling or other elements as needed

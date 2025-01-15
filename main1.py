import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.pipeline import make_pipeline
from transformers import BertTokenizer, TFBertForSequenceClassification
import tensorflow as tf
import nltk

# Step 1: Load the datasets
classified_tweets = pd.read_csv('Dataset/classified_tweets.csv')
cyberbullying_tweets = pd.read_csv('Dataset/cyberbullying_tweets.csv')
cyberbullying = pd.read_csv('Dataset/cyberbullying.csv')
cyberbullying_types = pd.read_csv('Dataset/CyberBullyingTypesDataset.csv')
cybertroll_dataset = pd.read_csv('Dataset/cybertroll_dataset.csv')
cyberTrollIEEE = pd.read_csv('Dataset/CyberTrollIEEE.csv')

# Step 2: Preprocessing the text (you can expand this as needed)
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # Lowercase, remove punctuation, and remove stopwords
    text = text.lower()
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

# Apply preprocessing to each dataset
datasets = [classified_tweets, cyberbullying_tweets, cyberbullying, cyberbullying_types, cybertroll_dataset, cyberTrollIEEE]
for dataset in datasets:
    dataset['text'] = dataset.iloc[:, 0].apply(preprocess_text)

# Step 3: Combine all the datasets (you can concatenate them, ensure 'cyberbullying' and 'class' labels are present)
all_data = pd.concat(datasets, ignore_index=True)

# Assuming the last column contains labels
all_data['label'] = all_data.iloc[:, -1]  # Modify this according to the actual label column

# Step 4: Fill NaN values with a placeholder (e.g., 'unknown' or 0)
all_data.fillna({'label': 0, 'text': 'unknown'}, inplace=True)

# Step 5: Split data into features and labels
X = all_data['text']
y = all_data['label']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 6: Vectorize the text data (TF-IDF or Word2Vec)
vectorizer = TfidfVectorizer(max_features=10000)

# Step 7: Train a Random Forest Classifier (or use SVM, Logistic Regression, or Neural Networks)
# You can experiment with different classifiers here
model = make_pipeline(vectorizer, RandomForestClassifier(n_estimators=100, random_state=42))

# Train the model
model.fit(X_train, y_train)

# Step 8: Evaluate the model
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Step 9: Deep Learning Model with BERT (Optional for more advanced NLP)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
X_train_enc = tokenizer(list(X_train), padding=True, truncation=True, return_tensors='tf')
X_test_enc = tokenizer(list(X_test), padding=True, truncation=True, return_tensors='tf')

# BERT model
bert_model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Train the model with BERT
bert_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), 
                   loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
                   metrics=['accuracy'])

# Fine-tune BERT
bert_model.fit(X_train_enc['input_ids'], y_train, epochs=3, batch_size=16)

# Step 10: Make predictions with BERT
def predict_bullying(text, model):
    # Tokenize the text before prediction
    tokenized_input = tokenizer([text], padding=True, truncation=True, return_tensors='tf')
    prediction = model.predict(tokenized_input['input_ids'])
    predicted_class = tf.argmax(prediction.logits, axis=1).numpy()[0]
    return "Bullying" if predicted_class == 1 else "Not Bullying"

# Example prediction
text_input = "I hate you, you're worthless!"
prediction = predict_bullying(text_input, bert_model)
print("Prediction:", prediction)

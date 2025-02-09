#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.pipeline import make_pipeline
from transformers import BertTokenizer, TFBertForSequenceClassification
import tensorflow as tf
import nltk


# In[2]:


classified_tweets = pd.read_csv('Dataset/classified_tweets.csv')
cyberbullying_tweets = pd.read_csv('Dataset/cyberbullying_tweets.csv')
cyberbullying = pd.read_csv('Dataset/cyberbullying.csv')
cyberbullying_types = pd.read_csv('Dataset/CyberBullyingTypesDataset.csv')
cybertroll_dataset = pd.read_csv('Dataset/cybertroll_dataset.csv')
cyberTrollIEEE = pd.read_csv('Dataset/CyberTrollIEEE.csv')


# In[3]:


# Step 2: Preprocessing the text (you can expand this as needed)
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))


# In[4]:


def preprocess_text(text):
    # Lowercase, remove punctuation, and remove stopwords
    text = text.lower()
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text


# In[5]:


datasets = [classified_tweets, cyberbullying_tweets, cyberbullying, cyberbullying_types, cybertroll_dataset, cyberTrollIEEE]
for dataset in datasets:
    dataset['text'] = dataset.iloc[:, 0].apply(preprocess_text)


# In[6]:


# Step 3: Combine all the datasets (you can concatenate them, ensure 'cyberbullying' and 'class' labels are present)
all_data = pd.concat(datasets, ignore_index=True)


# In[7]:


# Assuming the last column contains labels
all_data['label'] = all_data.iloc[:, -1]  # Modify this according to the actual label column


# In[8]:


# Step 4: Fill NaN values with a placeholder (e.g., 'unknown' or 0)
all_data.fillna({'label': 0, 'text': 'unknown'}, inplace=True)


# In[9]:


# Step 5: Split data into features and labels
X = all_data['text']
y = all_data['label']


# In[10]:


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[11]:


# Step 6: Vectorize the text data (TF-IDF or Word2Vec)
vectorizer = TfidfVectorizer(max_features=10000)


# In[12]:


# Step 7: Train a Random Forest Classifier (or use SVM, Logistic Regression, or Neural Networks)
# You can experiment with different classifiers here
model = make_pipeline(vectorizer, RandomForestClassifier(n_estimators=100, random_state=42))



# In[ ]:


# Train the model
model.fit(X_train, y_train)


# In[15]:


# Step 8: Evaluate the model
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))


# In[ ]:


from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification
import tensorflow as tf

# Use DistilBERT tokenizer and model
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
X_train_enc = tokenizer(list(X_train), padding=True, truncation=True, return_tensors='tf')
X_test_enc = tokenizer(list(X_test), padding=True, truncation=True, return_tensors='tf')




# In[ ]:


# Load DistilBERT model
distilbert_model = TFDistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)



# In[ ]:


# Compile the model
distilbert_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), 
                         loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
                         metrics=['accuracy'])



# In[ ]:


# Fine-tune DistilBERT with reduced epochs for faster training
distilbert_model.fit(X_train_enc['input_ids'], y_train, epochs=1, batch_size=8)


# In[ ]:


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


# In[ ]:


# Step 11: Save the Random Forest Classifier model
import joblib
joblib.dump(model, 'random_forest_model.pkl')
print("Random Forest model saved as 'random_forest_model.pkl'.")

# Step 12: Save the BERT model
bert_model.save_pretrained('./bert_model')
print("BERT model saved at './bert_model'.")


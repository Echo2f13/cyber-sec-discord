import time

# Measure time for each import
times = {}

start = time.time()
import pandas as pd
times['pandas'] = time.time() - start

start = time.time()
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
times['sklearn'] = time.time() - start

start = time.time()

from transformers import BertTokenizer, TFBertForSequenceClassification
times['transformers'] = time.time() - start

start = time.time()
import tensorflow as tf
times['tensorflow'] = time.time() - start

start = time.time()
import nltk
times['nltk'] = time.time() - start

# Output the timing results
print(times)
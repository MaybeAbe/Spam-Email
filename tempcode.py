import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import re #for regex
import nltk
from nltk.tokenize import word_tokenize    
from nltk.tokenize import RegexpTokenizer     

nltk.download('punkt_tab') #code wouldnt run without this?


from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
######################################
df = pd.read_csv('emails.csv')
print(df.head())
df.info()
print(df.isna().sum()) 
df = df.dropna()

print(df.duplicated().sum())
df.drop_duplicates(inplace=True) 
df.reset_index(inplace=True, drop=True)
###################################### Prepping Data
counts = df['spam'].value_counts()  # class count where 0 = not spam and 1 = spam

#percent of emails are spam (1) vs not spam (0)
percentages = df['spam'].value_counts() * 100

print("Counts:\n", counts)
print("\nPercentages:\n", percentages)

# character count of each email
df['length'] = df['text'].apply(len)
df['length'] = df['text'].str.len()

#Histogram of email lengths relative to chr
sns.histplot(df['length'], bins=50,)
plt.title("Distribution of Email Lengths")
plt.xlabel("Number of Characters")
plt.ylabel("Number of Emails")
plt.show()

#Boxplott email length by class to point out outliers to be aware of
sns.boxplot(x='spam', y='length', data=df)
plt.title("Email Length by Class")
plt.xlabel("Spam (1) / Not Spam (0)")
plt.ylabel("Number of Characters")
plt.show()


############################ Tokenizer/Normalization 
# For runtime reduction, i use a small set to test my desired tokenizers
sample = list(df["text"].sample(n=200, random_state=42))

#
regexp_tokenizer = RegexpTokenizer(r"\w+")  #regex symbol for chr and num


all_tokens = []
for email in sample:
    tokens = word_tokenize(email)
    all_tokens.append(tokens)

avg = sum(len(t) for t in all_tokens) / len(all_tokens)
print(f"NLTK word_tokenize— avg tokens per email: {avg:.1f}")

all_tokens = []
for email in sample:
    tokens = regexp_tokenizer.tokenize(email)
    all_tokens.append(tokens)

avg = sum(len(t) for t in all_tokens) / len(all_tokens)
print(f"RegexpTokenizer— avg tokens per email: {avg:.1f}")

#RegexpTokenizer wins! It is more concisely groups out punctuations 
########################################### TBA
X = df['text']
y = df['spam']

# 80/20 split of train 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, stratify=y, random_state=42)

import nltk
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.preprocessing import LabelEncoder

import logging

logger = logging.getLogger(__name__)

def perform_eda_and_pre(df):

    logger.info("================>>> Exploratory Data Analysis")
    logger.info("frequence Values in Label columns")
    print(df['label'].value_counts())

    logger.info("number of characters")
    df['num_characters'] = df['message'].apply(lambda x: len(x))

    logger.info("number of words")
    df['num_words'] = df['message'].apply(lambda x: len(nltk.word_tokenize(x)))

    logger.info("number of sentences")
    df['num_sentences'] = df['message'].apply(lambda x: len(nltk.sent_tokenize(x)))
    print(df.head())

    logger.info("Statistical Operation in all Data")
    Statis_Opera = df[['num_characters', 'num_words', 'num_sentences']].describe().round()
    print(Statis_Opera)

    logger.info("Statistical Operation (Ham)")
    Statis_Opera_Ham = df[df['label'] == 'ham'][['num_characters', 'num_words', 'num_sentences']].describe().round()
    print(Statis_Opera_Ham)

    logger.info("Statistical Operation (Spam)")
    Statis_Opera_Spam = df[df['label'] == 'spam'][['num_characters', 'num_words', 'num_sentences']].describe().round()
    print(Statis_Opera_Spam)

    encoder = LabelEncoder()
    df['label'] = encoder.fit_transform(df['label'])

    numerical_cols = ['label', 'num_characters', 'num_words', 'num_sentences']
    correlation_matrix = df[numerical_cols].corr()
    print("Correlation Matrix")
    print(correlation_matrix)

    logger.info("================>>> Text Preprocessing ")

    df['lower_message'] = df['message'].str.lower()

    df["tokenized_message"] = df['lower_message'].apply(word_tokenize)

    df['clean_tokens'] = df['tokenized_message'].apply(
        lambda tokens: [re.sub(r'[^a-zA-Z]', '', word) for word in tokens if word.isalpha()])

    stop_words = set(stopwords.words('english'))
    df['no_stopwords'] = df['clean_tokens'].apply(
        lambda tokens: [word for word in tokens if word not in stop_words]
    )

    stemmer = PorterStemmer()
    df['stemmed_tokens'] = df['no_stopwords'].apply(
        lambda tokens: [stemmer.stem(word) for word in tokens]
    )

    df['final_message'] = df['stemmed_tokens'].apply(lambda tokens: ' '.join(tokens))

    logger.info("✅ EDA and Preprocessing Done.")

    return df, correlation_matrix

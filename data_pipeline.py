import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk

import logging

logger = logging.getLogger(__name__)

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')


def prepare_data(file_path):

    pd.set_option('display.width', None)
    logger.info("Loading Data ....")
    df = pd.read_csv(file_path, encoding="latin1")

    print(df.head())
    df = df.rename(columns={'v1': 'label', 'v2': 'message'})
    print(df.head(20))

    logger.info("================>>> Basic Function")
    logger.info("number of rows and columns")
    print(df.shape)

    logger.info("Name of Columns")
    print(df.columns)

    logger.info("Information about Data")
    print(df.info())

    logger.info("Statistical Operation")
    print(df.describe(include='object'))

    logger.info("Data types in Data")
    print(df.dtypes)

    logger.info("Display the index Range")
    print(df.index)

    logger.info("Random rows in Dataset")
    print(df.sample(5))

    logger.info("================>>> Data Cleaning")
    logger.info("Missing Values")
    print(df.isnull().sum())

    logger.info("The Columns Unnamed: 2, Unnamed: 3, and Unnamed: 4 we don't have any information about them so we drop it .")
    df.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)

    logger.info("Number of Frequent Rows")
    print(df.duplicated().sum())  # 403

    logger.info("Remove Duplicates")
    df = df.drop_duplicates(keep='first')
    print(df.shape)

    logger.info(" There is Missing Values in Data")
    print(df.isnull().sum())

    sns.heatmap(df.isnull())
    plt.title('Missing Values after Cleaning')
    plt.show()

    return df

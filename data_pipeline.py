import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')


def prepare_data(file_path):

    pd.set_option('display.width', None)

    df = pd.read_csv(file_path, encoding="latin1")

    print(df.head())
    df = df.rename(columns={'v1': 'label', 'v2': 'message'})
    print(df.head(20))

    print("================>>> Basic Function")
    print("number of rows and columns")
    print(df.shape)

    print("Name of Columns")
    print(df.columns)

    print("Information about Data")
    print(df.info())

    print("Statistical Operation")
    print(df.describe(include='object'))

    print("Data types in Data")
    print(df.dtypes)

    print("Display the index Range")
    print(df.index)

    print("Random rows in Dataset")
    print(df.sample(5))

    print("================>>> Data Cleaning")
    print("Missing Values")
    print(df.isnull().sum())

    print("The Columns Unnamed: 2, Unnamed: 3, and Unnamed: 4 we don't have any information about them so we drop it .")
    df.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)

    print("Number of Frequent Rows")
    print(df.duplicated().sum())  # 403

    print("Remove Duplicates")
    df = df.drop_duplicates(keep='first')
    print(df.shape)

    print(" There is Missing Values in Data")
    print(df.isnull().sum())

    sns.heatmap(df.isnull())
    plt.title('Missing Values after Cleaning')
    plt.show()

    return df
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud


def run_visualizations(df, correlation_matrix):

    print("================>>> Visualization of Data")

    plt.figure(figsize=(6, 6))
    plt.pie(df['label'].value_counts(), labels=['Ham', 'Spam'], autopct='%1.0f%%')
    plt.title('Frequency Values in Label Columns')
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.histplot(df[df['label'] == 0][['num_characters']], label='Ham')
    sns.histplot(df[df['label'] == 1][['num_characters']], color='red', label='Spam')
    plt.title('Number of Characters in Label Columns')
    plt.xlabel("num_characters")
    plt.ylabel("Count")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.histplot(df[df['label'] == 0][['num_words']], label='Ham')
    sns.histplot(df[df['label'] == 1][['num_words']], color='orange', label='Spam')
    plt.title('Number of Words in Label Columns')
    plt.xlabel("num_words")
    plt.ylabel("Count")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()

    print("Generating Pairplot... please wait.")
    sns.pairplot(df[['label', 'num_characters', 'num_words', 'num_sentences']], hue='label')
    plt.show()

    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Correlation Between Numerical Features', fontsize=14)
    plt.show()

    print("Generating WordCloud...")
    all_text = " ".join(df['final_message'])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title("WordCloud of Preprocessed Messages", fontsize=16)
    plt.axis("off")
    plt.show()

    print(" All Visualizations displayed.")
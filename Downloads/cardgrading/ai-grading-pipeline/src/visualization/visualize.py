import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def visualize_edge_gradient_distribution(data_file):
    df = pd.read_csv(data_file)
    plt.figure(figsize=(8, 4))
    sns.histplot(df['edge_gradient'].dropna(), bins=30, kde=True)
    plt.title('Edge Gradient Distribution')
    plt.xlabel('Edge Gradient')
    plt.ylabel('Frequency')
    plt.show()

def visualize_feature_comparison(df1, df2, feature):
    plt.figure(figsize=(10, 5))
    sns.boxplot(data=[df1[feature], df2[feature]], palette="Set2")
    plt.xticks([0, 1], ['Dataset 1', 'Dataset 2'])
    plt.title(f'Comparison of {feature}')
    plt.ylabel(feature)
    plt.show()
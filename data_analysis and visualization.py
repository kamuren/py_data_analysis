import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

def load_and_prepare_data():
    try:
        # Load the Iris dataset
        iris = load_iris()
        df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
        df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def explore_data(df):
    print("\n=== Data Exploration ===")
    print("\nFirst 5 rows:")
    print(df.head())
    
    print("\nDataset Info:")
    print(df.info())
    
    print("\nBasic Statistics:")
    print(df.describe())

def analyze_data(df):
    print("\n=== Data Analysis ===")
    print("\nMean values by species:")
    print(df.groupby('species').mean())

def create_visualizations(df):
    # Set the style for better-looking graphs
    sns.set_style("whitegrid")
    
    # 1. Line plot
    plt.figure(figsize=(10, 6))
    plt.plot(df.index[:50], df['sepal length (cm)'][:50], label='Sepal Length')
    plt.plot(df.index[:50], df['petal length (cm)'][:50], label='Petal Length')
    plt.title('Sepal and Petal Length Trends')
    plt.xlabel('Sample Index')
    plt.ylabel('Length (cm)')
    plt.legend()
    plt.savefig('line_plot.png')
    plt.close()

    # 2. Bar plot
    plt.figure(figsize=(10, 6))
    species_means = df.groupby('species')['sepal length (cm)'].mean()
    species_means.plot(kind='bar')
    plt.title('Average Sepal Length by Species')
    plt.xlabel('Species')
    plt.ylabel('Sepal Length (cm)')
    plt.tight_layout()
    plt.savefig('bar_plot.png')
    plt.close()

    # 3. Histogram
    plt.figure(figsize=(10, 6))
    plt.hist(df['petal width (cm)'], bins=20)
    plt.title('Distribution of Petal Width')
    plt.xlabel('Petal Width (cm)')
    plt.ylabel('Frequency')
    plt.savefig('histogram.png')
    plt.close()

    # 4. Scatter plot
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='sepal length (cm)', y='petal length (cm)', 
                    hue='species', style='species')
    plt.title('Sepal Length vs Petal Length by Species')
    plt.savefig('scatter_plot.png')
    plt.close()

def main():
    # Load data
    df = load_and_prepare_data()
    if df is None:
        return
    
    # Perform analysis
    explore_data(df)
    analyze_data(df)
    create_visualizations(df)
    
    print("\nAnalysis complete! Visualization files have been saved.")

if __name__ == "__main__":
    main()
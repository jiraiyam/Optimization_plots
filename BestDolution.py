import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import os
st.set_page_config(page_title="Statistical Plots BestSolution", layout="wide")

@st.cache_data
def load_data(uploaded_file):
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        return data
    return None

def main():
    st.title("Statistical Plots BestSolution Visualization App")

    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])

    if uploaded_file is not None:
        df = load_data(uploaded_file)
        file_name_without_extension = os.path.splitext(uploaded_file.name)[0]

        st.write(f"Uploaded file name: {file_name_without_extension}")

        if df is not None:
            st.sidebar.title("Navigation")
            app_mode = st.sidebar.selectbox("Choose the plot type", ["Basic Plots", "Advanced Plots", "Statistical Plots"])

            if app_mode == "Basic Plots":
                basic_plots(df, file_name_without_extension)
            elif app_mode == "Advanced Plots":
                advanced_plots(df ,  file_name_without_extension)
            elif app_mode == "Statistical Plots":
                statistical_plots(df, file_name_without_extension)
        else:
            st.info("Please upload a CSV file to begin.")

def basic_plots(df, file_name):
    st.header("Basic Plots")

    # Box Plot
    plt.figure(figsize=(12, 8))
    sns.boxplot(data=df)
    plt.title(f'Boxplot of Algorithm Performance for  {file_name}', fontsize=16, fontweight='bold')
    plt.xlabel('Algorithm', fontsize=14)
    plt.ylabel('Performance Metrics', fontsize=14)
    plt.xticks(rotation=90)
    plt.tight_layout()
    st.pyplot(plt)

    st.pyplot(plt)
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df_melted, x='Algorithm', y='Value', palette='pastel')
    sns.swarmplot(data=df_melted, x='Algorithm', y='Value', color='black', alpha=0.6)
    plt.title(f'Boxplot with Swarm for {file_name}')
    plt.xticks(rotation=90)
    plt.show()
    st.pyplot(plt)


    # Custom Box Plot with Color Palette
    plt.figure(figsize=(12, 8))
    palette = sns.color_palette("Set2", len(df.columns))
    sns.boxplot(data=df, palette=palette, linewidth=2, width=0.6)
    plt.grid(True, linestyle='--', alpha=0.7)
    sns.swarmplot(data=df, color=".25", alpha=0.8, size=3)
    plt.title(f'Boxplot of Algorithm Performance Metrics for {file_name}', fontsize=16, fontweight='bold')
    plt.xlabel('Algorithm', fontsize=14, fontweight='bold')
    plt.ylabel('Performance Metrics', fontsize=14, fontweight='bold')
    plt.xticks(rotation=90, fontsize=12)
    plt.tight_layout()
    st.pyplot(plt)

    # Performance Comparison of Optimization Algorithms
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df)
    plt.xticks(rotation=45, fontsize=12)
    plt.title(f"Performance Comparison of Optimization Algorithms for {file_name}", fontsize=14, fontweight='bold')
    plt.ylabel("Performance Metric", fontsize=12)
    plt.xlabel("Optimization Algorithm", fontsize=12)
    plt.tight_layout()
    st.pyplot(plt)

    # Violin Plot
    plt.figure(figsize=(12, 6))
    sns.violinplot(data=df)
    plt.xticks(rotation=45, fontsize=12)
    plt.title(f"Distribution of Optimization Algorithm Metrics for {file_name}", fontsize=14, fontweight='bold')
    plt.ylabel("Performance Metric", fontsize=12)
    plt.xlabel("Optimization Algorithm", fontsize=12)
    plt.tight_layout()
    st.pyplot(plt)

    # Swarm Plot
    plt.figure(figsize=(12, 6))
    sns.swarmplot(data=df)
    plt.xticks(rotation=45, fontsize=12)
    plt.title(f"Swarm Plot of Optimization Algorithm Metrics for {file_name} ", fontsize=14, fontweight='bold')
    plt.ylabel("Performance Metric", fontsize=12)
    plt.xlabel("Optimization Algorithm", fontsize=12)
    plt.tight_layout()
    st.pyplot(plt)

    # Strip Plot
    plt.figure(figsize=(12, 6))
    sns.stripplot(data=df)
    plt.xticks(rotation=45, fontsize=12)
    plt.title(f"Strip Plot of Optimization Algorithm Metrics for {file_name}", fontsize=14, fontweight='bold')
    plt.ylabel("Performance Metric", fontsize=12)
    plt.xlabel("Optimization Algorithm", fontsize=12)
    plt.tight_layout()
    st.pyplot(plt)



    df_melted = df.melt(var_name='Algorithm', value_name='Value')
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df_melted, x='Algorithm', y='Value')
    plt.title(f'Boxplot of Optimization Algorithms for {file_name}')
    plt.xticks(rotation=90)
    plt.show()

    st.pyplot(plt)

    g = sns.FacetGrid(df_melted, col='Algorithm', col_wrap=4, height=4)
    g.map(sns.boxplot, 'Value', palette='Set2')
    g.add_legend()
    g.fig.suptitle(f'Facet Grid of Boxplots for Each Algorithm for {file_name}', y=1.05)
    plt.show()
    st.pyplot(plt)


    plt.figure(figsize=(12, 6))
    sns.boxenplot(data=df_melted, x='Algorithm', y='Value', palette='Set2')
    plt.title(f'Boxen Plot of Optimization Algorithms for {file_name}')
    plt.xticks(rotation=90)
    plt.show()

    st.pyplot(plt)
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df_melted, x='Algorithm', y='Value', color='lightgray')
    sns.swarmplot(data=df_melted, x='Algorithm', y='Value', color='black', alpha=0.6)
    plt.title(f'Boxplot with Swarm Plot of Optimization Algorithms for {file_name}')
    plt.xticks(rotation=90)
    plt.show()
    st.pyplot(plt)

    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df_melted, x='Algorithm', y='Value', color='lightgray')
    sns.stripplot(data=df_melted, x='Algorithm', y='Value', color='black', jitter=True, alpha=0.6)
    plt.title(f'Boxplot with Jittered Points for {file_name}')
    plt.xticks(rotation=90)
    plt.show()

    st.pyplot(plt)
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df_melted, x='Algorithm', y='Value', palette='pastel')
    sns.swarmplot(data=df_melted, x='Algorithm', y='Value', color='black', alpha=0.6)
    plt.title(f'Boxplot with Swarm for {file_name}')
    plt.xticks(rotation=90)
    plt.show()
    st.pyplot(plt)

    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df_melted, x='Value', y='Algorithm', orient='h')
    plt.title(f'Horizontal Boxplot of Optimization Algorithms for {file_name}')
    plt.show()
    st.pyplot(plt)

    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df_melted, x='Value', y='Algorithm', color='lightgray')
    sns.swarmplot(data=df_melted, x='Value', y='Algorithm', color='black', alpha=0.6)
    plt.title(f'Horizontal Boxplot with Swarm Overlay for {file_name}')
    plt.show()

    st.pyplot(plt)

def advanced_plots(df , file_name):
    # Melt the DataFrame for seaborn
    df_melted = df.melt(var_name='Algorithm', value_name='Value')

    # Set up the subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Boxplots of Optimization Algorithms for {file_name}', fontsize=16)

    # Basic Boxplot
    sns.boxplot(data=df_melted, x='Algorithm', y='Value', ax=axes[0, 0])
    axes[0, 0].set_title('Basic Boxplot')
    axes[0, 0].tick_params(axis='x', rotation=90)

    # Boxplot with Swarm Plot
    sns.boxplot(data=df_melted, x='Algorithm', y='Value', color='lightgray', ax=axes[0, 1])
    sns.swarmplot(data=df_melted, x='Algorithm', y='Value', color='black', alpha=0.6, ax=axes[0, 1])
    axes[0, 1].set_title(f'Boxplot with Swarm Plot for {file_name}')
    axes[0, 1].tick_params(axis='x', rotation=90)

    # Horizontal Boxplot
    sns.boxplot(data=df_melted, x='Value', y='Algorithm', orient='h', ax=axes[1, 0])
    axes[1, 0].set_title(f'Horizontal Boxplot for {file_name}')

    # Customized Boxplot
    sns.boxplot(data=df_melted, x='Algorithm', y='Value', palette='Set2', ax=axes[1, 1])
    axes[1, 1].set_title(f'Customized Boxplot for {file_name}')
    axes[1, 1].tick_params(axis='x', rotation=90)

    # Adjust layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust title position
    plt.show()

    st.pyplot(plt)

    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df_melted, x='Algorithm', y='Value', color='lightgray')
    mean_values = df.mean()
    plt.scatter(df.columns, mean_values, color='red', label='Mean', zorder=10)
    plt.title(f'Boxplot with Mean Points Overlay for {file_name}')
    plt.xticks(rotation=90)
    plt.legend()
    plt.show()

    st.pyplot(plt)

    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df_melted, x='Algorithm', y='Value', color='lightgray', showfliers=False)
    sns.swarmplot(data=df_melted, x='Algorithm', y='Value', color='red', alpha=0.6, size=3)
    plt.title(f'Boxplot with Emphasized Outliers for {file_name}')
    plt.xticks(rotation=90)
    plt.show()
    st.pyplot(plt)

    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df_melted, x='Algorithm', y='Value', color='lightgray')
    for algorithm in df.columns:
        plt.plot(df.index, df[algorithm], marker='o', label=algorithm)
    plt.title(f'Boxplot with Line Plots of Each Algorithm for {file_name}')
    plt.xlabel('Algorithm')
    plt.ylabel('Values')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), title='Algorithms', fontsize='small')
    plt.xticks(rotation=90)
    plt.show()

    st.pyplot(plt)

    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df_melted, x='Algorithm', y='Value', color='lightgray')
    mean_values = df.mean()
    plt.plot(df.columns, mean_values, marker='o', color='red', label='Mean Values')
    plt.title(f'Boxplot with Mean Line Overlay for {file_name}')
    plt.xlabel('Algorithm')
    plt.ylabel('Values')
    plt.legend()
    plt.xticks(rotation=90)
    plt.show()

    st.pyplot(plt)

def statistical_plots(df , file_name) :
    df_melted = df.melt(var_name='Algorithm', value_name='Value')

    plt.figure(figsize=(12, 6))
    for algorithm in df.columns:
        moving_average = df[algorithm].rolling(window=5).mean()
        plt.plot(moving_average.index, moving_average, marker='o', label=f'MA - {algorithm}')
    plt.title(f'Line Plot with Moving Average for {file_name}')
    plt.xlabel('Index')
    plt.ylabel('Moving Average Values')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), title='Algorithms', fontsize='small')
    plt.xticks(rotation=90)
    plt.show()
    st.pyplot(plt)

    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df_melted, x='Algorithm', y='Value', color='lightgray')

    for algorithm in df.columns:
        mean = df[algorithm].mean()
        median = df[algorithm].median()
        plt.plot(df.index, df[algorithm], marker='o', label=algorithm)
        plt.axhline(mean, color='red', linestyle='--', label=f'Mean {algorithm}' if algorithm == df.columns[0] else "")
        plt.axhline(median, color='blue', linestyle=':',
                    label=f'Median {algorithm}' if algorithm == df.columns[0] else "")

    plt.title(f'Boxplot with Mean and Median Lines for {file_name}')
    plt.xlabel('Algorithm')
    plt.ylabel('Values')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), title='Statistics', fontsize='small')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()
    st.pyplot(plt)


    plt.figure(figsize=(12, 6))
    sns.violinplot(data=df_melted, x='Algorithm', y='Value', inner=None, color='lightgray')

    # Plot mean and median
    for algorithm in df.columns:
        mean = df[algorithm].mean()
        median = df[algorithm].median()
        plt.plot(df.index, df[algorithm], marker='o', label=algorithm)
        plt.axhline(mean, color='red', linestyle='--', label=f'Mean {algorithm}' if algorithm == df.columns[0] else "")
        plt.axhline(median, color='blue', linestyle=':',
                    label=f'Median {algorithm}' if algorithm == df.columns[0] else "")

    plt.title(f'Violin Plot with Mean and Median Lines for {file_name}')
    plt.xlabel('Algorithm')
    plt.ylabel('Values')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), title='Statistics', fontsize='small')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

    st.pyplot(plt)

    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df_melted, x='Algorithm', y='Value', color='lightgray')
    sns.swarmplot(data=df_melted, x='Algorithm', y='Value', color='black', alpha=0.6)

    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df_melted, x='Algorithm', y='Value', color='lightgray')
    plt.title(f'Box Plot with Mean and Median Annotations for {file_name}')
    plt.xlabel('Algorithm')
    plt.ylabel('Values')
    plt.xticks(rotation=90)

    for algorithm in df.columns:
        mean_val = df[algorithm].mean()
        median_val = df[algorithm].median()
        plt.scatter([algorithm], [mean_val], color='blue', label='Mean' if algorithm == df.columns[0] else "")
        plt.scatter([algorithm], [median_val], color='orange', label='Median' if algorithm == df.columns[0] else "")
        plt.text(algorithm, mean_val, f'{mean_val:.2e}', color='blue', ha='center', fontsize=8)
        plt.text(algorithm, median_val, f'{median_val:.2e}', color='orange', ha='center', fontsize=8)

    plt.legend()
    plt.tight_layout()
    plt.show()


    st.pyplot(plt)
    # Annotate mean and standard deviation
    for i, algorithm in enumerate(df.columns):
        mean = df[algorithm].mean()
        std = df[algorithm].std()
        plt.text(i, mean + 0.01, f'Mean: {mean:.2e}\nSD: {std:.2e}', horizontalalignment='center')

    plt.title(f'Boxplot with Swarm Overlay and Statistical Annotations for {file_name}')
    plt.xlabel('Algorithm')
    plt.ylabel('Values')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()
    st.pyplot(plt)

    stats_df = pd.DataFrame({
        'Mean': df.mean(),
        'Median': df.median(),
        'Std': df.std()
    })

    plt.figure(figsize=(8, 6))
    sns.heatmap(stats_df, annot=True, cmap='YlGnBu', fmt=".2f")
    plt.title(f'Descriptive Statistics Heatmap for {file_name}')
    plt.show()

    st.pyplot(plt)

if __name__ == "__main__":
    main()

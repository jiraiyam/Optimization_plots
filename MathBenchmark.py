import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from math import pi
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import kde
import io

# Set page config
st.set_page_config(page_title="Math Benchmark Visualization", layout="wide")

# Load data
@st.cache_data
def load_data(file):
    if file.name.endswith('.csv'):
        df = pd.read_csv(file)
    elif file.name.endswith(('.xls', '.xlsx')):
        df = pd.read_excel(file)
    else:
        st.error("Unsupported file format. Please upload a CSV or Excel file.")
        return None, None, None

    if 'Metric' not in df.columns or 'Unnamed: 0' not in df.columns:
        st.error(
            "The uploaded file does not have the expected structure. Please ensure it contains 'Metric' and 'Unnamed: 0' columns.")
        return None, None, None

    mean_df = df[df['Metric'] == 'Mean'].set_index('Unnamed: 0').drop('Metric', axis=1)
    std_df = df[df['Metric'] == 'STD'].set_index('Unnamed: 0').drop('Metric', axis=1)
    return df, mean_df, std_df

# Main app
def main():
    st.title("Math Benchmark Visualization App")

    # File uploader
    uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=['csv', 'xlsx', 'xls'])

    if uploaded_file is not None:
        df, mean_df, std_df = load_data(uploaded_file)

        if df is not None and mean_df is not None and std_df is not None:
            st.sidebar.title("Navigation")
            app_mode = st.sidebar.selectbox("Choose the plot type",
                                            ["Basic Plots", "Advanced Plots", "Interactive Plots", "Statistical Plots",
                                             "Time Series Plots"])

            if app_mode == "Basic Plots":
                basic_plots(mean_df)
            elif app_mode == "Advanced Plots":
                advanced_plots(mean_df)
            elif app_mode == "Interactive Plots":
                interactive_plots(mean_df, std_df)
            elif app_mode == "Statistical Plots":
                statistical_plots(mean_df, std_df)
            elif app_mode == "Time Series Plots":
                time_series_plots(mean_df)
    else:
        st.info("Please upload a CSV or Excel file to begin.")

# Basic Plots
def basic_plots(mean_df):
    st.header("Basic Plots")
    plot_type = st.selectbox("Choose a plot type", ["Bar Plot", "Line Plot", "Heatmap"])

    if plot_type == "Bar Plot":
        fig, ax = plt.subplots(figsize=(12, 8))
        mean_df.plot(kind='bar', ax=ax)
        plt.title('Mean Values for Each Benchmark Function')
        plt.ylabel('Mean Value')
        plt.xlabel('Benchmark Function')
        plt.xticks(rotation=45)
        plt.legend(title='Algorithm', bbox_to_anchor=(1.05, 1), loc='upper left')
        st.pyplot(fig)

    elif plot_type == "Line Plot":
        fig, ax = plt.subplots(figsize=(12, 8))
        for column in mean_df.columns:
            plt.plot(mean_df.index, mean_df[column], marker='o', label=column)
        plt.title('Mean Values for Each Benchmark Function (Line Plot)')
        plt.xlabel('Benchmark Function')
        plt.ylabel('Mean Value')
        plt.xticks(rotation=45)
        plt.legend(title='Algorithm', bbox_to_anchor=(1.05, 1), loc='upper left')
        st.pyplot(fig)


    elif plot_type == "Heatmap":

        fig, ax = plt.subplots(figsize=(25, 15))

        # Create heatmap

        heatmap = sns.heatmap(

            mean_df.T,

            annot=True,

            cmap='viridis',  # Color map for better contrast and color blindness accessibility

            fmt='.1',

            annot_kws={"size": 10, "weight": 'bold'},  # Customize annotation size and weight

            ax=ax,

            cbar_kws={'label': 'Mean Value'}  # Add label to color bar

        )

        # Set titles and labels

        ax.set_title('Heatmap of Mean Values for Each Algorithm', fontsize=16, weight='bold')

        ax.set_xlabel('Benchmark Function', fontsize=14, weight='bold')

        ax.set_ylabel('Algorithm', fontsize=14, weight='bold')

        # Rotate x-axis labels for better readability

        plt.xticks(rotation=45, ha='right', fontsize=12)

        # Adjust y-axis label size and orientation

        plt.yticks(fontsize=12)

        # Add grid lines for better readability

        ax.grid(True, linestyle='--', linewidth=0.7, color='gray')

        st.pyplot(fig)

# Advanced Plots
def advanced_plots(mean_df):
    st.header("Advanced Plots")
    plot_type = st.selectbox("Choose a plot type", ["3D Surface Plot", "Radar Plot", "Stacked Bar Plot", "Clustermap"])

    if plot_type == "3D Surface Plot":
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        x = np.arange(mean_df.shape[0])
        y = np.arange(mean_df.shape[1])
        X, Y = np.meshgrid(y, x)
        Z = mean_df.values
        ax.plot_surface(X, Y, Z, cmap='viridis')
        ax.set_xlabel('Algorithm')
        ax.set_ylabel('Benchmark Function')
        ax.set_zlabel('Mean Value')
        ax.set_title('3D Surface Plot of Mean Values')
        st.pyplot(fig)

    elif plot_type == "Radar Plot":
        categories = mean_df.index
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
        for column in mean_df.columns:
            values = mean_df[column].tolist()
            values += values[:1]
            angles = [n / float(len(categories)) * 2 * pi for n in range(len(categories))]
            angles += angles[:1]
            ax.plot(angles, values, linewidth=1, linestyle='solid', label=column)
            ax.fill(angles, values, alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        plt.title('Radar Plot of Mean Values Across Algorithms')
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        st.pyplot(fig)

    elif plot_type == "Stacked Bar Plot":
        fig, ax = plt.subplots(figsize=(12, 8))
        mean_df.plot(kind='bar', stacked=True, ax=ax)
        plt.title('Stacked Bar Plot of Mean Values for Each Benchmark Function')
        plt.xlabel('Benchmark Function')
        plt.ylabel('Mean Value')
        plt.xticks(rotation=45)
        plt.legend(title='Algorithm', bbox_to_anchor=(1.05, 1), loc='upper left')
        st.pyplot(fig)


    elif plot_type == "Clustermap":

        # Create a clustermap

        g = sns.clustermap(

            mean_df.T,

            method='ward',

            cmap='coolwarm',  # Choose a color map that highlights differences

            annot=True,

            fmt='.1',

            linewidths=0.5,

            annot_kws={"size": 10, "weight": 'bold'},  # Customize annotation size and weight

            cbar_kws={'label': 'Mean Value'}  ,
            figsize=(22, 15)
            # Add label to color bar

        )

        # Update title and labels

        plt.suptitle('Clustermap of Algorithms Based on Mean Values Across Functions', fontsize=16, weight='bold')

        g.ax_heatmap.set_xlabel('Benchmark Function', fontsize=14, weight='bold')

        g.ax_heatmap.set_ylabel('Algorithm', fontsize=14, weight='bold')

        # Adjust tick label sizes

        plt.setp(g.ax_heatmap.get_xticklabels(), rotation=45, ha='right', fontsize=12)

        plt.setp(g.ax_heatmap.get_yticklabels(), fontsize=12)

        # Adjust spacing to make room for the title

        plt.subplots_adjust(top=0.9)

        st.pyplot(g.figure)

# Interactive Plots
def interactive_plots(mean_df, std_df):
    st.header("Interactive Plots")
    plot_type = st.selectbox("Choose a plot type",
                             ["Interactive Heatmap", "Interactive 3D Scatter Plot", "Interactive Box Plot"])

    if plot_type == "Interactive Heatmap":
        fig = go.Figure(data=go.Heatmap(z=mean_df.values,
                                        x=mean_df.columns,
                                        y=mean_df.index,
                                        colorscale='Viridis',
                                        hovertemplate='Algorithm: %{x}<br>Function: %{y}<br>Value: %{z:.2e}<extra></extra>'))
        fig.update_layout(title='Heatmap of Mean Values',
                          xaxis_title='Algorithm',
                          yaxis_title='Benchmark Function')
        st.plotly_chart(fig)

    elif plot_type == "Interactive 3D Scatter Plot":
        long_mean_df = mean_df.reset_index().melt(id_vars='Unnamed: 0', var_name='Algorithm', value_name='Mean Value')
        fig = px.scatter_3d(long_mean_df, x='Unnamed: 0', y='Algorithm', z='Mean Value',
                            color='Algorithm', title='Interactive 3D Scatter Plot of Mean Values')
        fig.update_layout(scene=dict(xaxis_title='Benchmark Function',
                                     yaxis_title='Algorithm',
                                     zaxis_title='Mean Value'))
        st.plotly_chart(fig)

    elif plot_type == "Interactive Box Plot":
        fig = go.Figure()
        for algorithm in std_df.columns:
            fig.add_trace(go.Box(y=std_df[algorithm], name=algorithm))
        fig.update_layout(title='Box Plot of Standard Deviation Values',
                          xaxis_title='Algorithm',
                          yaxis_title='Standard Deviation')
        st.plotly_chart(fig)

# Statistical Plots
def statistical_plots(mean_df, std_df):
    st.header("Statistical Plots")
    plot_type = st.selectbox("Choose a plot type",
                             ["Box Plot with Swarm", "Violin Plot", "KDE Plot", "Bar Plot with Error Bars"])

    if plot_type == "Box Plot with Swarm":
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.boxplot(data=mean_df.T, ax=ax, palette="Set3", showfliers=False)
        sns.swarmplot(data=mean_df.T, ax=ax, color=".25")
        plt.title('Box Plot of Algorithm Performance with Swarm Overlay')
        plt.xticks(rotation=90)
        st.pyplot(fig)

    elif plot_type == "Violin Plot":
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.violinplot(data=mean_df.T, ax=ax, palette="Set2")
        plt.title('Violin Plot of Algorithm Performance Across Functions')
        plt.xticks(rotation=90)
        st.pyplot(fig)

    elif plot_type == "KDE Plot":
        fig, ax = plt.subplots(figsize=(10, 6))
        for alg in mean_df.columns:
            sns.kdeplot(mean_df[alg], ax=ax, label=alg, shade=True)
        plt.title('Density Plot for Algorithm Performance Across Functions')
        plt.xlabel('Mean Value')
        plt.legend()
        st.pyplot(fig)

    elif plot_type == "Bar Plot with Error Bars":
        fig, ax = plt.subplots(figsize=(10, 6))
        mean_values = mean_df.mean()
        std_values = mean_df.std()
        plt.bar(mean_df.columns, mean_values, yerr=std_values, capsize=5, color='skyblue')
        plt.title('Bar Plot of Algorithm Performance with Error Bars')
        plt.ylabel('Mean Value with STD Error')
        plt.xticks(rotation=90)
        st.pyplot(fig)

# Time Series Plots
def time_series_plots(mean_df):
    st.header("Time Series Plots")
    plot_type = st.selectbox("Choose a plot type",
                             ["Cumulative Performance", "Rolling Statistics", "Performance Improvement Heatmap"])

    if plot_type == "Cumulative Performance":
        cumulative_means = mean_df.cumsum(axis=0)
        fig, ax = plt.subplots(figsize=(12, 6))
        for column in cumulative_means.columns:
            ax.plot(cumulative_means.index, cumulative_means[column], label=column)
        ax.set_title('Cumulative Performance of Algorithms Over Functions')
        ax.set_xlabel('Functions')
        ax.set_ylabel('Cumulative Mean')
        ax.legend(title='Algorithms', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(rotation=45)
        st.pyplot(fig)

    elif plot_type == "Rolling Statistics":
        rolling_mean = mean_df.rolling(window=3, axis=0).mean()
        rolling_std = mean_df.rolling(window=3, axis=0).std()

        fig, ax = plt.subplots(figsize=(12, 6))
        for column in rolling_mean.columns:
            ax.plot(rolling_mean.index, rolling_mean[column], label=f'{column} - Mean')
            ax.plot(rolling_std.index, rolling_std[column], linestyle='--', label=f'{column} - STD')

        ax.set_title('Rolling Mean and Standard Deviation of Algorithm Performance')
        ax.set_xlabel('Functions')
        ax.set_ylabel('Value')
        ax.legend(title='Metrics', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(rotation=45)
        st.pyplot(fig)


    elif plot_type == "Performance Improvement Heatmap":

        performance_over_time = mean_df.T

        fig, ax = plt.subplots(figsize=(20, 15))

        # Create heatmap

        heatmap = sns.heatmap(

            performance_over_time,

            cmap='coolwarm',  # Changed color map for better contrast

            annot=True,

            fmt='.1',

            annot_kws={"size": 10, "weight": 'bold'},  # Customize annotation

            ax=ax,

            cbar_kws={'label': 'Performance Improvement'}  # Add label to color bar

        )

        # Set titles and labels

        ax.set_title('Heatmap of Performance Improvement Over Time', fontsize=16, weight='bold')

        ax.set_xlabel('Functions', fontsize=14, weight='bold')

        ax.set_ylabel('Algorithms', fontsize=14, weight='bold')

        # Rotate x-axis labels for better readability

        plt.xticks(rotation=45, ha='right', fontsize=12)

        # Adjust y-axis label size and orientation

        plt.yticks(fontsize=12)

        # Add grid lines for better readability

        ax.grid(True, linestyle='--', linewidth=0.7, color='gray')

        st.pyplot(fig)


if __name__ == "__main__":
    main()
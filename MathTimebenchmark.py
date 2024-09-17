import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from scipy import stats

# Set page config
st.set_page_config(page_title="Math Time Benchmark Visualization", layout="wide")


# Load data
@st.cache_data
def load_data(file):

    if file.name.endswith('.csv'):
        df = pd.read_csv(file)
        df.columns = ['Function', 'Metric', 'DE', 'ECOA', 'GA', 'GWO',
                      'HDGCO', 'LCO', 'PSO', 'SKO', 'WOA']
    elif file.name.endswith(('.xls', '.xlsx')):
        try:
            df = pd.read_excel(file)
        except ImportError:
            st.error("Unable to read Excel file. Please make sure you have openpyxl installed.")
            st.info("You can install it by running: pip install openpyxl")
            return None
    else:
        st.error("Unsupported file format. Please upload a CSV or Excel file.")
        return None

    if 'Function' not in df.columns or 'Metric' not in df.columns:
        df.columns = ['Function', 'Metric', 'DE', 'ECOA', 'GA', 'GWO',
                      'HDGCO', 'LCO', 'PSO', 'SKO', 'WOA']

    return df


# Main app
def main():
    st.title("Math Benchmark Visualization App")

    # File uploader
    uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=['csv', 'xlsx', 'xls'])

    if uploaded_file is not None:
        df = load_data(uploaded_file)

        if df is not None:
            st.sidebar.title("Navigation")
            app_mode = st.sidebar.selectbox("Choose the plot type",
                                            ["Basic Plots", "Advanced Plots", "Interactive Plots", "Statistical Plots",
                                             "Time Series Plots"])

            if app_mode == "Basic Plots":
                basic_plots(df)
            elif app_mode == "Advanced Plots":
                advanced_plots(df)
            elif app_mode == "Interactive Plots":
                interactive_plots(df)
            elif app_mode == "Statistical Plots":
                statistical_plots(df)
            elif app_mode == "Time Series Plots":
                time_series_plots(df)
    else:
        st.info("Please upload a CSV or Excel file to begin.")


# Basic Plots
def basic_plots(df):
    st.header("Basic Plots")
    plot_type = st.selectbox("Choose a plot type", ["Bar Plot", "Line Plot", "Heatmap"])

    if plot_type == "Bar Plot":
        avg_time_df = df[df['Metric'] == 'avg_time']
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.barplot(x='Function', y='value', hue='Algorithm', data=avg_time_df.melt(id_vars=['Function', 'Metric'],
                                                                                    var_name='Algorithm',
                                                                                    value_name='value'), ax=ax)
        plt.title('Average Time Comparison')
        plt.ylabel('Average Time')
        plt.xticks(rotation=90)
        st.pyplot(fig)

    elif plot_type == "Line Plot":
        fig, ax = plt.subplots(figsize=(14, 7))
        sns.lineplot(data=df[df['Metric'] == 'avg_time'].melt(id_vars=['Function', 'Metric'], var_name='Algorithm',
                                                              value_name='Value'),
                     x='Function', y='Value', hue='Algorithm', marker='o', ax=ax)
        plt.title('Average Time by Function for Different Algorithms')
        plt.xticks(rotation=45)
        plt.xlabel('Function')
        plt.ylabel('Average Time')
        plt.legend(title='Algorithm', bbox_to_anchor=(1.05, 1), loc='upper left')
        st.pyplot(fig)


    elif plot_type == "Heatmap":
        avg_fes_df = df[df['Metric'] == 'Avg_FEs'].drop(columns=['Metric'])
        avg_fes_df = avg_fes_df.set_index('Function').transpose()
        fig, ax = plt.subplots(figsize=(14, 10))
        custom_x_labels = [f'F{i}' for i in range(1, avg_fes_df.shape[1] + 1)]  # Assuming 23 functions

        sns.heatmap(
            avg_fes_df, annot=True, fmt='.0f', cmap='coolwarm',
            cbar=True, cbar_kws={'label': 'Avg FEs', 'shrink': 0.8},
            linewidths=0.4, linecolor='gray', ax=ax, annot_kws={'size': 8, 'weight': 'bold'}
        )
        plt.title('Average Function Evaluations (FEs) by Function and Algorithm', fontsize=16, weight='bold')
        plt.xlabel('Function', fontsize=14, weight='bold')
        ax.set_xticklabels(custom_x_labels, rotation=45, ha='right', fontsize=12, weight='bold')

        plt.ylabel('Algorithm', fontsize=14, weight='bold')
        plt.xticks(rotation=45, ha='right', fontsize=12, weight='bold')
        plt.yticks(fontsize=12, weight='bold')
        plt.tight_layout()
        st.pyplot(fig)


# Advanced Plots
def advanced_plots(df):
    st.header("Advanced Plots")
    plot_type = st.selectbox("Choose a plot type", ["Radar Chart", "Stacked Area Chart avg_time" ,  "Swarm Plot"])

    if plot_type == "Radar Chart":
        radar_data = df[df['Metric'] == 'avg_time'].drop(columns=['Metric']).set_index('Function').transpose()

        labels = radar_data.columns
        num_vars = len(labels)

        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)

        plt.xticks(angles[:-1], labels)

        for index, row in radar_data.iterrows():
            values = row.tolist()
            values += values[:1]
            ax.plot(angles, values, label=index)
            ax.fill(angles, values, alpha=0.25)

        plt.title('Radar Chart of Average Time by Function and Algorithm', size=20, color='blue', y=1.1)
        plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))
        st.pyplot(fig)

    elif plot_type == "Stacked Area Chart avg_time":
        algorithms = ['DE', 'ECOA', 'GA', 'GWO', 'HDGCO', 'LCO', 'PSO', 'SKO', 'WOA']

        # Number of rows and columns for subplots (adjust as needed)
        n_cols = 3
        n_rows = len(algorithms) // n_cols + (len(algorithms) % n_cols > 0)

        fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(18, 12), constrained_layout=True)

        # Flatten axes for easy iteration if it's a 2D array
        axes = axes.flatten()

        for i, algo in enumerate(algorithms):
            df_area = df.pivot_table(index='Function', columns='Metric', values=algo)

            # Create the stacked area plot for the current algorithm
            df_area.plot(kind='area', stacked=True, ax=axes[i])

            axes[i].set_title(f'Stacked Area Chart of {algo} by Function and Metric', fontsize=12, weight='bold')
            axes[i].set_xlabel('Function', fontsize=10, weight='bold')
            axes[i].set_ylabel(algo, fontsize=10, weight='bold')
            axes[i].legend(title='Metric', fontsize=8)
            axes[i].tick_params(axis='x', labelrotation=45)

        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        st.pyplot(fig)






    elif plot_type == "Swarm Plot":
        value_vars = [col for col in df.columns if col not in ['Function', 'Metric']]
        df_swarm = df[df['Metric'] == 'avg_time'].melt(id_vars='Function', value_vars=value_vars, var_name='Optimizer',
                                                       value_name='Time')
        df_swarm['Optimizer'] = df_swarm['Optimizer'].str.split('_').str[0]

        fig, ax = plt.subplots(figsize=(14, 8))
        sns.swarmplot(x='Optimizer', y='Time', data=df_swarm, hue='Function', palette='Set1', dodge=True, ax=ax)
        plt.title('Swarm Plot of Average Time Across Optimizers')
        plt.xlabel('Optimizer')
        plt.ylabel('Average Time (seconds)')
        plt.legend(title='Function')
        plt.grid(True)
        st.pyplot(fig)


def interactive_plots(df):
    st.header("Interactive Plots")
    plot_type = st.selectbox("Choose a plot type",
                             ["Interactive Heatmap", "Interactive 3D Scatter Plot", "Interactive Box Plot"])

    if plot_type == "Interactive Heatmap":
        avg_time_df = df[df['Metric'] == 'avg_time']
        pivot_avg_time = avg_time_df.set_index('Function').drop(columns='Metric').T
        fig = go.Figure(data=go.Heatmap(z=pivot_avg_time.values,
                                        x=pivot_avg_time.columns,
                                        y=pivot_avg_time.index,
                                        colorscale='Viridis',
                                        hovertemplate='Function: %{x}<br>Algorithm: %{y}<br>Value: %{z:.2f}<extra></extra>'))
        fig.update_layout(title='Heatmap of Average Time',
                          xaxis_title='Function',
                          yaxis_title='Algorithm')
        st.plotly_chart(fig)

    elif plot_type == "Interactive 3D Scatter Plot":
        avg_time_df = df[df['Metric'] == 'avg_time']
        long_avg_time_df = avg_time_df.melt(id_vars='Function', var_name='Algorithm', value_name='Avg_Time')
        fig = px.scatter_3d(long_avg_time_df, x='Function', y='Algorithm', z='Avg_Time',
                            color='Algorithm', title='Interactive 3D Scatter Plot of Average Time')
        st.plotly_chart(fig)

    elif plot_type == "Interactive Box Plot":
        avg_time_df = df[df['Metric'] == 'avg_time']
        long_avg_time_df = avg_time_df.melt(id_vars='Function', var_name='Algorithm', value_name='Avg_Time')
        fig = px.box(long_avg_time_df, x='Algorithm', y='Avg_Time', color='Algorithm',
                     title='Box Plot of Average Time Values')
        st.plotly_chart(fig)


def statistical_plots(df):
    st.header("Statistical Plots")

    plot_type = st.selectbox("Choose a plot type", [
         "Q-Q Plot", "KDE Plot", "Box Plot", "Violin Plot", "Swarm Plot"
    ])

    if plot_type == "Q-Q Plot":
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        for i, metric in enumerate(['avg_time', 'std_time', 'Avg_FEs']):
            data = df[df['Metric'] == metric]['DE']
            stats.probplot(data, dist="norm", plot=axes[i])
            axes[i].set_title(f'Q-Q Plot of {metric}', fontsize=12, weight='bold')
            axes[i].grid(True, linestyle='--', alpha=0.7)
        plt.suptitle('Q-Q Plots for DE Metric', fontsize=16, weight='bold', y=1.05)
        st.pyplot(fig)

    elif plot_type == "KDE Plot":
        avg_time_df = df[df['Metric'] == 'avg_time'].drop(columns=['Metric']).set_index('Function').T
        fig, ax = plt.subplots(figsize=(10, 6))
        for col in avg_time_df.columns:
            sns.kdeplot(avg_time_df[col], ax=ax, label=col, shade=True, linewidth=2)
        plt.title('KDE Plot of Average Time for Algorithms Across Functions', fontsize=16, weight='bold', pad=15)
        plt.xlabel('Average Time', fontsize=12, weight='bold')
        plt.ylabel('Density', fontsize=12, weight='bold')
        plt.legend(title='Function', fontsize=10, title_fontsize=12, loc='upper right')
        plt.grid(True, linestyle='--', alpha=0.5)
        st.pyplot(fig)

    elif plot_type == "Box Plot":
        box_data = df[df['Metric'] == 'avg_time'].drop(columns=['Metric']).set_index('Function').T
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.boxplot(data=box_data, ax=ax, palette='Set3', linewidth=2)
        plt.title('Box Plot of Average Time Across Functions', fontsize=16, weight='bold', pad=15)
        plt.xlabel('Function', fontsize=12, weight='bold')
        plt.ylabel('Average Time', fontsize=12, weight='bold')
        plt.xticks(rotation=45, fontsize=10, weight='bold')
        ax.grid(True, linestyle='--', alpha=0.5)
        st.pyplot(fig)

    elif plot_type == "Violin Plot":
        violin_data = df[df['Metric'] == 'avg_time'].drop(columns=['Metric']).set_index('Function').T
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.violinplot(data=violin_data, ax=ax, palette='muted')
        plt.title('Violin Plot of Average Time Across Functions', fontsize=16, weight='bold', pad=15)
        plt.xlabel('Function', fontsize=12, weight='bold')
        plt.ylabel('Average Time', fontsize=12, weight='bold')
        plt.xticks(rotation=45, fontsize=10, weight='bold')
        ax.grid(True, linestyle='--', alpha=0.5)
        st.pyplot(fig)

    elif plot_type == "Swarm Plot":
        swarm_data = df[df['Metric'] == 'avg_time'].drop(columns=['Metric']).set_index('Function').T.melt(var_name='Function', value_name='Average Time')
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.swarmplot(x='Function', y='Average Time', data=swarm_data, ax=ax, palette='husl', size=6)
        plt.title('Swarm Plot of Average Time Across Functions', fontsize=16, weight='bold', pad=15)
        plt.xlabel('Function', fontsize=12, weight='bold')
        plt.ylabel('Average Time', fontsize=12, weight='bold')
        plt.xticks(rotation=45, fontsize=10, weight='bold')
        ax.grid(True, linestyle='--', alpha=0.5)
        st.pyplot(fig)


def time_series_plots(df):
    st.header("Time Series Plots")

    plot_type = st.selectbox("Choose a plot type", ["Line Plot", "Area Plot", "Stacked Area Plot"])

    if plot_type == "Line Plot":
        avg_time_df = df[df['Metric'] == 'avg_time']

        fig, ax = plt.subplots(figsize=(14, 8))

        for optimizer in avg_time_df.columns[2:]:
            sns.lineplot(data=avg_time_df, x='Function', y=optimizer, label=optimizer, marker='o', ax=ax, linewidth=2.5)

        plt.title('Average Time Across Functions for Each Optimizer', fontsize=18, weight='bold')
        plt.xlabel('Function', fontsize=14, weight='bold')
        plt.ylabel('Average Time', fontsize=14, weight='bold')
        plt.xticks(rotation=45, fontsize=12, weight='bold')
        plt.yticks(fontsize=12, weight='bold')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(title='Optimizer', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12, title_fontsize=14)

        st.pyplot(fig)

    elif plot_type == "Area Plot":
        std_time_df = df[df['Metric'] == 'std_time']

        function_labels = [f'F{i}' for i in range(1, 24)]
        std_time_df['Function'] = function_labels[:len(std_time_df)]

        pivot_std_time = std_time_df.set_index('Function').drop(columns='Metric').T

        fig, ax = plt.subplots(figsize=(16, 9))

        pivot_std_time.plot(kind='area', stacked=False, ax=ax, cmap='coolwarm', alpha=0.85)

        plt.title('Area Plot of Standard Time Across Functions and Optimizers', fontsize=18, weight='bold', pad=20)
        plt.xlabel('Function', fontsize=14, weight='bold')
        plt.ylabel('Standard Time', fontsize=14, weight='bold')
        plt.xticks(rotation=45, fontsize=12, weight='bold')
        plt.yticks(fontsize=12, weight='bold')

        ax.grid(True, linestyle='--', alpha=0.5)

        plt.legend(title='Functions', fontsize=12, title_fontsize=14, loc='upper left', bbox_to_anchor=(1, 1))

        st.pyplot(fig)

    elif plot_type == "Stacked Area Plot":
        avg_time_df = df[df['Metric'] == 'avg_time']

        function_labels = [f'F{i}' for i in range(1, 24)]
        avg_time_df['Function'] = function_labels[:len(avg_time_df)]

        pivot_avg_time = avg_time_df.set_index('Function').drop(columns='Metric').T

        fig, ax = plt.subplots(figsize=(16, 9))

        pivot_avg_time.plot(kind='area', stacked=True, ax=ax, cmap='coolwarm', alpha=0.85)

        plt.title('Stacked Area Plot of Average Time Across Functions and Optimizers', fontsize=18, weight='bold',
                  pad=20)
        plt.xlabel('Function', fontsize=14, weight='bold')
        plt.ylabel('Average Time', fontsize=14, weight='bold')
        plt.xticks(rotation=45, fontsize=12, weight='bold')
        plt.yticks(fontsize=12, weight='bold')

        ax.grid(True, linestyle='--', alpha=0.5)

        plt.legend(title='Optimizers', fontsize=12, title_fontsize=14, loc='upper left', bbox_to_anchor=(1, 1))

        st.pyplot(fig)


if __name__ == "__main__":
    main()

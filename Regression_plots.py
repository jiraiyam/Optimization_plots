import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from math import pi
import numpy as np
import scipy.stats as stats
from scipy.stats import norm
from scipy.stats import zscore
from scipy.cluster.hierarchy import linkage, dendrogram
from pandas.plotting import parallel_coordinates
from pandas.plotting import andrews_curves
import networkx as nx
import scipy.cluster.hierarchy as sch
from scipy.interpolate import CubicSpline
import math
sns.set(style="whitegrid")
plt.rcParams.update(
    {'font.weight': 'bold', 'axes.labelweight': 'bold',
     'axes.titleweight': 'bold', 'axes.titlesize': 14}
)

# Load file function
def load_file(uploaded_file):
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith('.xlsx'):
        df = pd.read_excel(uploaded_file)
    # Template to drop 'Fitted Time' column if it exists
    if 'Fitted Time' in df.columns:
        df.drop(['Fitted Time'], axis=1, inplace=True)
    return df


def section_1(df):
    st.subheader("Section 1 Analysis")

    # Copy and clean up the data
    data = df.copy()
    data.columns = data.columns.str.strip()

    plt.figure(figsize=(12, 6))
    sns.boxplot(data=data.drop('Models', axis=1))
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.title("Box Plot of Model Metrics")
    plt.ylabel("Values")
    st.pyplot(plt)
    plt.clf()

    melted_data = data.melt(id_vars='Models', var_name='Metric', value_name='Value')

    # Create the boxplot with swarm overlay
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='Metric', y='Value', data=melted_data, palette="Set3")
    sns.swarmplot(x='Metric', y='Value', data=melted_data, color=".25", size=5)
    plt.title("Box Plot with Swarm Overlay of Model Performance Metrics")
    plt.xticks(rotation=45)
    st.pyplot(plt)  # Display the plot in the Streamlit app
    plt.clf()

    melted_data = data.melt(id_vars='Models', var_name='Metric', value_name='Value')

    # Create the Violin plot with jittered data points
    plt.figure(figsize=(12, 6))
    sns.violinplot(x='Metric', y='Value', data=melted_data, palette="muted", inner=None)
    sns.stripplot(x='Metric', y='Value', data=melted_data, color="black", size=4, jitter=0.3)
    plt.title("Violin Plot with Jittered Data Points for Model Performance Metrics")
    plt.xticks(rotation=45)
    st.pyplot(plt)  # Display the plot in the Streamlit app
    plt.clf()
####################################################################
    metrics = data.copy()  # Assuming `data` contains the metrics you want to plot

    # Dynamically determine the number of columns and rows
    num_metrics = len(metrics.columns[1:])  # Exclude the first column (e.g., 'Models')
    num_cols = math.ceil(math.sqrt(num_metrics))  # Choose columns based on square root
    num_rows = math.ceil(num_metrics / num_cols)  # Calculate rows accordingly

    plt.figure(figsize=(16, 12))  # Adjust the figure size dynamically if needed

    for i, metric in enumerate(metrics.columns[1:], 1):  # Assuming the first column is 'Models'
        plt.subplot(num_rows, num_cols, i)  # Create subplot for each metric

        # Boxplot with added customization (palette and linewidth)
        sns.boxplot(data=metrics, y=metric, color="lightblue", linewidth=1.5)

        # Swarm plot with customized color and size of points
        sns.swarmplot(data=metrics, y=metric, color="black", alpha=0.7, size=6)

        # Add mean and standard deviation text annotations
        mean_value = metrics[metric].mean()
        std_value = metrics[metric].std()

        plt.text(0.75, mean_value, f'Mean:\n {mean_value:.4f}', horizontalalignment='left',
                 verticalalignment='center', fontsize=10, color='darkblue')
        plt.text(0.45, mean_value + std_value, f'STD:\n {std_value:.4f}\n', horizontalalignment='left',
                 verticalalignment='center', fontsize=10, color='darkred')

        # Title for each subplot
        plt.title(f"{metric} Distribution Across Models", fontsize=12)

        # Customizing y-axis limits and grid
        plt.ylim(min(metrics[metric]) - 0.05, max(metrics[metric]) + 0.05)
        plt.grid(True, axis='y', linestyle='--', alpha=0.6)

    plt.tight_layout()  # Adjust layout for spacing
    plt.suptitle("Swarm Plot Overlayed on Box Plot for Individual Metrics with Mean and STD", y=1.05, fontsize=16)
    st.pyplot(plt)  # Display the plot in the Streamlit app
    plt.clf()
#################################################
    metrics.columns = [metrics.columns[0]] + metrics.columns[1:].str.lower().tolist()

    g = sns.FacetGrid(metrics, col="Models", col_wrap=4, height=3, aspect=1.5)

    # Map boxplot to the grid for MSE
    g.map(sns.boxplot, 'Models', 'mse', color='lightblue')

    # Map swarmplot to the grid for MSE
    g.map(sns.swarmplot, 'Models', 'mse', color='black', alpha=0.5)

    # Customize the grid appearance
    g.set_titles("{col_name}")  # Set titles for each facet
    g.set_axis_labels('Models', 'MSE')  # Label axes
    g.set_xticklabels(rotation=45)  # Rotate x-axis labels for better visibility

    # Adjust layout for better spacing
    plt.subplots_adjust(top=0.9)

    # Set the title for the entire grid

    # Display the plot in Streamlit
    st.pyplot(plt.gcf())  # Show the plot
    plt.clf()
###################################################################
    num_metrics = len(metrics.columns[1:])  # Exclude the first column (e.g., 'Models')

    # Dynamically calculate the number of columns and rows
    num_cols = math.ceil(math.sqrt(num_metrics))  # Optimal number of columns
    num_rows = math.ceil(num_metrics / num_cols)  # Calculate rows based on columns

    plt.figure(figsize=(16, 12))  # Set figure size dynamically

    for i, metric in enumerate(metrics.columns[1:], 1):  # Start with index 1
        plt.subplot(num_rows, num_cols, i)  # Dynamically set the subplot grid

        # Boxplot with added customization (palette and linewidth)
        sns.boxplot(data=metrics, y=metric, color="lightblue", linewidth=1.5)

        # Swarm plot with customized color and size of points
        sns.swarmplot(data=metrics, y=metric, color="black", alpha=0.7, size=6, warn_thresh=0)

        # Add mean and standard deviation text annotations
        mean_value = metrics[metric].mean()
        std_value = metrics[metric].std()

        # Adding text with relative positioning
        plt.text(0.5, mean_value, f'Mean: {mean_value:.4f}', horizontalalignment='center',
                 verticalalignment='center', fontsize=10, color='darkblue')
        plt.text(0.5, mean_value + std_value, f'STD: {std_value:.4f}', horizontalalignment='center',
                 verticalalignment='center', fontsize=10, color='darkred')

        # Title for each subplot
        plt.title(f"{metric} Distribution Across Models", fontsize=12)

        # Customizing y-axis limits and grid
        plt.ylim(min(metrics[metric]) - 0.05, max(metrics[metric]) + 0.05)
        plt.grid(True, axis='y', linestyle='--', alpha=0.6)

    plt.tight_layout()  # Adjust layout for better spacing
    plt.suptitle("Swarm Plot Overlayed on Box Plot for Individual Metrics with Mean and STD", y=1.02, fontsize=16)
    st.pyplot(plt)  # Display the plot in the Streamlit app
    plt.clf()
###########################################################
    metrics = data.copy()  # Assuming `data` contains the metrics you want to plot

    # Determine the number of metrics to plot (excluding the first column)
    num_metrics = len(metrics.columns[1:])  # Exclude the first column (e.g., 'Models')

    # Dynamically calculate the number of columns and rows
    num_cols = math.ceil(math.sqrt(num_metrics))  # Optimal number of columns
    num_rows = math.ceil(num_metrics / num_cols)  # Calculate rows based on columns

    plt.figure(figsize=(16, 12))  # Set figure size dynamically

    # Loop through each metric and create a Violin plot with Swarm plot overlay
    for i, metric in enumerate(metrics.columns[1:], 1):  # Start with index 1
        plt.subplot(num_rows, num_cols, i)  # Dynamically set the subplot grid

        # Violin plot with lightblue color and no inner data (to highlight distribution)
        sns.violinplot(data=metrics, y=metric, color="lightblue", inner=None)

        # Swarm plot with black color and alpha 0.5 for transparency
        sns.swarmplot(data=metrics, y=metric, color="black", alpha=0.5)

        # Title for each subplot
        plt.title(f"{metric} Distribution Across Models", fontsize=12)

    # Adjust layout for better spacing
    plt.tight_layout()

    # Set a title for the entire figure
    plt.suptitle("Violin Plot with Swarm Plot Overlay for Metrics", y=1.02, fontsize=16)

    # Display the plot in the Streamlit app
    st.pyplot(plt)  # Render the plot in Streamlit
    plt.clf()
###################################################################
    metrics = data.copy()  # Assuming `data` contains the metrics you want to plot

    # Determine the number of metrics to plot (excluding the first column)
    num_metrics = len(metrics.columns[1:])  # Exclude the first column (e.g., 'Models')

    # Dynamically calculate the number of columns and rows
    num_cols = math.ceil(math.sqrt(num_metrics))  # Optimal number of columns
    num_rows = math.ceil(num_metrics / num_cols)  # Calculate rows based on columns

    plt.figure(figsize=(16, 12))  # Set figure size dynamically

    # Loop through each metric and create a Box plot with Horizontal Swarm plot overlay
    for i, metric in enumerate(metrics.columns[1:], 1):  # Start with index 1
        plt.subplot(num_rows, num_cols, i)  # Dynamically set the subplot grid

        # Box plot with horizontal orientation
        sns.boxplot(data=metrics, x=metric, color="blue", width=0.5)

        # Swarm plot with horizontal orientation
        sns.swarmplot(data=metrics, x=metric, color="black", alpha=0.5)

        # Title for each subplot
        plt.title(f"{metric} Distribution Across Models", fontsize=12)

    # Adjust layout for better spacing
    plt.tight_layout()

    # Set a title for the entire figure
    plt.suptitle("Box Plot with Horizontal Swarm Plot for Metrics", y=1.02, fontsize=16)

    # Display the plot in the Streamlit app
    st.pyplot(plt)  # Render the plot in Streamlit
    plt.clf()
############################################################

    metrics = data.copy()  # Assuming `data` contains the metrics you want to plot

    # Determine the number of metrics to plot (excluding the first column)
    num_metrics = len(metrics.columns[1:])  # Exclude the first column (e.g., 'Models')

    # Dynamically calculate the number of columns and rows
    num_cols = math.ceil(math.sqrt(num_metrics))  # Optimal number of columns
    num_rows = math.ceil(num_metrics / num_cols)  # Calculate rows based on columns

    plt.figure(figsize=(16, 12))  # Set figure size dynamically

    # Loop through each metric and create a Box plot with Mean and Std annotations
    for i, metric in enumerate(metrics.columns[1:], 1):  # Start with index 1
        plt.subplot(num_rows, num_cols, i)  # Dynamically set the subplot grid

        # Create the box plot
        sns.boxplot(data=metrics, y=metric, color="lightblue")

        # Calculate mean and standard deviation for the metric
        mean_value = metrics[metric].mean()
        std_value = metrics[metric].std()

        # Plot the mean and std lines using axhline
        plt.axhline(mean_value, color="black", linewidth=2, label=f'Mean: {mean_value:.4f}')
        plt.axhline(mean_value + std_value, color="red", linestyle='--', label=f'Mean + Std: {mean_value + std_value:.4f}')
        plt.axhline(mean_value - std_value, color="green", linestyle='--', label=f'Mean - Std: {mean_value - std_value:.4f}')

        # Title for each subplot
        plt.title(f"{metric} Distribution with Mean & Std", fontsize=12)

        # Show legend
        plt.legend(loc="upper right", fontsize=8)

    # Adjust layout for better spacing
    plt.tight_layout()

    # Set a title for the entire figure
    plt.suptitle("Box Plot with Mean & Std for Each Metric", y=1.02, fontsize=16)

    # Display the plot in the Streamlit app
    st.pyplot(plt)  # Render the plot in Streamlit
    plt.clf()
######################################################
    metrics_columns = data.columns[1:]

    # Dynamically calculate the number of columns and rows
    num_metrics = len(metrics_columns)  # Number of metrics to plot
    num_cols = math.ceil(math.sqrt(num_metrics))  # Optimal number of columns
    num_rows = math.ceil(num_metrics / num_cols)  # Calculate rows based on columns

    plt.figure(figsize=(16, 12))  # Set figure size dynamically

    # Loop through each metric and create a Violin plot with Box plot overlay
    for i, metric in enumerate(metrics_columns, 1):
        plt.subplot(num_rows, num_cols, i)  # Dynamically set the subplot grid

        # Create the Violin plot with stick plot inside
        sns.violinplot(data=data, y=metric, inner="stick", color="lightgreen")

        # Create the Box plot overlaying the Violin plot
        sns.boxplot(data=data, y=metric, width=0.3, color="black")

        # Title for each subplot
        plt.title(f'Violin Plot with Box  Plot Overlay: {metric}', fontsize=12)

    # Adjust layout for better spacing
    plt.tight_layout()

    # Set a title for the entire figure
    plt.suptitle("Violin Plots with Box Plot \n Overlay for All Metrics", fontsize=16, y=1.02)

    # Display the plot in the Streamlit app
    st.pyplot(plt)  # Render the plot in Streamlit
    plt.clf()
###############################################
    num_metrics = len(metrics_columns)  # Number of metrics to plot

    # Dynamically calculate the number of columns and rows for the grid
    num_cols = math.ceil(math.sqrt(num_metrics))  # Optimal number of columns
    num_rows = math.ceil(num_metrics / num_cols)  # Calculate rows based on columns

    # Create the figure with a specific size
    plt.figure(figsize=(12, 8))  # Adjust the figure size as needed

    # Loop through each metric and create the mixed plot (Swarm + Violin + Boxplot)
    for i, metric in enumerate(metrics_columns, 1):  # Start with index 1
        plt.subplot(num_rows, num_cols, i)  # Dynamically set the subplot grid

        # Violin plot for distribution with stick representation for individual data points
        sns.violinplot(data=data, y=metric, inner="stick", color="lightblue")

        # Box plot for summary statistics (with no outliers displayed)
        sns.boxplot(data=data, y=metric, width=0.3, color="black", fliersize=0)

        # Swarm plot for individual data points with transparency
        sns.swarmplot(data=data, y=metric, color="orange", alpha=0.6)

        # Title for each subplot
        plt.title(f'Swarm + Violin + Boxplot: {metric}', fontsize=12)

    # Adjust layout for better spacing
    plt.tight_layout()

    # Set a main title for the entire figure
    plt.suptitle("Mixed Plot: Swarm + Violin + Boxplot for Metrics", fontsize=16, y=1.02)

    # Display the plot in Streamlit
    st.pyplot(plt)  # Render the plot in Streamlit
    plt.clf()
################################################
    metrics_columns = data.columns[1:]  # Exclude the first column (e.g., 'Models')
    num_metrics = len(metrics_columns)  # Total number of metrics to plot

    # Dynamically calculate the number of columns and rows for the grid
    num_cols = math.ceil(math.sqrt(num_metrics))  # Optimal number of columns
    num_rows = math.ceil(num_metrics / num_cols)  # Calculate rows based on columns

    plt.figure(figsize=(12, 8))  # Set figure size dynamically

    # Loop through each metric and create the mixed plot (Boxplot + Violin plot)
    for i, metric in enumerate(metrics_columns, 1):  # Start with index 1
        plt.subplot(num_rows, num_cols, i)  # Dynamically set the subplot grid

        # Violin plot with inner "stick" for better visualization of individual points
        sns.violinplot(data=data, y=metric, inner="stick", color="lightgreen", alpha=0.5)

        # Boxplot overlayed on the violin plot
        sns.boxplot(data=data, y=metric, width=0.3, color="black", fliersize=0)

        # Title for each subplot
        plt.title(f'Boxplot + Violin Plot:\n {metric}', fontsize=10)

    # Adjust layout for better spacing
    plt.tight_layout()

    # Set a title for the entire figure
    plt.suptitle("Mixed Plot: Boxplot + Violin Plot for Metrics", fontsize=16, y=1.02)

    # Display the plot in Streamlit
    st.pyplot(plt)  # Render the plot in Streamlit
    plt.clf()
###########################################################
    metrics_columns = data.columns[1:]  # Exclude the first column (e.g., 'Models')
    num_metrics = len(metrics_columns)  # Total number of metrics to plot

    # Dynamically calculate the number of columns and rows for the grid
    num_cols = math.ceil(math.sqrt(num_metrics))  # Optimal number of columns
    num_rows = math.ceil(num_metrics / num_cols)  # Calculate rows based on columns

    # Create the figure with a specific size
    plt.figure(figsize=(12, 8))  # Adjust the figure size as needed

    # Loop through each metric and create the mixed plot (KDE Plot + Boxplot)
    for i, metric in enumerate(metrics_columns, 1):  # Start with index 1
        plt.subplot(num_rows, num_cols, i)  # Dynamically set the subplot grid

        # KDE plot for density estimation
        sns.kdeplot(data=data[metric], color="blue", lw=2, fill=True)

        # Box plot overlay with no fliers
        sns.boxplot(data=data, y=metric, width=0.3, color="black", fliersize=0)

        # Title for each subplot
        plt.title(f'KDE Plot + Boxplot: {metric}')

    # Adjust layout for better spacing
    plt.tight_layout()

    # Set a main title for the entire figure
    plt.suptitle("Mixed Plot: KDE + Boxplot for Metrics", fontsize=16, y=1.02)

    # Display the plot in Streamlit
    st.pyplot(plt)  # Render the plot in Streamlit
    plt.clf()

############################################
    metrics_columns = data.columns[1:]  # Assuming the first column is 'Models'

    # Box Plot for Model Performance Metrics (horizontal box plot for each metric)
    plt.figure(figsize=(12, 8))
    data[metrics_columns].plot(kind='box', vert=False, figsize=(12, 8))
    plt.title("Box Plot of Model Performance Metrics")
    plt.xlabel("Metric Value")
    st.pyplot(plt)  # Render the plot in Streamlit
    plt.clf()


##########################################
    metrics_columns = data.columns[1:]

    mean_values = data[metrics_columns].mean()
    std_values = data[metrics_columns].std()

    plt.figure(figsize=(12, 8))
    data[metrics_columns].plot(kind='box', vert=False, figsize=(12, 8))

    for i, metric in enumerate(metrics_columns):
        plt.scatter(mean_values[metric], i, color='b', label=f'Mean - {metric}' if i == 0 else "", zorder=5, s=100)
        plt.scatter(std_values[metric], i, color='r', label=f'Standard Deviation - {metric}' if i == 0 else "",
                    zorder=5, s=100)

    plt.title("Box Plot of Model Performance Metrics with Mean and Standard Deviation", fontsize=14)
    plt.xlabel("Metric Value", fontsize=12)
    plt.legend(loc='best', fontsize=12)
    plt.tight_layout()

    st.pyplot(plt)
    plt.clf()
####################################################
    metrics_columns = data.columns[1:]  # Assuming the first column is 'Models'

    # Violin Plot for Distribution of Metrics Across Models
    violin_data = pd.DataFrame({
        'Model': np.repeat(data['Models'], len(data.columns) - 1),
        'Metric': np.tile(data.columns[1:], len(data)),
        'Value': data.drop(columns='Models').values.flatten()
    })

    # Set up the plot
    plt.figure(figsize=(14, 8))
    sns.violinplot(x='Metric', y='Value', hue='Model', data=violin_data, split=True, inner='quart', palette='muted')

    # Customize the plot
    plt.title('Violin Plot for Distribution of Metrics Across Models', fontsize=14)
    plt.xlabel('Metrics', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Show plot in Streamlit
    st.pyplot(plt)  # Render the plot in Streamlit
    plt.clf()
#####################################################
    metrics_columns = data.columns[1:]  # Assuming the first column is 'Models'

    # Calculate mean and standard deviation for each metric
    mean_values = data[metrics_columns].mean()
    std_values = data[metrics_columns].std()

    # Violin Plot with Mean and Standard Deviation
    plt.figure(figsize=(12, 8))

    # Plot the violin plot for each metric
    sns.violinplot(data=data[metrics_columns], inner="quart", linewidth=1)

    # Add mean and standard deviation lines
    for i, metric in enumerate(metrics_columns):
        plt.axvline(mean_values[metric], color='b', linestyle='--', label=f'Mean - {metric}' if i == 0 else "")
        plt.axvline(mean_values[metric] + std_values[metric], color='r', linestyle=':',
                    label=f'Standard Dev. (+1) - {metric}' if i == 0 else "")
        plt.axvline(mean_values[metric] - std_values[metric], color='r', linestyle=':',
                    label=f'Standard Dev. (-1) - {metric}' if i == 0 else "")

    # Labels and title
    plt.title("Violin Plot of Model Metrics with Mean and Standard Deviation", fontsize=14)
    plt.xlabel("Metric Value", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.legend(loc="best")
    plt.tight_layout()

    # Show plot in Streamlit
    st.pyplot(plt)  # Render the plot in Streamlit
    plt.clf()
########################################################
    metrics_columns = data.columns[1:]  # Exclude the first column (e.g., 'Models')

    # Dynamically calculate the number of columns and rows
    num_metrics = len(metrics_columns)
    num_cols = math.ceil(math.sqrt(num_metrics))  # Optimal number of columns
    num_rows = math.ceil(num_metrics / num_cols)  # Calculate rows based on columns

    # Create the figure with a specific size
    plt.figure(figsize=(12, 8))  # Adjust the figure size as needed

    # Loop through each metric and create the mixed plot (Violin Plot + KDE Plot)
    for i, metric in enumerate(metrics_columns, 1):
        plt.subplot(num_rows, num_cols, i)  # Dynamically set the subplot grid

        # Violin plot for distribution
        sns.violinplot(data=data, y=metric, color="lightblue")

        # KDE plot for density estimation
        sns.kdeplot(data=data[metric], color="black", linewidth=2)

        # Title for each subplot
        plt.title(f"{metric} Distribution with KDE")

    # Adjust layout for better spacing
    plt.tight_layout()

    # Set a main title for the entire figure
    plt.suptitle("Violin Plot with KDE for Each Metric", y=1.02, fontsize=16)

    # Display the plot in Streamlit
    st.pyplot(plt)  # Render the plot in Streamlit
    plt.clf()

    #############################################
    melted_data = data.melt(id_vars='Models', var_name='variable', value_name='value')

    sns.set(style="ticks")

    # Create FacetGrid for Box and Swarm Plot Comparison for Each Metric
    g = sns.FacetGrid(melted_data, col="variable", col_wrap=4, height=3)
    g.map(sns.boxplot, "value", "Models", color="lightblue")
    g.map(sns.swarmplot, "value", "Models", color="black", alpha=0.6)
    g.set_titles("{col_name}")

    # Set the main title and adjust layout
    g.fig.suptitle("Facet Grid: Box and Swarm Plot Comparison for Each Metric", y=1.02, fontsize=16)
    plt.tight_layout()

    # Display plot in Streamlit
    st.pyplot(g.fig)  # Render the FacetGrid in Streamlit
    plt.clf()
####################################################

    metrics_columns = data.columns[1:]  # Exclude the first column (e.g., 'Models')

    # Calculate number of rows and columns dynamically
    num_metrics = len(metrics_columns)
    num_cols = math.ceil(math.sqrt(num_metrics))  # Calculate optimal number of columns
    num_rows = math.ceil(num_metrics / num_cols)  # Calculate number of rows

    # Initialize figure for Density + KDE plots
    plt.figure(figsize=(12, 8))

    # Loop through each metric to create subplots
    for i, metric in enumerate(metrics_columns, 1):
        plt.subplot(num_rows, num_cols, i)  # Dynamically arrange subplots

        # KDE plot for smooth density
        sns.kdeplot(data=data[metric], color="green", lw=2, fill=True)

        # Density plot with dashed line
        sns.kdeplot(data=data[metric], color="blue", lw=2, fill=False, linestyle="--")

        # Title for each subplot
        plt.title(f'Density + KDE for {metric}')

    # Layout adjustments and main title
    plt.tight_layout()
    plt.suptitle("Mixed Plot: Density + KDE for Metrics", fontsize=16, y=1.02)

    # Display the plot in Streamlit
    st.pyplot(plt.gcf())  # Show the current figure in Streamlit
    plt.clf()
############################################

    data.columns = data.columns.str.strip()

    # حساب عدد الأعمدة والصفوف ديناميكيًا بناءً على عدد المقاييس
    num_metrics = len(data.columns[1:])  # استبعاد العمود الأول 'Models'
    num_cols = math.ceil(math.sqrt(num_metrics))  # عدد الأعمدة الأمثل
    num_rows = math.ceil(num_metrics / num_cols)  # عدد الصفوف

    # إعداد الشكل العام للمخططات
    plt.figure(figsize=(16, 12))

    # تكرار لكل مقياس لإنشاء المخططات الفرعية
    for i, metric in enumerate(data.columns[1:], 1):  # استبعاد عمود 'Models'
        plt.subplot(num_rows, num_cols, i)  # تحديد توزيع الصفوف والأعمدة ديناميكيًا
        sns.kdeplot(data[metric], shade=True, color='lightblue', alpha=0.7)
        plt.title(f"KDE Plot of {metric}", fontsize=12)
        plt.grid(True)

    # ضبط التخطيط بشكل جيد بين المخططات
    plt.tight_layout()

    # إضافة عنوان رئيسي للمخططات
    plt.suptitle('KDE Plots for Metrics Distribution', y=1.05, fontsize=16)

    # عرض المخطط في Streamlit
    st.pyplot(plt.gcf())  # عرض الشكل الحالي في Streamlit
    plt.clf()
############################
    data.columns = data.columns.str.strip()

    # Calculate optimal number of columns and rows dynamically
    num_metrics = len(data.columns[1:])  # Exclude 'Models' column
    num_cols = math.ceil(math.sqrt(num_metrics))  # Optimal number of columns
    num_rows = math.ceil(num_metrics / num_cols)  # Calculate number of rows based on columns

    # Set figure size and layout for subplots
    plt.figure(figsize=(15, 10))

    # Iterate through each metric to create histograms with KDE
    for i, metric in enumerate(data.columns[1:], 1):  # Exclude 'Models' column
        plt.subplot(num_rows, num_cols, i)
        sns.histplot(data[metric], kde=True, color='skyblue', bins=15)
        plt.title(f'Histogram & KDE: {metric}', fontsize=12)

    # Adjust layout for better spacing
    plt.tight_layout()
    plt.suptitle("Histograms with KDE for All Metrics", fontsize=16, y=1.02)

    # Render the plot in Streamlit
    st.pyplot(plt.gcf())  # Display the current figure
    plt.clf()
##################################
    metrics_columns = data.columns[1:]  # Exclude the first column (e.g., 'Models')

    # Calculate number of rows and columns dynamically
    num_metrics = len(metrics_columns)
    num_cols = math.ceil(math.sqrt(num_metrics))  # Calculate optimal number of columns
    num_rows = math.ceil(num_metrics / num_cols)  # Calculate number of rows

    # Initialize figure for Rug plots
    plt.figure(figsize=(12, 8))

    # Loop through each metric to create subplots
    for i, metric in enumerate(metrics_columns, 1):
        plt.subplot(num_rows, num_cols, i)  # Dynamically arrange subplots

        # Create Rug plot
        sns.rugplot(data[metric], color="green", height=0.05)

        # Title for each subplot
        plt.title(f'Rug Plot: {metric}')

    # Layout adjustments and main title
    plt.tight_layout()
    plt.suptitle("Rug Plots for All Metrics", fontsize=16, y=1.02)

    # Display the plot in Streamlit
    st.pyplot(plt.gcf())  # Show the current figure in Streamlit
    plt.clf()

#####################################

    data = df.copy()
    data.columns = data.columns.str.strip()

    # Set figure size and layout for subplots
    plt.figure(figsize=(12, 8))

    # Iterate through each metric to create Distribution Plots with KDE
    for i, metric in enumerate(data.columns[1:], 1):  # Exclude 'Models' column
        plt.subplot(2, 4, i)
        sns.histplot(data[metric], kde=True, color="skyblue", bins=20)
        plt.title(f'Distribution Plot with KDE: {metric}')

    # Adjust layout and set the overall title
    plt.tight_layout()
    plt.suptitle("Distribution Plots with KDE for Metrics", fontsize=16, y=1.02)

    # Render the plot in Streamlit
    st.pyplot(plt.gcf())  # Display the current figure
    plt.clf()

#################################
    df=data.copy()
    melted_data = pd.melt(df, id_vars=["Models"], var_name="Metric", value_name="Value")

    # Create a FacetGrid with bar plots for each metric
    g = sns.FacetGrid(melted_data, col="Metric", col_wrap=3, height=4, sharey=False)
    g.map(sns.barplot, "Models", "Value", palette="viridis", order=df["Models"])

    # Set titles, x-tick labels, and overall figure title
    g.set_titles("{col_name}")
    g.set_xticklabels(rotation=45)
    plt.subplots_adjust(top=0.9)
    g.fig.suptitle("Facet Grid of Model Performance Metrics")

    # Render the plot in Streamlit
    st.pyplot(plt.gcf())  # Display the current figure
    plt.clf()

###################################

    melted_metrics = metrics.melt(id_vars="Models", var_name="Metric", value_name="Value")

    # Create a FacetGrid with bar plots for each metric
    g = sns.FacetGrid(melted_metrics, col="Metric", col_wrap=3, height=4, sharey=False)
    g.map(sns.barplot, "Models", "Value", order=metrics["Models"])

    # Customize the appearance
    g.set_xticklabels(rotation=90)  # Rotate x-axis labels for better visibility
    g.fig.suptitle("Facet Grid of Model Performance Metrics", y=1.02, fontsize=16)  # Title for the entire grid

    # Display the plot in Streamlit
    st.pyplot(plt.gcf())  # Show the plot
    plt.clf()

#############################################

    metrics_columns = df.columns[1:]  # Excluding the first column (e.g., 'Models')

    # Dynamically calculate the number of rows and columns for subplots
    num_metrics = len(metrics_columns)
    num_cols = math.ceil(math.sqrt(num_metrics))  # Optimal number of columns
    num_rows = math.ceil(num_metrics / num_cols)  # Calculate number of rows

    # Create a figure for Q-Q plots of all metrics
    plt.figure(figsize=(15, 10))

    # Iterate through each metric and create Q-Q plots
    for i, metric in enumerate(metrics_columns, 1):
        plt.subplot(num_rows, num_cols, i)  # Dynamically arrange subplots
        stats.probplot(df[metric], dist="norm", plot=plt)  # Create Q-Q plot
        plt.title(f'Q-Q Plot: {metric}', fontsize=12)

    # Adjust the layout and add a title for the entire figure
    plt.tight_layout()
    plt.suptitle("Q-Q Plots for All Metrics", fontsize=16, y=1.02)

    # Render the plot in Streamlit
    st.pyplot(plt)
    plt.clf()


def section_2(df):
    st.subheader("Section 2 Analysis")

    data = df.copy()
    data.columns = data.columns.str.strip()  # Clean column names
    metrics= data.copy()
    # Melt the DataFrame to long format for plotting
    data_melted = data.melt(id_vars='Models', value_vars=data.columns[1:4])

    # Set the seaborn style
    sns.set(style="whitegrid")

    # Plot the barplot
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Models', y='value', hue='variable', data=data_melted)

    # Customize the plot
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.title("Model Comparison - MSE, RMSE, MAE")
    plt.ylabel("Error Metrics")
    plt.legend(title='Metric', loc="upper right", bbox_to_anchor=(1.1, 1))

    # Display the plot in Streamlit
    st.pyplot(plt.gcf())  # Show the plot
    plt.clf()

    ################################

    data.columns = data.columns.str.strip()  # Clean column names

    # Melt the DataFrame to long format for plotting (starting from the 5th column onward)
    data_melted = data.melt(id_vars='Models', value_vars=data.columns[5:])

    # Set the seaborn style
    sns.set(style="whitegrid")

    # Plot the barplot
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Models', y='value', hue='variable', data=data_melted)

    # Customize the plot
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.title("Model Comparison - MSE, RMSE, MAE")
    plt.ylabel("Error Metrics")
    plt.legend(title='Metric', loc="upper right", bbox_to_anchor=(1.1, 1))

    # Display the plot in Streamlit
    st.pyplot(plt.gcf())  # Show the plot
    plt.clf()

    ###########################################################
    metrics = df[data.columns[1:4]]  # Assuming these columns exist in the dataframe
    model_names =df[df.columns[0]].values  # Assuming there's a 'Model' column with model names
    mean_values = metrics.mean()  # Calculate the mean values for each metric
    std_values = metrics.std()  # Calculate the standard deviation values for each metric

    # Create the bar plot
    plt.figure(figsize=(12, 8))

    # Plot the bar plot for metrics
    ax = metrics.plot(kind='bar', figsize=(12, 8))

    # Adding mean and standard deviation bars for each metric
    for i, metric in enumerate(metrics.columns):
        plt.bar(i - 0.2, mean_values[metric], 0.4, label=f'Mean {metric}', color='b', alpha=0.6)
        plt.bar(i + 0.2, std_values[metric], 0.4, label=f'Standard Deviation {metric}', color='r', alpha=0.6)

    # Set X-ticks to model names
    ax.set_xticklabels(model_names, rotation=90, ha="right", fontsize=10)

    # Title and labels
    plt.title("Comparison of Models by MSE, RMSE, and MAE with Mean and Standard Deviation", fontsize=14)
    plt.ylabel("Metric Value", fontsize=12)
    plt.xlabel("Model", fontsize=12)
    plt.legend()

    # Show plot in Streamlit
    st.pyplot(plt)

########################################
    data = df.copy()
    data.columns = data.columns.str.strip()

    # Extract relevant columns for metrics (MSE, RMSE, MAE) from the DataFrame
    metrics = data.copy()

    # Calculate the mean and standard deviation for each metric
    metrics_mean = metrics.drop(data.columns[0], axis=1).mean()
    metrics_std = metrics.drop(data.columns[0], axis=1).std()

    # Plot the bar plot with error bars
    plt.figure(figsize=(12, 8))
    metrics_mean.plot(kind='bar', yerr=metrics_std, capsize=5, color='skyblue', edgecolor='black')

    # Adding titles and labels
    plt.title("Bar Plot of Model Performance with Error Bars", size=14)
    plt.ylabel("Performance Metric Value", size=12)
    plt.xticks(rotation=45, ha='right')

    # Show the plot
    plt.tight_layout()
    st.pyplot(plt)  # Render the plot in Streamlit
    plt.clf()


#####################

    data = df.copy()
    data.columns = data.columns.str.strip()

    # Extract relevant columns for metrics (MSE, RMSE, MAE) from the DataFrame
    metrics = data.copy()

    # Get the descriptive statistics for the metrics
    desc_stats = metrics.drop(data.columns[0], axis=1).describe().transpose()

    # Plot the descriptive statistics as a bar plot
    plt.figure(figsize=(12, 8))
    desc_stats[['mean', 'std']].plot(kind='bar', color=['skyblue', 'lightcoral'], edgecolor='black')

    # Adding titles and labels
    plt.title("Descriptive Statistics: Mean and Standard Deviation for Each Metric", size=14)
    plt.ylabel("Value", size=12)
    plt.xticks(rotation=45, ha='right')

    # Show the plot
    plt.tight_layout()
    st.pyplot(plt)  # Render the plot in Streamlit
    plt.clf()

####################
    plt.figure(figsize=(8, 6))
    correlation_matrix = data.drop(data.columns[0], axis=1).corr()  # Remove 'Models' column and compute correlation
    sns.heatmap(correlation_matrix, annot=True, cmap="YlGnBu", fmt=".2f")  # annot=True for values in cells
    plt.title("Correlation Matrix of Metrics")
    plt.tight_layout()
    st.pyplot(plt)  # Render the plot in Streamlit
    plt.clf()

############
    normalized_metrics = (metrics.iloc[:, 1:] - metrics.iloc[:, 1:].min()) / (
                metrics.iloc[:, 1:].max() - metrics.iloc[:, 1:].min())

    # Plot Heatmap of Normalized Metrics
    plt.figure(figsize=(10, 8))
    sns.heatmap(normalized_metrics, annot=True, cmap="YlGnBu", cbar_kws={'label': 'Normalized Scale (0-1)'})
    plt.title("Heatmap of Normalized Model Performance Metrics")
    plt.xlabel("Metrics")
    plt.ylabel("Models")
    plt.tight_layout()
    st.pyplot(plt)  # Render the plot in Streamlit
    plt.clf()


#########################
    best_metrics = metrics.iloc[:, 1:].min()  # Best (lowest) metric values across all models
    improvement_ratios = best_metrics / metrics.iloc[:, 1:]

    # Plot Improvement Ratio Heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(improvement_ratios, annot=True, cmap="coolwarm", cbar_kws={'label': 'Improvement Ratio'})
    plt.title("Improvement Ratio Matrix (Relative to Best Performance per Metric)")
    plt.xlabel("Metrics")
    plt.ylabel("Models")
    plt.tight_layout()
    st.pyplot(plt)  # Render the plot in Streamlit
    plt.clf()


############################

    performance_change = metrics.iloc[:, 1:].pct_change(axis=1) * 100  # Percent change across rows

    # Set the model names as the index for the performance_change DataFrame
    performance_change.index = metrics[data.columns[0]]

    # Plot the heatmap of performance change
    plt.figure(figsize=(12, 8))
    sns.heatmap(performance_change, annot=True, cmap="coolwarm", linewidths=0.5,
                cbar_kws={'label': 'Percentage Improvement'})
    plt.title("Heatmap of Percentage Improvement in Model Performance")
    plt.ylabel("Models")  # Add the 'Models' label for the y-axis
    plt.tight_layout()
    st.pyplot(plt)  # Render the plot in Streamlit
    plt.clf()


######################################
    corr_matrix = metrics.iloc[:, 1:].corr()  # Calculate correlation between the metrics

    # Perform hierarchical clustering using Ward's method
    linkage_matrix = sch.linkage(corr_matrix, method='ward')

    # Create and plot a clustered heatmap
    plt.figure(figsize=(12, 8))
    sns.clustermap(corr_matrix, row_linkage=linkage_matrix, col_linkage=linkage_matrix,
                   annot=True, cmap="coolwarm", figsize=(10, 8))
    plt.title("Clustered Heatmap of Performance Metric Correlations", size=16)
    st.pyplot(plt)  # Render the plot in Streamlit
    plt.clf()

#########################################

    mean_values = metrics.drop(data.columns[0], axis=1).mean()
    std_values = metrics.drop(data.columns[0], axis=1).std()

    # Create a DataFrame for mean and std
    mean_std_df = pd.DataFrame({
        'Mean': mean_values,
        'Standard Deviation': std_values
    })

    # Plot the heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(mean_std_df.T, annot=True, cmap="coolwarm", fmt=".4f", cbar=True, linewidths=0.5)

    # Add title and labels
    plt.title("Heatmap of Mean and Standard Deviation for Metrics", fontsize=14)
    plt.ylabel("Statistic", fontsize=12)
    plt.xlabel("Metrics", fontsize=12)

    # Render the plot in Streamlit
    st.pyplot(plt)
    plt.clf()

##########################
    data.columns = data.columns.str.strip()  # Strip any extra spaces

    heatmap_data = data[data.columns[1:]].copy()  # Select all columns except the first one
    heatmap_data['Model'] = data[data.columns[0]]  # Add the model names as a new column

    # Set the 'Model' column as the index
    heatmap_data.set_index('Model', inplace=True)

    # Plot the heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(heatmap_data, annot=True, cmap='coolwarm', fmt='.3f', linewidths=1, annot_kws={'size': 12})

    # Customize the plot
    plt.title('Heatmap of Model Metrics Comparison', fontsize=14)
    plt.tight_layout()

    # Render the plot in Streamlit
    st.pyplot(plt)
    plt.clf()
############################

    std_matrix = data.drop(data.columns[0], axis=1).std()

    # Plot the heatmap of standard deviation
    plt.figure(figsize=(10, 8))
    sns.heatmap(std_matrix.to_frame().T, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Heatmap of Standard Deviation Across Models for Each Metric', fontsize=16)
    plt.tight_layout()

    # Render the heatmap in Streamlit
    st.pyplot(plt)
    plt.clf()

#####################
    performance_change = data.iloc[:, 1:].pct_change(axis=1) * 100

    # Set the model names as the index for the performance_change DataFrame
    performance_change.index = data[data.columns[0]]

    # Plot the heatmap for percentage improvement
    plt.figure(figsize=(12, 8))
    sns.heatmap(performance_change, annot=True, cmap="coolwarm", linewidths=0.5,
                cbar_kws={'label': 'Percentage Improvement'})
    plt.title("Heatmap of Percentage Improvement in Model Performance")
    plt.ylabel("Models")  # Add the 'Models' label for the y-axis
    plt.tight_layout()

    # Render the heatmap in Streamlit
    st.pyplot(plt)
    plt.clf()

################
    linkage_matrix = sch.linkage(data[df.columns[1:4]], method='ward')

    # Create the dendrogram
    plt.figure(figsize=(12, 8))
    sch.dendrogram(linkage_matrix, labels=data[data.columns[0]].values, color_threshold=0)
    plt.title("Dendrogram for Hierarchical Clustering of Models", size=16)
    plt.xlabel('Models', size=12)
    plt.ylabel('Distance', size=12)
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Render the dendrogram in Streamlit
    st.pyplot(plt)
    plt.clf()



######################
    sns.pairplot(df.drop(df.columns[0], axis=1), kind='kde', diag_kind='kde', corner=True)
    plt.suptitle("Pairplot of Model Performance Metrics with KDE", y=1.02)
    plt.tight_layout()

    # Render the plot in Streamlit
    st.pyplot(plt)
    plt.clf()
#################
    sns.pairplot(df.iloc[:, 1:], kind="reg", plot_kws={"line_kws": {"color": "orange"}})
    plt.suptitle("Pair Plot with Regression Lines of Model Performance Metrics", y=1.02)
    plt.tight_layout()

    # Render the plot in Streamlit
    st.pyplot(plt)
    plt.clf()


###########
    metrics_pairplot = df.drop(df.columns[0], axis=1)

    # Plot pairplot for model performance metrics (scatter plot version)
    sns.pairplot(metrics_pairplot, kind='scatter', plot_kws={'alpha': 0.7}, height=2.5)
    plt.suptitle("Pairplot of Model Performance Metrics", size=16, y=1.02)
    plt.tight_layout()

    # Render the plot in Streamlit
    st.pyplot(plt)
    plt.clf()

######################
    sns.pairplot(df, hue=df.columns[0], corner=True, palette="tab10", diag_kind="kde")
    plt.suptitle("Scatter Plot Matrix for Model Performance Metrics", y=1.02)

    # Render the plot in Streamlit
    st.pyplot(plt)
    plt.clf()








    ##############################################
    metrics = data.copy()
    metrics_data = metrics.drop('Models', axis=1).mean().values
    labels = metrics.columns[1:]  # excluding the 'Models' column

    # Number of metrics
    num_vars = len(labels)

    # Compute angle for each axis
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

    # Make the plot circular by repeating the first value
    metrics_data = np.concatenate((metrics_data, [metrics_data[0]]))
    angles += angles[:1]

    # Create the radar plot
    fig, ax = plt.subplots(figsize=(8, 8), dpi=80, subplot_kw=dict(polar=True))
    ax.fill(angles, metrics_data, color='skyblue', alpha=0.25)
    ax.plot(angles, metrics_data, color='blue', linewidth=2)

    # Customize labels and title
    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=12)
    plt.title("Radar Plot of Model Performance Metrics", size=16)

    # Show the radar plot in Streamlit
    st.pyplot(fig)  # Show the radar plot
    plt.clf()
    ###################################

    def radar_chart_section(df):
        st.subheader("Radar Chart for Model Metrics")

        # Prepare the data
        data = df.copy()
        data.columns = data.columns.str.strip()  # Clean column names
        categories = list(data.columns[1:])  # Get all metric names except 'Models'
        values = data.drop('Models', axis=1).values  # Get the values without the 'Models' column

        # Create radar chart
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

        for i, model in enumerate(data['Models']):
            ax.plot([*categories, categories[0]], [*values[i], values[i][0]],
                    label=model)  # Close the loop by appending the first value
            ax.fill([*categories, categories[0]], [*values[i], values[i][0]],
                    alpha=0.2)  # Add shaded area for better visibility

        # Add title and legend
        plt.title("Radar Chart for Model Metrics")
        ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1.2))

        # Show the plot in Streamlit
        st.pyplot(fig)
        plt.clf()

    radar_chart_section(df)
#############################################
    st.subheader("Radar Plot: Metrics Mean and Std Across Models")
    metrics= data.copy()
    # Compute the mean and standard deviation for each metric
    metrics_mean = metrics.drop('Models', axis=1).mean()
    metrics_std = metrics.drop('Models', axis=1).std()

    # Create the angles for the radar plot (equal spacing for each metric)
    angles = np.linspace(0, 2 * np.pi, len(metrics.columns) - 1, endpoint=False).tolist()
    angles += angles[:1]  # Complete the loop (close the circle)

    # Set up the figure and axis for the radar plot
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    # Plot the mean values (as a blue line and fill)
    ax.plot(angles, metrics_mean.tolist() + [metrics_mean[0]], color='blue', linewidth=2, label='Mean')
    ax.fill(angles, metrics_mean.tolist() + [metrics_mean[0]], color='blue', alpha=0.3)

    # Plot the standard deviation as shaded areas around the mean
    ax.fill(angles, (metrics_mean + metrics_std).tolist() + [metrics_mean[0] + metrics_std[0]], color='orange',
            alpha=0.3, label='Mean + Std')
    ax.fill(angles, (metrics_mean - metrics_std).tolist() + [metrics_mean[0] - metrics_std[0]], color='orange',
            alpha=0.3)

    # Remove radial ticks and set the labels for each axis (metric)
    ax.set_yticklabels([])  # No radial ticks (values)
    ax.set_xticks(angles[:-1])  # Set x-axis ticks at the angles
    ax.set_xticklabels(metrics.columns[1:], fontsize=12)  # Label metrics on the axes

    # Set title and legend
    plt.title("Radar Plot: Metrics Mean and Std Across Models", fontsize=16)
    plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1))

    # Adjust layout for better fit
    plt.tight_layout()
    st.pyplot(fig)  # Show the plot in Streamlit
    plt.clf()


    ###############################################

    # Prepare data for radar plot
    metrics_data = metrics.drop('Models', axis=1).mean().values
    labels = metrics.columns[1:]  # excluding the 'Models' column

    # Number of metrics
    num_vars = len(labels)

    # Compute angle for each axis
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

    # Make the plot circular by repeating the first value
    metrics_data = np.concatenate((metrics_data, [metrics_data[0]]))
    angles += angles[:1]

    # Create the radar plot
    fig, ax = plt.subplots(figsize=(8, 8), dpi=80, subplot_kw=dict(polar=True))
    ax.fill(angles, metrics_data, color='skyblue', alpha=0.25)
    ax.plot(angles, metrics_data, color='blue', linewidth=2)

    # Customize labels and title
    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=12)
    plt.title("Radar Plot of Model Performance Metrics", size=16)

    # Show the radar plot in Streamlit
    st.pyplot(fig)  # Show the radar plot
    plt.clf()
##############################################


def section_3(df):
    st.subheader("Section 3 Analysis")
    data = df.copy()
    data.columns = data.columns.str.strip()  # Clean column names
    metrics = data.copy()


    plt.figure(figsize=(12, 6))
    parallel_coordinates(df, df.columns[0], color=sns.color_palette("husl", len(df)))
    plt.title("Parallel Coordinates Plot of Model Performance Metrics")
    plt.xlabel("Metrics")
    plt.ylabel("Values")
    plt.xticks(rotation=45)
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))

    # Render the plot in Streamlit
    st.pyplot(plt)
    plt.clf()
####################################
    metrics_mean = df.drop(df.columns[0], axis=1).mean()
    metrics_std = df.drop(df.columns[0], axis=1).std()

    plt.figure(figsize=(12, 8))

    # Plotting metrics means
    plt.plot(df.columns[1:], metrics_mean, label="Mean Metric", color='blue', marker='o')

    # Adding error bars based on standard deviation
    plt.errorbar(df.columns[1:], metrics_mean, yerr=metrics_std, fmt='o', color='blue', alpha=0.7, capsize=5)

    # Title and labels
    plt.title('Comparison of Model Metrics (Mean + Error Bars)', fontsize=16)
    plt.xlabel('Metrics')
    plt.ylabel('Value')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.legend()

    # Render the plot in Streamlit
    st.pyplot(plt)
    plt.clf()

    #######################
    x = np.arange(len(metrics))
    y = df[df.columns[1]].values

    # Perform cubic spline interpolation
    spline = CubicSpline(x, y)

    # Generate new x values for smooth interpolation
    x_smooth = np.linspace(x.min(), x.max(), 500)
    y_smooth = spline(x_smooth)

    # Plotting the spline interpolation
    plt.figure(figsize=(12, 8))
    plt.plot(x_smooth, y_smooth, label='Cubic Spline Interpolation', color='blue', lw=2)
    plt.scatter(x, y, color='red', label='Original Data Points')
    plt.title("Spline Interpolation for MSE Trend Across Models", size=16)
    plt.xlabel('Models', size=12)
    plt.ylabel('MSE', size=12)
    plt.legend()

    # Render the plot in Streamlit
    st.pyplot(plt)
    plt.clf()

###################

    normalized_metrics = metrics.copy()
    for col in df.columns[1:7]:
        normalized_metrics[col] = (metrics[col] - metrics[col].min()) / (metrics[col].max() - metrics[col].min())

    # Create the parallel coordinates plot
    plt.figure(figsize=(14, 10))
    parallel_coordinates(normalized_metrics, metrics.columns[0], cols=metrics.columns[1:], color=plt.cm.Set1.colors)
    plt.title("Parallel Coordinates Plot of Model Performance Metrics with Mean and Std", size=16)

    # Add the mean and std to the plot
    for i, metric in enumerate(metrics.columns[1:]):
        plt.axhline(y=normalized_metrics[metric].mean(), color='k', linestyle='--', label=f'Mean {metric}')
        plt.axhline(y=normalized_metrics[metric].mean() + normalized_metrics[metric].std(), color='r', linestyle=':', label=f'Standard Deviation {metric}')
        plt.axhline(y=normalized_metrics[metric].mean() - normalized_metrics[metric].std(), color='r', linestyle=':')

    # Adding x and y labels
    plt.xlabel('Metrics', size=12)
    plt.ylabel('Normalized Value', size=12)

    # Rotate x-ticks for better visibility
    plt.xticks(rotation=45)

    # Add legend
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))

    # Render the plot in Streamlit
    st.pyplot(plt)
    plt.clf()

##########################
    mean_values = metrics[data.columns[1:]].mean()
    std_values = metrics[data.columns[1:]].std()

    plt.figure(figsize=(12, 8))

    # Plot mean and standard deviation as lines
    plt.plot(mean_values.index, mean_values, label="Mean", color='b', marker='o', linestyle='-', linewidth=2)
    plt.plot(std_values.index, std_values, label="Standard Deviation", color='r', marker='x', linestyle='--',
             linewidth=2)

    # Adding labels and title
    plt.title("Line Plot of Mean and Standard Deviation for Metrics", fontsize=14)
    plt.xlabel("Metrics", fontsize=12)
    plt.ylabel("Value", fontsize=12)
    plt.legend(loc="upper right")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    # Render the plot in Streamlit
    st.pyplot(plt)
    plt.clf()

##########################
    metrics_columns=df.columns[1:]
    plt.figure(figsize=(12, 8))
    for i, metric in enumerate(metrics_columns, 1):
        sns.kdeplot(df[metric], bw_adjust=0.5, fill=True, label=metric, alpha=0.6)

    plt.legend(title="Metrics")
    plt.title("Stacked KDE Streamgraph of Metrics")

    # Render the plot in Streamlit
    st.pyplot(plt)
    plt.clf()
#############################################################################################
    # Create a contour plot with a scatter overlay for MAE vs RMSE
    df.columns = [df.columns[0]] + df.columns[1:].str.lower().tolist()

    plt.figure(figsize=(10, 8))
    sns.kdeplot(data=df, x="mae", y="rmse", cmap="viridis", fill=True)  # Contour plot
    sns.scatterplot(data=df, x="mae", y="rmse", color="white", edgecolor="black", alpha=0.5)  # Scatter plot overlay

    plt.title("Contour Plot with Scatter Overlay: MAE vs RMSE")

    # Render the plot in Streamlit
    st.pyplot(plt)
    plt.clf()




def main():
    st.title("Regrssion Analysis")
    st.write("Upload a CSV or Excel file and explore different types of plots.")

    uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])

    if uploaded_file is not None:
        df = load_file(uploaded_file)
        st.write("Data Preview:")
        st.dataframe(df.head())

        option = st.sidebar.selectbox("Choose a Section", ["Section 1", "Section 2", "Section 3"])

        if option == "Section 1":
            section_1(df)
        elif option == "Section 2":
            section_2(df)
        elif option == "Section 3":
            section_3(df)

if __name__ == "__main__":
    main()

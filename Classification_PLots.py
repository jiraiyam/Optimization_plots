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




sns.set(style="whitegrid")
plt.rcParams.update(
    {'font.weight': 'bold', 'axes.labelweight': 'bold', 'axes.titleweight': 'bold', 'axes.titlesize': 14})


def load_file(uploaded_file):
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith('.xlsx'):
        df = pd.read_excel(uploaded_file)
    return df


def section_1(df):
    st.subheader("Section 1: Basic Plots")

    # Bar Plots for each metric
    st.subheader("Metrics by Model")
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))  # Create a figure and axes grid
    axes = axes.flatten()  # Flatten the axes for easy indexing
    metrics = df.columns[1:]  # Assuming the first column is the model name
    for i, metric in enumerate(metrics, 1):
        sns.barplot(x='Models', y=metric, data=df, palette="viridis", ax=axes[i-1])  # Use the specific axis
        axes[i-1].set_xticklabels(axes[i-1].get_xticklabels(), rotation=45, ha='right')  # Rotate x-axis labels
        axes[i-1].set_title(f'{metric} by Model')
        axes[i-1].set_xlabel('Models')
        axes[i-1].set_ylabel(metric)
    plt.tight_layout()
    st.pyplot(fig)  # Pass the figure object to st.pyplot()
###################################################################
    # Violin Plots for each metric
    st.subheader("Distribution of Metrics Across Models (Violin Plots)")
    fig_violin, axes_violin = plt.subplots(2, 3, figsize=(14, 10))

    axes_violin = axes_violin.flatten()  # Flatten the axes for easy indexing
    for i, metric in enumerate(metrics, 1):
        sns.violinplot(x='Models', y=metric, data=df, palette="Set3", ax=axes_violin[i-1])
        axes_violin[i-1].set_xticklabels(axes_violin[i-1].get_xticklabels(), rotation=45, ha='right')
        axes_violin[i-1].set_title(f'{metric} Distribution by Model')
        axes_violin[i-1].set_xlabel('Models')
        axes_violin[i-1].set_ylabel(metric)

    plt.suptitle("Distribution of Metrics Across Classification Models", fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    st.pyplot(fig_violin)  # Pass the figure object to st.pyplot
###################################################################
    st.subheader("Swarm Plot of Metrics for Classification Models")
    fig_swarm, axes_swarm = plt.subplots(2, 3, figsize=(14, 10))  # Create a figure and axes grid
    axes_swarm = axes_swarm.flatten()  # Flatten the axes for easy indexing

    for i, metric in enumerate(metrics, 1):
        sns.swarmplot(x='Models', y=metric, data=df, palette="Paired", size=7,
                      ax=axes_swarm[i - 1])  # Use the specific axis
        axes_swarm[i - 1].set_xticklabels(axes_swarm[i - 1].get_xticklabels(), rotation=45,
                                          ha='right')  # Rotate x-axis labels
        axes_swarm[i - 1].set_title(f'{metric} by Model')
        axes_swarm[i - 1].set_xlabel('Models')
        axes_swarm[i - 1].set_ylabel(metric)

    plt.suptitle("Swarm Plot of Metrics for Classification Models", fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    st.pyplot(fig_swarm)
###############################################################################
    st.subheader("Strip Plot with Box Plot Overlay of Model Metrics")
    fig_boxstrip, axes_boxstrip = plt.subplots(2, 3, figsize=(12, 10))  # Create a figure and axes grid
    axes_boxstrip = axes_boxstrip.flatten()  # Flatten the axes for easy indexing

    for i, metric in enumerate(metrics, 1):
        sns.boxplot(x='Models', y=metric, data=df, color='lightgrey', ax=axes_boxstrip[i - 1])  # Boxplot
        sns.stripplot(x='Models', y=metric, data=df, jitter=True, alpha=0.6, color='blue',
                      ax=axes_boxstrip[i - 1])  # Stripplot
        axes_boxstrip[i - 1].set_xticklabels(axes_boxstrip[i - 1].get_xticklabels(), rotation=45,
                                             ha='right')  # Rotate x-axis labels
        axes_boxstrip[i - 1].set_title(f'{metric} by Model')
        axes_boxstrip[i - 1].set_xlabel('Models')
        axes_boxstrip[i - 1].set_ylabel(metric)

    plt.suptitle("Strip Plot with Box Plot Overlay of Model Metrics", fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    st.pyplot(fig_boxstrip)


#####################################################################################################
    st.subheader("Violin and Box Plots of Metrics Across Models")
    fig_violin_box, axes_violin_box = plt.subplots(2, 3, figsize=(14, 12))  # Create a figure and axes grid
    axes_violin_box = axes_violin_box.flatten()  # Flatten the axes for easy indexing

    for i, metric in enumerate(metrics, 1):
        sns.violinplot(x="Models", y=metric, data=df, inner=None, color="skyblue", linewidth=1,
                       ax=axes_violin_box[i - 1])  # Violin plot
        sns.boxplot(x="Models", y=metric, data=df, width=0.1, color="grey",
                    ax=axes_violin_box[i - 1])  # Box plot overlay
        axes_violin_box[i - 1].set_title(f"Violin and Box Plot of {metric}", fontsize=12)
        axes_violin_box[i - 1].set_xticklabels(axes_violin_box[i - 1].get_xticklabels(),
                                               rotation=45)  # Rotate x-axis labels

    plt.suptitle("Violin and Box Plots of Metrics Across Models", fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    st.pyplot(fig_violin_box)
#####################################################################################
    # Dot Plots (Strip Plots) for each metric
    st.subheader("Dot Plots for Metrics Across Models")
    fig_dot, axes_dot = plt.subplots(2, 3, figsize=(14, 8))  # Create a figure and axes grid
    axes_dot = axes_dot.flatten()  # Flatten the axes for easy indexing

    for i, metric in enumerate(metrics, 1):
        sns.stripplot(x="Models", y=metric, data=df, jitter=True, color='red', alpha=0.7,
                      ax=axes_dot[i - 1])  # Dot plot
        axes_dot[i - 1].set_title(f"Dot Plot for {metric}", fontsize=12)
        axes_dot[i - 1].set_xticklabels(axes_dot[i - 1].get_xticklabels(), rotation=45)  # Rotate x-axis labels

    plt.suptitle("Dot Plots for Metrics Across Models", fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    st.pyplot(fig_dot)  # Pass the figure object to st.pyplot()
#########################################################################

    st.subheader("Box Plots with Mean and Median for Metrics Across Models")

    plt.figure(figsize=(16, 12))

    for i, metric in enumerate(metrics, 1):
        plt.subplot(3, 3, i)

        # Boxplot for each metric
        sns.boxplot(x="Models", y=metric, data=df, palette="Set2")

        # Calculate mean and median
        mean_value = df[metric].mean()
        median_value = df[metric].median()

        # Plot mean and median as horizontal lines
        plt.axhline(y=mean_value, color='red', linestyle='--', label=f'Mean {metric}')
        plt.axhline(y=median_value, color='green', linestyle='-', label=f'Median {metric}')

        plt.title(f"Box Plot of {metric}\n with Mean and Median", fontsize=12)
        plt.xticks(rotation=45)

    # Only display the legend for the last subplot
    plt.legend(loc='upper left', bbox_to_anchor=(1.2, 1.1))

    plt.suptitle("Box Plots with Mean and Median for Metrics Across Models", fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    st.pyplot(plt.gcf())

####################################################################################

    plt.figure(figsize=(16, 12))

    for i, metric in enumerate(metrics, 1):
        plt.subplot(3, 3, i)

        # Violin plot for each metric
        sns.violinplot(x="Models", y=metric, data=df, inner="quart", color="lightblue")

        # Calculate mean and median
        mean_value = df[metric].mean()
        median_value = df[metric].median()

        # Plot mean and median as horizontal lines
        plt.axhline(y=mean_value, color='red', linestyle='--', label=f'Mean {metric}')
        plt.axhline(y=median_value, color='green', linestyle='-', label=f'Median {metric}')

        plt.title(f"Violin Plot of {metric}\n with Mean and Median", fontsize=12)
        plt.xticks(rotation=45)

    # Only display the legend once outside the subplot grid
    plt.legend(loc='upper left', bbox_to_anchor=(1.2, 1.1))

    plt.suptitle("Violin Plots with Mean and Median for Metrics Across Models", fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    st.pyplot(plt.gcf())
    ##########################################################################

    plt.figure(figsize=(17, 13))
    for i, metric in enumerate(metrics, 1):
        plt.subplot(3, 3, i)

        # Create swarm plot
        sns.swarmplot(x="Models", y=metric, data=df, color='orange')

        # Overlay mean and std dev
        mean_value = df[metric].mean()
        std_dev_value = df[metric].std()
        plt.axhline(mean_value, color='red', linestyle='--', label=f'Mean {metric}')
        plt.axhline(mean_value + std_dev_value, color='blue', linestyle='-.', label=f'Mean + Std Dev {metric}')
        plt.axhline(mean_value - std_dev_value, color='blue', linestyle='-.', label=f'Mean - Std Dev {metric}')

        plt.title(f"Swarm Plot for {metric} \n with Mean and Std Dev", fontsize=12)
        plt.xticks(rotation=45)

    plt.legend(loc='best', bbox_to_anchor=(0.8, 0.91))
    plt.suptitle("Swarm Plots for Metrics Across Models", fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    st.pyplot(plt.gcf())

######################################################################

    st.subheader("Categorical Plots with Mean and Median for Metrics Across Models")

    plt.figure(figsize=(14, 10))

    for i, metric in enumerate(metrics, 1):
        plt.subplot(3, 3, i)

        # Categorical strip plot with jitter
        sns.stripplot(x="Models", y=metric, data=df, jitter=True, color="blue", alpha=0.5)

        # Calculate mean and median
        mean_value = df[metric].mean()
        median_value = df[metric].median()

        # Overlay mean and median lines
        plt.axhline(mean_value, color='red', linestyle='--', label=f'Mean {metric}')
        plt.axhline(median_value, color='green', linestyle='-', label=f'Median {metric}')

        plt.title(f"Categorical Plot of {metric}\n with Mean and Median", fontsize=12)
        plt.xticks(rotation=45)

    # Display legend outside the plot grid
    plt.legend(loc='upper left', bbox_to_anchor=(1.6, 1.1))

    plt.suptitle("Categorical Plots with Mean and Median for Metrics Across Models", fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Render the plot in Streamlit
    st.pyplot(plt.gcf())



##################################################################################
    st.subheader("Kernel Density Estimation (KDE) Plots for Model Metrics")
    fig_kde, axes_kde = plt.subplots(2, 3, figsize=(14, 10))  # Create a figure and axes grid
    axes_kde = axes_kde.flatten()  # Flatten the axes for easy indexing

    for i, metric in enumerate(metrics, 1):
        sns.kdeplot(df[metric], shade=True, color="blue", label="All Models", ax=axes_kde[i-1])
        for model in df['Models']:
            sns.kdeplot(df[df['Models'] == model][metric], shade=True, label=model, ax=axes_kde[i-1])
        axes_kde[i-1].set_title(f"KDE Plot of {metric} for All Models", fontsize=12)
        axes_kde[i-1].legend()
        axes_kde[i-1].set_xticklabels(axes_kde[i-1].get_xticklabels(), rotation=45)

    plt.suptitle("Kernel Density Estimation Plots for Model Metrics", fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    st.pyplot(fig_kde)
######################################################################################

    # CDF Plots for each metric
    st.subheader("CDF Plots for Metrics Across Models")
    fig_cdf, axes_cdf = plt.subplots(2, 3, figsize=(14, 8))  # Create a figure and axes grid
    axes_cdf = axes_cdf.flatten()  # Flatten the axes for easy indexing

    for i, metric in enumerate(metrics, 1):
        sorted_data = np.sort(df[metric])  # Sort the data for the CDF
        cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)  # Compute the CDF
        axes_cdf[i - 1].plot(sorted_data, cdf, label=f'{metric} CDF', color='orange')
        axes_cdf[i - 1].set_title(f"CDF for {metric}", fontsize=12)
        axes_cdf[i - 1].set_xlabel(metric)
        axes_cdf[i - 1].set_ylabel("CDF")
        axes_cdf[i - 1].legend()

    plt.suptitle("CDF Plots for Metrics Across Models", fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    st.pyplot(fig_cdf)


####################################################################



#################################################################################
    st.subheader("Time-Series Analysis for Metrics Across Models")
    fig_time_series, axes_time_series = plt.subplots(2, 3, figsize=(14, 8))
    axes_time_series = axes_time_series.flatten()  # Flatten the axes for easy indexing

    for i, metric in enumerate(metrics, 1):
        axes_time_series[i - 1].plot(df['Models'], df[metric], marker='o', label=f'{metric}')
        axes_time_series[i - 1].set_title(f"Time-Series Plot for {metric}", fontsize=12)
        axes_time_series[i - 1].set_xlabel("Models")
        axes_time_series[i - 1].set_ylabel(metric)
        axes_time_series[i - 1].tick_params(axis='x', rotation=45)  # Rotate x-axis labels

        axes_time_series[i - 1].legend(loc='best')

    plt.suptitle("Time-Series Plots for Metrics Across Models", fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    st.pyplot(fig_time_series)


####################################################################

    st.subheader("Line Plots with Mean and Standard Deviation for Metrics Across Models")

    plt.figure(figsize=(14, 8))

    for i, metric in enumerate(metrics, 1):
        plt.subplot(3, 3, i)

        # Line plot for the metric across models
        plt.plot(df['Models'], df[metric], marker='o', linestyle='-', label=f'{metric} Trend')

        # Calculate mean and standard deviation
        mean_value = df[metric].mean()
        std_value = df[metric].std()

        # Plot mean and standard deviation lines
        plt.axhline(mean_value, color='red', linestyle='--', label=f'Mean {metric}')
        plt.axhline(mean_value + std_value, color='green', linestyle=':', label=f'Mean + Std Dev')
        plt.axhline(mean_value - std_value, color='green', linestyle=':', label=f'Mean - Std Dev')

        plt.title(f"Line Plot of {metric}\n with Mean and Std Dev", fontsize=12)
        plt.xticks(rotation=45)

    # Display legend outside the plot grid
    plt.legend(loc='upper left', bbox_to_anchor=(1.1, 1.1))

    plt.suptitle("Line Plots with Mean and Std Dev for Metrics Across Models", fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Render the plot in Streamlit
    st.pyplot(plt.gcf())


########################################################################

    st.subheader("Bar Plots with Error Bars for Metrics Across Models")
    fig_bar, axes_bar = plt.subplots(2, 3, figsize=(14, 8))
    axes_bar = axes_bar.flatten()  # Flatten the axes for easy indexing

    for i, metric in enumerate(metrics, 1):
        mean_val = df[metric].mean()
        std_dev = df[metric].std()

        axes_bar[i - 1].bar(df['Models'], df[metric], yerr=std_dev, capsize=5, label=f'{metric}', color='skyblue')
        axes_bar[i - 1].set_title(f"Bar Plot with Error Bars for {metric}", fontsize=12)
        axes_bar[i - 1].set_xticklabels(axes_bar[i - 1].get_xticklabels(), rotation=45, ha='right')  # Rotate x-axis labels
        axes_bar[i - 1].set_ylabel(metric)
        axes_bar[i - 1].legend(loc='best')

    plt.suptitle("Bar Plots with Error Bars for Metrics Across Models", fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    st.pyplot(fig_bar)

###################################################

    st.subheader("Histograms with Mean and Standard Deviation for Metrics Across Models")

    plt.figure(figsize=(14, 10))

    for i, metric in enumerate(metrics, 1):
        plt.subplot(3, 3, i)

        # Plot histogram with KDE for each metric
        sns.histplot(df[metric], kde=True, color="blue", label="Distribution", bins=20)

        # Calculate mean and standard deviation
        mean_value = df[metric].mean()
        std_value = df[metric].std()

        # Overlay mean and std deviation lines
        plt.axvline(mean_value, color='red', linestyle='--', label=f'Mean {metric}')
        plt.axvline(mean_value + std_value, color='green', linestyle=':', label=f'Mean + Std Dev {metric}')
        plt.axvline(mean_value - std_value, color='green', linestyle=':', label=f'Mean - Std Dev {metric}')

        plt.title(f"Histogram of {metric} \n with Mean and Std Dev", fontsize=12)

    # Display the legend outside the plot grid
    plt.legend(loc='upper left', bbox_to_anchor=(1.1, 1.1))

    plt.suptitle("Histograms with Mean and Std Dev for Metrics Across Models", fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Render the plot in Streamlit
    st.pyplot(plt.gcf())

    ##################################################


    fig_ecdf, axes_ecdf = plt.subplots(2, 3, figsize=(14, 8))
    axes_ecdf = axes_ecdf.flatten()  # Flatten the axes for easy indexing

    for i, metric in enumerate(metrics, 1):
        sns.ecdfplot(df[metric], ax=axes_ecdf[i - 1], color='blue', label="ECDF")

        # Calculate mean, median, and std deviation
        mean_value = df[metric].mean()
        median_value = df[metric].median()
        std_dev_value = df[metric].std()

        # Plot mean, median, and std
        axes_ecdf[i - 1].axvline(mean_value, color='red', linestyle='--', label=f'Mean {metric}')
        axes_ecdf[i - 1].axvline(median_value, color='green', linestyle='-', label=f'Median {metric}')
        axes_ecdf[i - 1].axvline(mean_value + std_dev_value, color='orange', linestyle=':',
                                 label=f'Mean + Std Dev {metric}')
        axes_ecdf[i - 1].axvline(mean_value - std_dev_value, color='orange', linestyle=':',
                                 label=f'Mean - Std Dev {metric}')

        axes_ecdf[i - 1].set_title(f"ECDF of {metric} with \n Mean and Std Dev", fontsize=12)

    plt.legend(loc='best', bbox_to_anchor=(1.1, 1.1))
    plt.suptitle("ECDF Plots with Mean and Std Dev for Metrics Across Models", fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    st.pyplot(fig_ecdf)
######################################################################

    st.subheader("Cumulative Distribution Plots for Metrics Across Models")
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()  # Flatten the axes for easy indexing

    for i, metric in enumerate(metrics, 1):
        ax = axes[i - 1]

        # Cumulative Distribution
        sns.histplot(df[metric], kde=False, cumulative=True, color='blue', bins=20, label=f"Cumulative {metric}", ax=ax)

        # Calculate mean, median, and std dev
        mean_value = df[metric].mean()
        median_value = df[metric].median()
        std_dev_value = df[metric].std()

        # Plot mean, median, and std dev
        ax.axvline(mean_value, color='red', linestyle='--', label=f'Mean {metric}')
        ax.axvline(median_value, color='green', linestyle='-', label=f'Median {metric}')
        ax.axvline(mean_value + std_dev_value, color='orange', linestyle=':', label=f'Mean + Std Dev')
        ax.axvline(mean_value - std_dev_value, color='orange', linestyle=':', label=f'Mean - Std Dev')

        ax.set_title(f"Cumulative Distribution of {metric}", fontsize=12)

    plt.legend(loc='best', bbox_to_anchor=(1.1, 1.1))
    plt.suptitle("Cumulative Distribution Plots for Metrics Across Models", fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Pass the figure to st.pyplot
    st.pyplot(fig)
############################################################################
    # Histogram and KDE Plots Section
    plt.figure(figsize=(14, 8))
    for i, metric in enumerate(metrics, 1):
        plt.subplot(3, 3, i)
        sns.histplot(df[metric], kde=True, color='skyblue', bins=15)
        plt.title(f"Histogram and KDE for {metric}", fontsize=12)
        plt.xlabel(metric)
        plt.ylabel('Density')
        plt.legend([f'{metric} Histogram & KDE'])

    plt.suptitle("Histogram and KDE Plots for Metrics Across Models", fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    st.pyplot(plt.gcf())
#################################################################
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(10, 12))
    fig.suptitle("Q-Q Plots for Normality Test of Each Metric", fontsize=16, fontweight='bold')

    axes = axes.flatten()  # Flatten the axes for easy indexing

    for i, metric in enumerate(metrics):
        stats.probplot(df[metric], dist="norm", plot=axes[i])
        axes[i].set_title(f"Q-Q Plot for {metric}", fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    st.pyplot(fig)

##################################################################################

    plt.figure(figsize=(12, 8))

    for i, metric in enumerate(metrics, 1):
        plt.subplot(2, 3, i)

        # Create histogram with KDE
        sns.histplot(df[metric], kde=True, stat='density', bins=20, color='skyblue', label='Data')

        # Generate the x values for the normal curve
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 100)

        # Calculate normal distribution curve
        p = norm.pdf(x, df[metric].mean(), df[metric].std())

        # Plot the normal curve
        plt.plot(x, p, 'k', linewidth=2, label='Normal Curve')

        plt.title(f'Histogram with Normal Curve for {metric}', fontsize=12)
        plt.legend()

    plt.suptitle("Histograms with Normal Distribution Curve for Metrics Across Models", fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    st.pyplot(plt.gcf())


############################################################################
# Function to display plots for Section 2
def section_2(df):
    metrics = df.columns[1:]  # Assuming the first column is the model name

    # Radar Plot for Model Performance Metrics
    st.subheader("Radar Chart Highlighting {selected_model}")

    labels = metrics
    num_vars = len(labels)

    angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
    angles += angles[:1]  # Complete the loop

    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))

    for i, row in df.iterrows():
        values = row[metrics].tolist()
        values += values[:1]  # Complete the loop
        ax.plot(angles, values, label=row['Models'], linewidth=2)
        ax.fill(angles, values, alpha=0.25)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_yticklabels([])  # Hide radial ticks
    ax.set_title("Radar Plot of Model Performance Metrics", size=16, weight='bold')

    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    st.pyplot(fig)
####################################################################################

    st.subheader("Radar Plot of Model Performance Metrics")

    df_normalized = df.copy()
    for metric in metrics:
        df_normalized[metric] = (df[metric] - df[metric].min()) / (df[metric].max() - df[metric].min())

    selected_model = df['Models'][0]  # Choose the model you want to highlight
    labels = metrics
    angles = [n / float(len(labels)) * 2 * pi for n in range(len(labels))]
    angles += angles[:1]  # Close the loop for the radar chart

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    # Plot each model's metrics on the radar chart
    for i, row in df_normalized.iterrows():
        values = row[metrics].tolist() + [row[metrics[0]]]  # Close the radar plot loop
        if row['Models'] == selected_model:
            ax.plot(angles, values, linewidth=3, linestyle='solid', label=row['Models'], color='blue')
            ax.fill(angles, values, alpha=0.3, color='blue')
        else:
            ax.plot(angles, values, linewidth=1, linestyle='dashed', color='gray')

    # Set up chart labels and title
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_yticklabels([])

    plt.title(f"Radar Chart Highlighting {selected_model}", size=16, fontweight='bold')
    plt.legend(loc='upper right', bbox_to_anchor=(1.4, 1))
    st.pyplot(fig)


    ##############################################################################

    st.subheader(" Circular Correlation Plot for Model Metrics")

    # Calculate the correlation matrix for the selected metrics
    corr_matrix = df[metrics].corr()

    # Number of metrics
    N = len(corr_matrix)

    # Calculate angles for circular plot
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    # Draw the correlations in the circular plot
    for i in range(N):
        ax.plot([angles[i], angles[(i + 1) % N]], [corr_matrix.iloc[i, (i + 1) % N], corr_matrix.iloc[i, (i + 1) % N]],
                color='blue', linewidth=2)

    # Set up circular labels
    ax.set_yticklabels([])
    ax.set_xticks(angles)
    ax.set_xticklabels(corr_matrix.columns, fontsize=12, color='black')

    # Title and layout
    ax.set_title("Circular Correlation Plot for Model Metrics", fontsize=16, fontweight='bold')
    plt.tight_layout()

    # Display in Streamlit
    st.pyplot(fig)
###############################################################

    st.subheader(" Correlation Heatmap of Model Metrics")

    corr = df[metrics].corr()

    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, square=True, linewidths=0.5)
    plt.title("Correlation Heatmap of Model Metrics", fontsize=16, fontweight='bold')

    st.pyplot(plt.gcf())


##########################################################

    st.subheader(" Z-Score Heatmap of Model Performance Metrics")

    # Calculate Z-scores for each metric
    df_zscores = df[metrics].apply(zscore)

    # Set up the figure for the Z-score heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(df_zscores, annot=True, cmap='coolwarm', center=0, linewidths=0.5)
    plt.title("Z-Score Heatmap of Model Performance Metrics", fontsize=16, fontweight='bold')

    # Adjust x-ticks to align with model names
    plt.xticks(ticks=[i + 0.5 for i in range(len(df))], labels=df['Models'], rotation=45, ha="right")
    plt.yticks(rotation=0)

    # Display the heatmap in Streamlit
    st.pyplot(plt.gcf())

############################################################

    st.subheader("Seaborn Heatmap with Annotations for Metric Comparisons Across Models")

    # Set up the figure for the heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(df[metrics].transpose(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, square=True)
    plt.title("Seaborn Heatmap with Annotations for Metric Comparisons Across Models", fontsize=16, fontweight='bold')

    # Adjust layout for display in Streamlit
    plt.tight_layout()

    # Display the heatmap in Streamlit
    st.pyplot(plt.gcf())

####################################################################

    st.subheader("Detailed Heatmap of Model Metrics with Annotations")

    # Set up the figure for the heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(df[metrics].T, annot=True, fmt=".4f", cmap="YlGnBu", linewidths=0.5, cbar=True)

    # Customize tick labels for models and metrics
    plt.xticks(ticks=[x + 0.5 for x in range(len(df))], labels=df['Models'], rotation=45, ha="right")
    plt.yticks(rotation=0)

    # Add title and adjust layout
    plt.title("Detailed Heatmap of Model Metrics with Annotations", fontsize=16, fontweight='bold')
    plt.tight_layout()

    # Display the plot in Streamlit
    st.pyplot(plt.gcf())

##############################################################

    st.subheader(" Model Performance with Mean, Median, and Std Dev")

    means = df[metrics].mean()
    medians = df[metrics].median()
    std_devs = df[metrics].std()

    performance_df = pd.DataFrame({
        'Mean': means,
        'Median': medians,
        'Std Dev': std_devs
    }).T

    plt.figure(figsize=(12, 8))
    sns.heatmap(performance_df, annot=True, cmap='Blues', linewidths=0.5, fmt='.2f', cbar_kws={'label': 'Score'})

    plt.title("Model Performance with Mean, Median, and Std Dev", fontsize=16, fontweight='bold')
    plt.tight_layout()

    st.pyplot(plt.gcf())
#######################################################

    st.subheader("Section 2: Heatmap of Mean and Standard Deviation for Each Metric")

    # Calculate mean and standard deviation for each metric
    mean_values = df[metrics].mean()
    std_values = df[metrics].std()

    # Set up the subplot figure for heatmaps of mean and std dev
    fig, ax = plt.subplots(1, 2, figsize=(16, 8))

    # Heatmap for the mean values
    sns.heatmap(mean_values.to_frame().T, annot=True, cmap='coolwarm', ax=ax[0], fmt=".2f")
    ax[0].set_title("Heatmap of Mean for Each Metric", fontsize=16, fontweight='bold')

    # Heatmap for the standard deviation values
    sns.heatmap(std_values.to_frame().T, annot=True, cmap='coolwarm', ax=ax[1], fmt=".2f")
    ax[1].set_title("Heatmap of Standard Deviation for Each Metric", fontsize=16, fontweight='bold')

    plt.tight_layout()

    # Display the plot in Streamlit
    st.pyplot(fig)

##########################################

    st.subheader(" Heatmap with Hierarchical Clustering of Model Metrics")

    # Calculate the correlation matrix of the metrics
    corr = df[metrics].corr()

    # Perform hierarchical clustering on the correlation matrix
    Z = linkage(corr, method='ward')

    # Plot the clustered heatmap
    clustermap_fig = sns.clustermap(
        corr, row_linkage=Z, col_linkage=Z, cmap='coolwarm', annot=True,
        figsize=(12, 8), linewidths=1, fmt='.2f'
    )
    plt.title("Heatmap with Hierarchical Clustering of Model Metrics", fontsize=16, fontweight='bold')
    plt.tight_layout()

    st.pyplot(clustermap_fig)

############################################################
    st.subheader(" Clustermap for Metrics Clustering Across Models")

    # Create the clustermap to visualize clustering of models based on metrics
    clustermap_fig = sns.clustermap(
        df[metrics].transpose(), method='ward', cmap='coolwarm', annot=True,
        figsize=(12, 8), linewidths=0.5, fmt='.2f'
    )

    # Set title for the clustermap
    plt.title("Clustermap for Metrics Clustering Across Models", fontsize=16, fontweight='bold')
    plt.tight_layout()

    # Display the clustermap in Streamlit
    st.pyplot(clustermap_fig)


#############################################

    st.subheader(" Cluster Map for Model Performance Metrics")

    # Plot the clustermap for model performance metrics
    clustermap_fig = sns.clustermap(
        df[metrics], method='ward', cmap='viridis', row_cluster=True, col_cluster=True,
        standard_scale=1, figsize=(10, 8)
    )

    # Set title for the clustermap
    plt.title("Cluster Map for Model Performance Metrics", fontsize=16, fontweight='bold')

    # Display the clustermap in Streamlit
    st.pyplot(clustermap_fig)

#####################################
    st.subheader(" Pairplot of Metrics with Regression Lines")

    # Create the pairplot with regression lines
    pairplot_fig = sns.pairplot(
        df[metrics], kind='reg', diag_kind='kde',
        plot_kws={'line_kws': {'color': 'red'}}
    )

    # Set the title and adjust layout
    pairplot_fig.fig.suptitle("Pairplot of Metrics with Regression Lines", y=1.02, fontsize=16, fontweight='bold')

    # Display the pairplot in Streamlit
    st.pyplot(pairplot_fig)

#######################################################

    st.subheader(" Pairwise Scatter Matrix of Metrics with Density Plots")

    # Create the pairplot with scatter plots and density on the diagonal
    scatter_matrix_fig = sns.pairplot(
        df[metrics],
        diag_kind="kde",
        kind="scatter",
        plot_kws={'alpha': 0.7, 's': 80, 'edgecolor': 'k'},
        diag_kws={'shade': True}
    )

    # Set title and layout adjustments
    scatter_matrix_fig.fig.suptitle("Pairwise Scatter Matrix of Metrics with Density Plots", y=1.02, fontsize=16,
                                    fontweight='bold')

    # Display the scatter matrix in Streamlit
    st.pyplot(scatter_matrix_fig)

###########################################################

    st.subheader(" Pairwise KDE Plot of Model Performance Metrics")

    # Create the KDE pairplot with shading and transparency
    kde_pairplot_fig = sns.pairplot(
        df[metrics],
        kind="kde",
        plot_kws={'shade': True, 'alpha': 0.7}
    )

    # Set title and adjust layout
    kde_pairplot_fig.fig.suptitle("Pairwise KDE Plot of Model Performance Metrics", y=1.02, fontsize=16,
                                  fontweight='bold')

    # Display the KDE pairplot in Streamlit
    st.pyplot(kde_pairplot_fig)







# Function to display plots for Section 3
def section_3(df):
    metrics = df.columns[1:]  # Assuming the first column is the model name
    st.subheader("Parallel Coordinates Plot of Model Metrics")

    # Normalize the data for consistent scale in visualization
    df_normalized = df.copy()
    for metric in metrics:
        df_normalized[metric] = (df[metric] - df[metric].min()) / (df[metric].max() - df[metric].min())

    # Set up the figure for parallel coordinates
    plt.figure(figsize=(12, 8))
    parallel_coordinates(
        df_normalized,
        'Models',
        cols=metrics,
        color=sns.color_palette("Set2", len(df['Models'].unique()))
    )
    plt.title("Parallel Coordinates Plot of Model Metrics", fontsize=16, fontweight='bold')
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.xticks(rotation=45)

    # Display plot in Streamlit
    st.pyplot(plt)

    st.subheader("Section 3: Andrews Curves for Model Performance Metrics")

    # Andrews Curves plot
    plt.figure(figsize=(10, 6))
    andrews_curves(df, 'Models', colormap='viridis', alpha=0.8)
    plt.title("Andrews Curves for Model Performance Metrics", fontsize=16, fontweight='bold')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel('X', fontsize=14)
    plt.ylabel('Y', fontsize=14)
    plt.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1))

    # Display plot in Streamlit
    st.pyplot(plt)
#########################################################

    st.subheader("Section 3: Bump Chart of Model Rankings Across Metrics")

    # Calculate rankings for each model based on each metric
    ranked_df = df.set_index('Models')[metrics].rank(ascending=False)

    # Plot Bump Chart
    fig, ax = plt.subplots(figsize=(10, 8))
    for i, model in enumerate(ranked_df.index):
        ax.plot(ranked_df.columns, ranked_df.loc[model], label=model, marker='o', linewidth=2)

    ax.set_ylim(ranked_df.max().max() + 0.5, ranked_df.min().min() - 0.5)
    ax.set_ylabel('Rank')
    ax.set_title("Bump Chart of Model Rankings Across Metrics", fontsize=16, fontweight='bold')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xticks(rotation=45)

    # Display plot in Streamlit
    st.pyplot(fig)

#############################################################

    st.subheader("Section 3: Line Plot with Confidence Intervals for Metrics Across Models")

    # Define a function to calculate confidence intervals assuming normal distribution
    def ci(data, confidence=0.95):
        ci_range = 1.96 * np.std(data) / np.sqrt(len(data))
        return np.mean(data) - ci_range, np.mean(data) + ci_range

    # Create the line plot with confidence intervals
    plt.figure(figsize=(14, 8))

    for metric in metrics:
        # Calculate confidence intervals for each metric
        lower, upper = ci(df[metric])
        plt.plot(df['Models'], df[metric], label=metric, marker='o')
        plt.fill_between(df['Models'], lower, upper, alpha=0.2)

    plt.title("Line Plot with Confidence Intervals for Metrics Across Models", fontsize=16, fontweight='bold')
    plt.xlabel('Models')
    plt.ylabel('Metric Value')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()

    # Display the plot in Streamlit
    st.pyplot(plt.gcf())
######################################################################

    st.subheader("Section 3: Stacked Bar Chart of Metrics by Model")

    # Calculate the sum of selected metrics for each model
    df['Total_Score'] = df[metrics].sum(axis=1)

    # Plot stacked bar chart
    plt.figure(figsize=(10, 6))
    df.set_index('Models')[metrics].plot(kind='bar', stacked=True, colormap='viridis', edgecolor='black')
    plt.title("Stacked Bar Chart of Metrics by Model", fontsize=16, fontweight='bold')
    plt.xlabel('Models')
    plt.ylabel('Metric Score')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title="Metrics", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    # Display the plot in Streamlit
    st.pyplot(plt.gcf())

####################################

    st.subheader("Section 3: Waterfall Chart for Model Metric Breakdown")

    # Data for Waterfall Chart (first row of metrics values for each model)
    performance_metrics = df[metrics].iloc[0].values
    labels = metrics

    # Create the Waterfall plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(labels, performance_metrics, color='skyblue')

    # Annotating the bars with values
    for i, value in enumerate(performance_metrics):
        ax.text(i, value + 0.02, f'{value:.2f}', ha='center', fontsize=12)

    # Add title and labels
    plt.title("Waterfall Chart for Model Metric Breakdown", fontsize=16, fontweight='bold')
    plt.xticks(rotation=45)

    # Tight layout for better spacing
    plt.tight_layout()

    # Display the plot in Streamlit
    st.pyplot(plt)
########################################################################

    st.subheader("Section 3: Grouped Bar Plot for Model Metrics Comparison")

    # Reshape data for grouped bar plot
    df_melted = df.melt(id_vars=['Models'], value_vars=metrics, var_name='Metric', value_name='Value')

    # Create grouped bar plot
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Models', y='Value', hue='Metric', data=df_melted, palette='Set1')

    # Customize plot
    plt.title("Grouped Bar Plot for Model Metrics Comparison", fontsize=16, fontweight='bold')
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Display the plot in Streamlit
    st.pyplot(plt)


#################################################################
    st.subheader("Section 3: Combined Bar Plot and Line Plot for FScore and Accuracy")

    # Create figure and axes
    fig, ax1 = plt.subplots(figsize=(14, 8))

    # Bar plot for FScore
    ax1.bar(df['Models'], df['FScore'], color='lightblue', label='FScore')
    ax1.set_xlabel('Models')
    ax1.set_ylabel('FScore', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')

    # Line plot for Accuracy
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.plot(df['Models'], df['Accuracy'], color='orange', marker='o', label='Accuracy')
    ax2.set_ylabel('Accuracy', color='orange')
    ax2.tick_params(axis='y', labelcolor='orange')

    # Title and labels
    plt.title("Combined Bar Plot and Line Plot for FScore and Accuracy", fontsize=16, fontweight='bold')
    plt.xticks(rotation=45)

    # Adjust layout for better fitting
    fig.tight_layout()

    # Display the plot in Streamlit
    st.pyplot(fig)

#####################################################################
    st.subheader("Section 3: Dual Bar and Line Plot for FScore and Sensitivity")

    # Create figure and axes
    fig, ax1 = plt.subplots(figsize=(14, 8))

    # Bar plot for FScore
    ax1.bar(df['Models'], df['FScore'], color='lightblue', label='FScore')
    ax1.set_xlabel('Models')
    ax1.set_ylabel('FScore', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')

    # Line plot for Sensitivity (TRP)
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.plot(df['Models'], df['Sensitivity (TRP)'], color='green', marker='o', label='Sensitivity (TRP)', linewidth=2)
    ax2.set_ylabel('Sensitivity (TRP)', color='green')
    ax2.tick_params(axis='y', labelcolor='green')

    # Title and labels
    plt.title("Dual Bar and Line Plot for FScore and Sensitivity", fontsize=16, fontweight='bold')
    plt.xticks(rotation=45)

    # Adjust layout for better fitting
    fig.tight_layout()

    # Display the plot in Streamlit
    st.pyplot(fig)

    st.subheader("Section 3: Dual Axis Plot for Accuracy and Specificity")

    # Create figure and axes
    fig, ax1 = plt.subplots(figsize=(14, 8))

    # Scatter plot for Accuracy
    ax1.scatter(df['Models'], df['Accuracy'], color='blue', label='Accuracy', zorder=5)
    ax1.set_xlabel('Models')
    ax1.set_ylabel('Accuracy', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')

    # Line plot for Specificity (TNP)
    ax2 = ax1.twinx()  # Create second y-axis sharing the same x-axis
    ax2.plot(df['Models'], df['Specificity(TNP)'], color='orange', marker='o', label='Specificity (TNP)', linewidth=2,
             zorder=3)
    ax2.set_ylabel('Specificity (TNP)', color='orange')
    ax2.tick_params(axis='y', labelcolor='orange')

    # Title and labels
    plt.title("Dual Axis Plot for Accuracy (Scatter) and Specificity (TNP) (Line)", fontsize=16, fontweight='bold')
    plt.xticks(rotation=45)

    # Adjust layout for better fitting
    fig.tight_layout()

    # Display the plot in Streamlit
    st.pyplot(fig)

    st.subheader("Section 3: Correlation Network Plot for Metrics")

    # Calculate the correlation matrix
    corr = df[metrics].corr()

    # Create an empty graph
    G = nx.Graph()

    # Add nodes for each metric
    for metric in corr.columns:
        G.add_node(metric)

    # Add edges between metrics with correlation > 0.5
    for i in corr.columns:
        for j in corr.columns:
            if i != j and abs(corr[i][j]) > 0.5:  # Threshold correlation for edges
                G.add_edge(i, j, weight=corr[i][j])

    # Draw the network plot
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G, seed=42)  # Layout of the graph
    nx.draw_networkx_nodes(G, pos, node_size=1000, node_color='lightblue', alpha=0.6)
    nx.draw_networkx_edges(G, pos, width=2, alpha=0.6, edge_color='black')
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold', font_color='black')

    # Set plot title
    plt.title("Correlation Network Plot for Metrics", fontsize=16, fontweight='bold')
    plt.tight_layout()

    # Display the plot in Streamlit
    st.pyplot(plt)

#######################################################################
    st.subheader("Section 3: KDE Plot Comparing Multiple Metrics Across Models")

    # Create KDE plot
    plt.figure(figsize=(14, 8))
    for metric in metrics:
        sns.kdeplot(df[metric], label=metric, shade=True)

    # Customize the plot
    plt.title("KDE Plot Comparing Multiple Metrics Across Models", fontsize=16, fontweight='bold')
    plt.xlabel('Metric Value')
    plt.ylabel('Density')
    plt.legend(title='Metrics')

    # Display the plot in Streamlit
    plt.tight_layout()
    st.pyplot(plt)

####################################

    st.subheader("Section 3: Violin Plot with KDE for Accuracy Across Models")

    # Create the violin plot
    plt.figure(figsize=(12, 8))
    sns.violinplot(x='Models', y='Accuracy', data=df, inner='stick', color='lightgreen', linewidth=1.5)

    # Overlay the KDE plot for Accuracy
    sns.kdeplot(df['Accuracy'], color='blue', fill=True, alpha=0.4)

    # Customize the plot
    plt.title("Violin Plot with KDE for Accuracy Across Models", fontsize=16, fontweight='bold')
    plt.xticks(rotation=45)

    # Display the plot in Streamlit
    plt.tight_layout()
    st.pyplot(plt)
# Streamlit App Layout
def main():
    st.title("Classification Analysis ")
    st.write("Upload a CSV or Excel file and explore different types of plots.")

    uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])

    if uploaded_file is not None:
        df = load_file(uploaded_file)
        st.write(f"Data Preview:")
        st.dataframe(df.head())

        # Sidebar Navigation
        option = st.sidebar.selectbox("Choose a Section", ["Section 1",
                                                           "Section 2",
                                                           "Section 3"])

        if option == "Section 1":
            section_1(df)
        elif option == "Section 2":
            section_2(df)
        elif option == "Section 3":
            section_3(df)


if __name__ == "__main__":
    main()

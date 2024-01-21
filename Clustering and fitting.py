# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 00:29:54 2024

@author: Bharani
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.optimize import curve_fit

# Define scaler and kmeans_model as global variables
scaler = StandardScaler()
kmeans_model = None

def read_and_filter_data(file_path, indicator1, indicator2, year):
    """
    Read and filter data based on indicator names and a specific year.

    Parameters:
    - file_path: str, path to the CSV file
    - indicator1: str, name of the first indicator
    - indicator2: str, name of the second indicator
    - year: str, the target year

    Returns:
    - merged_data: pd.DataFrame, merged data containing specified indicators for the given year
    """
    df = pd.read_csv(file_path, skiprows=3)
    data1 = df.loc[df['Indicator Name'] == indicator1, ['Country Name', 
                                                        'Country Code', year]
                   ].rename(columns={year: indicator1})
    data2 = df.loc[df['Indicator Name'] == indicator2, [
        'Country Name', 'Country Code',
        year]].rename(columns={year: indicator2})
    merged_data = pd.merge(data1, data2, on=['Country Name', 'Country Code'],
                           how='outer').dropna(how='any')
    return merged_data

def read_file(file_path, indicator1, country):
    """
    Read and filter data for a specific indicator and country.

    Parameters:
    - file_path: str, path to the CSV file
    - indicator1: str, name of the indicator
    - country: str, name of the country

    Returns:
    - data_t: pd.DataFrame, transposed and processed data for the specified indicator and country
    """
    df = pd.read_csv(file_path, skiprows=3)
    data3 = df.loc[(df['Indicator Name'] == indicator1) & (df[
        'Country Name'] == country)]
    data3 = data3.drop(['Country Code', 'Indicator Code', 'Indicator Name',
                        'Unnamed: 67'], axis=1).reset_index(drop=True)
    data_t = data3.transpose()
    data_t.columns = data_t.iloc[0]
    data_t = data_t.iloc[1:]
    data_t.index = pd.to_numeric(data_t.index)
    data_t['Years'] = pd.to_numeric(data_t.index)
    data_t = data_t.apply(pd.to_numeric, errors='coerce')
    return data_t

def kmeans_clustering(data, cluster_columns, num_clusters=4):
    """
    Apply KMeans clustering to the specified data.

    Parameters:
    - data: pd.DataFrame, input data for clustering
    - cluster_columns: list, columns used for clustering
    - num_clusters: int, number of clusters for KMeans

    Returns:
    - data: pd.DataFrame, input data with an additional 'Cluster_KMeans' column
    - scaler: StandardScaler, scaler used for data scaling
    - kmeans_model: KMeans, trained KMeans model
    """
    global scaler, kmeans_model 
    if data.empty:
        return data

    kmeans_model = KMeans(n_clusters=num_clusters, random_state=42)

    scaled_data = scaler.fit_transform(data[cluster_columns])

    data['Cluster_KMeans'] = kmeans_model.fit_predict(scaled_data)
    data['Cluster_KMeans'] = data['Cluster_KMeans'].astype('category')

    return data

def plot_elbow(data, cluster_columns, ax, title, color):
    """
    Plot Elbow plot for KMeans clustering.

    Parameters:
    - data: pd.DataFrame, input data
    - cluster_columns: list, columns used for clustering
    - ax: plt.Axes, axis for plotting
    - title: str, plot title
    - color: str, plot color
    """
    if data.empty:
        return

    inertias = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, random_state=42)
        kmeans.fit(data[cluster_columns])
        inertias.append(kmeans.inertia_)

    ax.plot(range(1, 11), inertias, marker='o', label=title, color=color)
    ax.set_title('Elbow Plot')
    ax.set_xlabel('Number of Clusters')
    ax.set_ylabel('Inertia')
    ax.grid(True)
    ax.legend()

def plot_scatter(data, cluster_columns, ax, title, color):
    """
    Plot scatter plot for KMeans clustering results.

    Parameters:
    - data: pd.DataFrame, input data
    - cluster_columns: list, columns used for clustering
    - ax: plt.Axes, axis for plotting
    - title: str, plot title
    - color: str, plot color
    """
    if data.empty:
        return

    scatter_plot = ax.scatter(data[cluster_columns[0]],
                              data[cluster_columns[1]], c=data[
                                  'Cluster_KMeans'], cmap='viridis')
    ax.set_title(title)
    ax.set_xlabel(cluster_columns[0])
    ax.set_ylabel(cluster_columns[1])

    # Add cluster centers to the plot
    cluster_centers = scaler.inverse_transform(kmeans_model.cluster_centers_)
    center_scatter = ax.scatter(cluster_centers[:, 0], cluster_centers[:, 1],
                                c='black', marker='X', s=70,
                                label='Cluster Centers')

    # Add cluster symbol to the legend
    handles, labels = scatter_plot.legend_elements()
    handles.append(center_scatter)
    labels.append('Cluster Centers')
    ax.grid(True)
    ax.legend(handles, labels, title='Clusters')

def fit_curve(x, y):
    """
    Fit a quadratic curve to the given data.

    Parameters:
    - x: np.array, x values
    - y: np.array, y values

    Returns:
    - fitted_curve: callable, function representing the fitted curve
    """
    valid_data = ~np.isnan(y)
    x = x[valid_data]
    y = y[valid_data]

    if len(x) == 0 or len(y) == 0:
        raise ValueError("No valid data for curve fitting.")

    popt, _ = curve_fit(lambda x, a, b, c: a*x**2 + b*x + c, x, y)
    fitted_curve = lambda x: popt[0]*x**2 + popt[1]*x + popt[2]

    return fitted_curve

def bootstrap_confidence_interval(fitted_curve, x, y, num_samples=100):
    """
    Compute bootstrap confidence interval for the fitted curve.

    Parameters:
    - fitted_curve: callable, function representing the fitted curve
    - x: np.array, x values
    - y: np.array, y values
    - num_samples: int, number of bootstrap samples

    Returns:
    - y_pred_lower: np.array, lower bound of the confidence interval
    - y_pred_upper: np.array, upper bound of the confidence interval
    """
    y_pred_samples = []

    for _ in range(num_samples):
        sample_indices = np.random.choice(len(y), len(y), replace=True)
        x_sample, y_sample = x.iloc[sample_indices], y.iloc[sample_indices]

        popt, _ = curve_fit(lambda x, a, b, c: a*x**2 + b*x + c, x_sample,
                            y_sample)
        fitted_curve_sample = lambda x: popt[0]*x**2 + popt[1]*x + popt[2]

        y_pred_samples.append(fitted_curve_sample(x))

    y_pred_samples = np.array(y_pred_samples)
    y_pred_mean = np.mean(y_pred_samples, axis=0)
    y_pred_std = np.std(y_pred_samples, axis=0)

    y_pred_upper = y_pred_mean + 1.96 * y_pred_std
    y_pred_lower = y_pred_mean - 1.96 * y_pred_std

    return y_pred_lower, y_pred_upper

def plot_urban_population_line(data, country, ax, color, year_to_predict=2025,
                               confidence_level=0.95):
    """
    Plot the line graph for 'Urban population' over the years for a specific country.

    Parameters:
    - data: pd.DataFrame, input data
    - country: str, name of the country
    - ax: plt.Axes, axis for plotting
    - color: str, plot color
    - year_to_predict: int, year for prediction

    Returns:
    None
    """
    x = data['Years']
    y = data[country]

    ax.plot(x, y, color=color, marker='.', label=f'{country} - Actual Data')

    fitted_curve = fit_curve(x, y)

    # Use the length of the original data for x_fit_full
    x_fit_full = np.linspace(min(x), max(x), 100)
    y_fit_full = fitted_curve(x_fit_full)
    
    # Different color for the fitted curve
    ax.plot(x_fit_full, y_fit_full, linestyle='--', color='darkgreen',
            label=f'{country} - Fitted Curve ({year_to_predict} prediction)')

    # Confidence interval
    y_pred_lower, y_pred_upper = bootstrap_confidence_interval(fitted_curve,
                                                               x, y)
    
    # Use the length of the confidence interval bounds for x_fit_confidence
    x_fit_confidence = np.linspace(min(x), max(x), len(y_pred_lower))
    
    # Lighter shadow for the confidence interval
    ax.fill_between(x_fit_confidence, y_pred_lower, y_pred_upper,
                    color='lightgreen', alpha=0.5,
                    label=f'{country} - {int(confidence_level * 100)}% Confidence Interval')

    predicted_value_2030 = fitted_curve(year_to_predict)
    ax.scatter([year_to_predict], [predicted_value_2030],
               color='red', marker='o', s=70, label=f'{country} - Prediction for {year_to_predict}')

    ax.set_xlabel('Year')
    ax.set_ylabel('Urban population')
    ax.legend()
    ax.grid(True)
    ax.set_xticks(np.arange(min(x), 2041, step=10))
    ax.set_xlim(min(x), 2040)

# Example usage
file_path = 'API_19_DS2_en_csv_v2_6300757.csv'
indicator1 = 'Urban population'
indicator2 = 'Arable land (% of land area)'
country1 = 'Aruba'
country2 = 'Belgium'
year_2000 = '2000'
year_2010 = '2010'

data_2000 = read_and_filter_data(file_path, indicator1, indicator2, year_2000)
data_2010 = read_and_filter_data(file_path, indicator1, indicator2, year_2010)

data_up1 = read_file(file_path, indicator1, country1)
data_up2 = read_file(file_path, indicator1, country2)

cluster_columns = [indicator1, indicator2]

data_2000 = kmeans_clustering(data_2000, cluster_columns)
data_2010 = kmeans_clustering(data_2010, cluster_columns)

# Plot Elbow plots for both years in one figure
fig, ax_elbow = plt.subplots(figsize=(8, 5))
plot_elbow(data_2000, cluster_columns, ax_elbow,
           title=f'KMeans Clustering - {year_2000}', color='blue')
plot_elbow(data_2010, cluster_columns, ax_elbow,
           title=f'KMeans Clustering - {year_2010}', color='orange')
plt.tight_layout()
plt.show()

# Plot KMeans clustering results for both years with cluster centers
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
plot_scatter(data_2000, cluster_columns, axes[0],
             title=f'KMeans Clustering Results - {year_2000}', color='blue')
plot_scatter(data_2010, cluster_columns, axes[1],
             title=f'KMeans Clustering Results - {year_2010}', color='orange')
plt.tight_layout()
plt.show()

# Plot line graph for 'Urban population' over the years for the first country
fig, ax_urban_population_1 = plt.subplots(figsize=(8, 6))
plot_urban_population_line(data_up1, country1,
                           ax_urban_population_1, color='blue')
ax_urban_population_1.set_title(f'Urban Population - {country1}')
plt.tight_layout()
plt.show()

# Plot line graph for 'Urban population' over the years for the second country
fig, ax_urban_population_2 = plt.subplots(figsize=(8, 6))
plot_urban_population_line(data_up2, country2,
                           ax_urban_population_2, color='orange')
ax_urban_population_2.set_title(f'Urban Population - {country2}')
plt.tight_layout()
plt.show()

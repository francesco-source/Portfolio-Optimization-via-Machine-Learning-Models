from sklearn.cluster import AgglomerativeClustering
from sklearn.model_selection import ParameterGrid,GridSearchCV
from sklearn.metrics import silhouette_score,accuracy_score,confusion_matrix
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np

def calculate_centroids(df, labels):
    """
    Calculate centroids for each cluster.

    Args:
        df (pd.DataFrame): DataFrame containing the data points.
        labels (array-like): Array of cluster labels assigned to each data point.

    Returns:
        np.ndarray: Array containing the centroids of each cluster.
    """
    # Get unique cluster labels
    unique_labels = np.unique(labels)

    centroids = []
    # Iterate over each unique cluster label
    for label in unique_labels:
        if label == -1:  # Skip noise points (if using DBSCAN or similar)
            continue
        # Extract data points belonging to the current cluster
        cluster_points = df[labels == label]
        # Calculate centroid of the cluster
        centroid = cluster_points.mean(axis=0)
        centroids.append(centroid)

    # Stack centroids into a single numpy array
    return np.vstack(centroids)



def max_rsi_lab_cluster(dictionary:dict):
    """
    Args:
        dictionary (dict): dictionary in the form label:centroid
                           where the first dimension of it it the rsi.

    Returns:
        the clustering label associated with the highest rsi.
    """
    if not dictionary:
        return None  # Return None if the dictionary is empty

    highest_key = max(dictionary, key=lambda k: dictionary[k][0])

    return highest_key



def kmeans_best_rsi(df: pd.DataFrame, n_clusters: int = 4):
    """
    Perform KMeans clustering on the given feature DataFrame, selecting the cluster
    with the highest performing RSI stocks.

    Args:
        df (pd.DataFrame): The feature DataFrame. The first dimension of it must be RSI.
        n_clusters (int, optional): The number of clusters for KMeans clustering. Defaults to 4.

    Returns:
        pd.DataFrame: The filtered DataFrame for each month with the highest performing RSI stocks.
    """
    
    X = MinMaxScaler().fit_transform(df)

    km = KMeans(n_clusters=n_clusters, random_state=0).fit(X)

    df['cluster'] = km.labels_

    # Calculate centroids based on selected dimensions and cluster labels
    rsi_index = df.columns.get_loc('rsi')

    atr_index = df.columns.get_loc('atr')


    selected_dimensions = [rsi_index, atr_index]
    
    # Centroids based on selected dimensions
    centroids = km.cluster_centers_[:, selected_dimensions]
    
    # Map cluster labels to their centroids
    label_centroid_map = {label: ct for label, ct in zip(range(n_clusters), centroids)}
    
    # Select the cluster label with the maximum RSI
    label = max_rsi_lab_cluster(label_centroid_map)
    
    df = df[df["cluster"] == label]

    # Remove the cluster label column
    df = df.drop("cluster", axis=1)

    return df



def agglomerative_best_rsi(df: pd.DataFrame, num_clusters: int = 3):
    """
    Perform Agglomerative Clustering with the best linkage method based on silhouette score.
    Select clusters based on RSI and return filtered DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing input features.
        num_clusters (int, optional): Number of clusters for Agglomerative Clustering. Defaults to 3.

    Returns:
        pd.DataFrame: Filtered DataFrame containing selected clusters based on RSI.

    """

    result_ac = []
    
    param_grid = {"linkage": ['ward', 'complete', 'average', 'single']}
    
    X = MinMaxScaler().fit_transform(df)
    
    param_grid = ParameterGrid(param_grid)
    
    # Iterate over parameter grid
    for par in param_grid:
        # Initialize Agglomerative Clustering with specified parameters
        agc = AgglomerativeClustering(n_clusters=num_clusters, linkage=par.get("linkage"))
        
        # Fit the clustering model and predict cluster labels
        y_agc = agc.fit_predict(df)
        
        # Calculate silhouette score and append to result list
        result_ac.append([par.get('linkage'), silhouette_score(df, y_agc)])
    

    df_result_ac = pd.DataFrame(data=result_ac, columns=['linkage', 'silhouette_score'])
    
    best_params_row = df_result_ac.sort_values(by='silhouette_score', ascending=False).head(1).iloc[0]
    linkage = best_params_row["linkage"]
    
    agc = AgglomerativeClustering(n_clusters=num_clusters, linkage=linkage).fit(X)
    
    # Assign cluster labels to DataFrame
    df["cluster"] = agc.labels_
    
    
    # Calculate centroids based on selected dimensions and cluster labels
    rsi_index = df.columns.get_loc('rsi')

    atr_index = df.columns.get_loc('atr')


    selected_dimensions = [rsi_index, atr_index]
    
    centroids = calculate_centroids(df, df["cluster"])[:, selected_dimensions]
    
    # Map cluster labels to their centroids
    label_centroid_map = {label: ct for label, ct in zip(range(int(num_clusters)), centroids)}
    
    # Select the cluster label with the maximum RSI
    label = max_rsi_lab_cluster(label_centroid_map)
    
    df = df[df["cluster"] == label]
    
    # Remove the cluster label column
    df = df.drop("cluster", axis=1)
    
    return df


def dbscan_best_rsi(df: pd.DataFrame):
    """
    Perform DBSCAN clustering on the given feature DataFrame, selecting the 
    cluster with the best performing RSI stocks.

    Args:
        df (pd.DataFrame): The feature DataFrame.

    Returns:
        pd.DataFrame: The filtered DataFrame containing the best performing RSI stocks.
    """
    # Define parameter grid for DBSCAN
    parameter_grid = ParameterGrid({"eps": np.arange(0.1, 1, 0.01), "min_samples": range(2, 10)})
    param_grid = list(parameter_grid)
    
    # Initialize DataFrame to store DBSCAN output
    dbscan_out = pd.DataFrame(columns=['eps', 'min_samples', 'n_clusters', 'silhouette', 'unclust%'])
    
    # Use MinMaxScaler to scale the selected columns
    columns_to_scale = df.columns
    scaler = MinMaxScaler()
    df_scaled = df.copy().dropna()
    df_scaled[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])
    
    # Iterate over parameter grid
    for i in param_grid:
        # Fit DBSCAN model with current parameters
        db = DBSCAN(eps=i.get("eps"), min_samples=i.get("min_samples"))
        y_db = db.fit_predict(df_scaled)
        
        # Compute cluster information
        cluster_labels_all = np.unique(y_db)
        cluster_labels = cluster_labels_all[cluster_labels_all != -1]
        n_clusters = len(cluster_labels)
        
        # Calculate silhouette score and percentage of unclustered points
        if n_clusters > 1:
            X_cl = df_scaled.values[y_db != -1, :]
            y_db_cl = y_db[y_db != -1]
            silhouette = silhouette_score(X_cl, y_db_cl)
            uncl_p = (1 - (len(y_db_cl) / len(y_db))) * 100
            dbscan_out.loc[len(dbscan_out)] = [db.eps, db.min_samples, n_clusters, silhouette, uncl_p]
    
    # Filter DBSCAN output based on specified thresholds
    n_clu_max_thr = 5
    n_clu_min_thr = 2
    data_frame_db = dbscan_out[(dbscan_out['n_clusters'] <= n_clu_max_thr) & 
                               (dbscan_out['n_clusters'] >= n_clu_min_thr)]
    
    # Select the best parameters based on silhouette score and percentage of unclustered points
    best_params_row = data_frame_db.sort_values(by=["silhouette", "unclust%"], ascending=[False, True]).head(1).iloc[0]
    best_eps = np.float64(best_params_row['eps'])
    best_min_samples = int(best_params_row['min_samples'])
    number_of_clusters = best_params_row["n_clusters"]
    
    # Fit DBSCAN with the best parameters
    best_dbscan = DBSCAN(eps=best_eps, min_samples=best_min_samples).fit(df_scaled)
    
    # Assign cluster labels to DataFrame
    df['cluster'] = best_dbscan.labels_
    
    # Calculate centroids and select the cluster with the maximum RSI
    selected_dimensions = [1, 5]
    centroids = calculate_centroids(df, df["cluster"])[:, selected_dimensions]
    label_centroid_map = {label: ct for label, ct in zip(range(int(number_of_clusters)), centroids)}
    label = max_rsi_lab_cluster(label_centroid_map)
    
    # Filter DataFrame to include only rows belonging to the selected cluster
    df = df[df["cluster"] == label]

    # Remove the cluster label column
    df = df.drop("cluster", axis=1)

    return df

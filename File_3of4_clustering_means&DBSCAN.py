#Final Project DS5110: Fall 2025
#Madi Augustine
#Liz Mutina
# Clustering, fuzzy C-means, Gaussian, DBSCAN

# CODE FILE: 3 OF 4

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import skfuzzy as fuzz
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.cluster import DBSCAN


def load_data(file):
    t1 = pd.read_csv(file)
    epsi_vars = [col for col in t1.columns if col.startswith("epsi")]
    print("Epsi Variables:", epsi_vars)
    epsi_data = t1[epsi_vars].values 
    print("Epsi Data", epsi_data)

    return epsi_vars, epsi_data

"""
Fuzzy clustering uses fuzzy logic, which allows data points instead of falling into only one cluster to be 
expressed over a continuum and fall into more than one cluster. 
referenced: https://www.geeksforgeeks.org/machine-learning/ml-fuzzy-clustering/
https://scikit-fuzzy.github.io/scikit-fuzzy/userguide/fuzzy_control_primer.html#example
https://pythonhosted.org/scikit-fuzzy/api/skfuzzy.cluster.html 
https://www.geeksforgeeks.org/data-analysis/principal-component-analysis-with-python/
"""
def fuzzy_clustering(epsi_data, epsi_vars):
    n_clusters = 3 # Number of clusters deteremined by elbow graph 
    m = 1.7 #fuzziness, higher the number fuzzier the clustering close to 1 would be similar to k means
    error = 1e-5 #only allows small changes(errors) in the clustering 
    maxiter = 1000 #maximum iterations for the algorithm to run
    scaler= StandardScaler() #So the sample can be on the same scale ((value - mean) / std_deviation)
    data = scaler.fit_transform(epsi_data) 

    """
    cntr: gets the mean of each cluster
    u: shows how much each data point belongs to each cluster
    u0: starting ranom membership before the iterations 
    d: Euclidean distance between data points and cluster centers
    jm: shows how the algorithm converges, loss/cost of each iteration 
    p: iteration count
    fcp: fuzzy partition coefficient, shows how well the clustering performed- higher better seperation 
    data.T: transposes data so there are x samples and 64 features (EPSi questions)
    """
    cntr, u, u0,d,jm,p, fpc = fuzz.cluster.cmeans(
    data.T, n_clusters, m, error, maxiter, init=None
)
    cluster_membership = np.argmax(u, axis=0) #Assign each data point to the cluster with highest membership value
    print("Cluster Centers:\n", cntr)
    print("\nFuzzy Membership Matrix (first 5 data points):")
    print(u[:, :5])
    #If is is bad all respondents answered too similiarly so no meanigful patterns can be found
    print(f"Fuzzy Partition Coefficient: {fpc}") #For 3 clusters, 0.333 is bad, 2clusters- 0.5 is bad, 4clusters- 0.25 is bad, 
    
    pca = PCA(n_components=3) #Need to reduce dimensions to 2D so it can be plotted 
    


    X = pca.fit_transform(data)  
    center = pca.transform(cntr)

    print("Explained variance ratio:", pca.explained_variance_ratio_)
    print("Components shape:", pca.components_.shape) 

    plt.figure(figsize=(8, 6))
    for i in range(n_clusters):
        plt.scatter(X[cluster_membership == i, 0], X[cluster_membership == i, 1], label=f'Cluster {i+1}')

    plt.scatter(center[0], center[1], marker='x', color='black', label='Centroids')

    #Negative plot values don't matter since it is PCA and the axes are arbitrary
    plt.title('Fuzzy C-Means Clustering on EPSi Data')
    plt.xlabel('Principal Component 1') #Reduced dimensions from 64 to 2
    plt.ylabel('Principal Component 2')
    plt.legend()
    plt.grid(True)
    plt.show()


    """
    Interpretting Results: higher scores in a variable means that cluster scores higher in that area
    0 = average
    Needed to look up how to print clusters 
    """
    cluster_1_vars = epsi_vars  # All variable names
    cluster_1_values = cntr[0]   # Their values in cluster 1
    print("Cluster 1 epsi:")
    for var, val in zip(cluster_1_vars, cluster_1_values):
        print(f"  {var}: {val:.3f}")
    cluster_2_vars = epsi_vars  # All variable names
    cluster_2_values = cntr[1]   # Their values in cluster 2
    print("Cluster 2 epsi:")
    for var, val in zip(cluster_2_vars, cluster_2_values):
        print(f"  {var}: {val:.3f}")
    cluster_3_vars = epsi_vars  # All variable names
    cluster_3_values = cntr[2]   # Their values in cluster 3
    print("Cluster 3 epsi:")
    for var, val in zip(cluster_3_vars, cluster_3_values):
        print(f"  {var}: {val:.3f}")
    #Uncomment if we add 4 clusters instead
    # cluster_4_vars = epsi_vars  # All variable names
    # cluster_4_values = cntr[3]   # Their values in cluster 4
    # print("Cluster 4 epsi:")
    # for var, val in zip(cluster_4_vars, cluster_4_values):
    #     print(f"  {var}: {val:.4}")

"""
Gaussian Mixture Model(GMM) clustering assumes the data in general is normally distributed, but it is taken form several different
normal distributions in the data, not just one as normal Gaussian Naive Bayes assumes. GMM allows the data to fall into 
numerous data points as well instead of just one. This model assumes there are multiple different types of clusters 
including elongated, tilted, and overlapping clusters instead of only having spherical clusters. 
Referenced: 
https://www.geeksforgeeks.org/machine-learning/gaussian-mixture-model/
https://jakevdp.github.io/PythonDataScienceHandbook/05.12-gaussian-mixtures.html
"""
def GMM_clustering(epsi_data, epsi_vars):
    n_components = 3 #number of clusters/ Guassian components 
    scaler= StandardScaler() #So the sample can be on the same scale  
    data = scaler.fit_transform(epsi_data)
    gmm = GaussianMixture(n_components, random_state=42) #Initialze GMM
    gmm.fit(data) #Fit the model to the data
    labels = gmm.predict(data) #Predict the cluster for each data point

    pca = PCA(n_components=2) #Need to reduce dimensions to 2D so it can be plotted 
    X = pca.fit_transform(data) 

    print("Explained variance ratio:", pca.explained_variance_ratio_)
    print("Components shape:", pca.components_.shape) 

    plt.figure(figsize=(8, 6)) 
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis') #Plot data points colored by their assigned cluster
    #Plot the centers of the Gaussian components
    plt.scatter(
        gmm.means_[:, 0],
        gmm.means_[:, 1],
        s=200,
        c='red',
        marker='X',
        label='Centers'
)
    plt.title('Gaussian Mixture Model Clustering on EPSi Data')
    plt.xlabel('Principal Component 1') #Reduced dimensions from 64 to 2
    plt.ylabel('Principal Component 2')
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.show()

    """
    Bayesian Information Criterion (BIC) helps to evaluate the model fit, if the model is overfitting the data
    it makes the score worse. We can change the number of clusters and see if it affects the score.
    Akaike Information Ctirterion (AIC) this is used specifically when you really are not sure if the model is correct or not. 
    It gives the maximum likelihood that the estimate of the model is actually accurate and not overfitting the data. 
    Referenced:
    https://www.geeksforgeeks.org/machine-learning/bayesian-information-criterion-bic/
    https://builtin.com/data-science/what-is-aic 
    """
    print(f"BIC score: {gmm.bic(data):.2f}")  # Lower is better (negative is ok)
    print(f"AIC score: {gmm.aic(data):.2f}") #Lower is better, but it is more comparing it to other models and then using the best one amoungst them 

    # Cluster 1
    cluster_1_vars = epsi_vars
    cluster_1_values = gmm.means_[0]
    print("Cluster 1 epsi:")
    for var, val in zip(cluster_1_vars, cluster_1_values):
        print(f"  {var}: {val:.3f}")

    # Cluster 2
    cluster_2_vars = epsi_vars
    cluster_2_values = gmm.means_[1]
    print("Cluster 2 epsi:")
    for var, val in zip(cluster_2_vars, cluster_2_values):
        print(f"  {var}: {val:.3f}")

    # Cluster 3
    cluster_3_vars = epsi_vars
    cluster_3_values = gmm.means_[2]
    print("Cluster 3 epsi:")
    for var, val in zip(cluster_3_vars, cluster_3_values):
        print(f"  {var}: {val:.3f}")
    
    # Cluster 4
    # cluster_4_vars = epsi_vars
    # cluster_4_values = gmm.means_[3]
    # print("Cluster 3 epsi:")
    # for var, val in zip(cluster_4_vars, cluster_4_values):
    #     print(f"  {var}: {val:.3f}")
"""
Density based clustering algorithm (DBSCAN) looks at points that cluster closely together regardless of the shape that is
ultimately formed. Any points that fall outside of the form that is shaped are considered noise, and not assigned to any
cluster at all. This works particularly well in data that does cluster, but does so in irregular patterns. 
Referenced: https://www.geeksforgeeks.org/machine-learning/dbscan-clustering-in-ml-density-based-clustering/
"""
def DBSCAN_clustering(epsi_data, epsi_vars):
    scaler= StandardScaler() #So the sample can be on the same scale  
    data = scaler.fit_transform(epsi_data)
    pca = PCA(n_components=2) #Need to reduce dimensions to 2D so it can be plotted 
    X = pca.fit_transform(data) 
    #eps: max distance between two points, that can still be neighbors (get from elbow graph)
    #min_samples: min number of points to cluster (2 * number of features(64 EPSi questions) = 128)
    #Could try different eps and min_samples values to see how it affects the clustering
    db = DBSCAN(eps = 3, min_samples=56).fit(X)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool) #Boolean mask to create the core samples
    core_samples_mask[db.core_sample_indices_] = True #Mark core samples as True, this becomes center of clustering
    labels = db.labels_ #Getting assignment points for the clusters 

    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0) #removing noise that isn't in a cluster
    n_noise = list(labels).count(-1) #Counting number of noise points

    print(f"Number of clusters: {n_clusters_}")
    print(f"Number of noise points: {n_noise}")
    unique_labels = set(labels)
    colors = ['y', 'b', 'g', 'r'] #color assignmnet for actual clusters, not noise 
    for k, col in zip(unique_labels, colors):
        if k == -1:
            col = 'k' #if it is noise then it is black

        #Plotting the actual clusters and core points 
        class_member_mask = (labels == k)
        xy = X[class_member_mask & core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
                markeredgecolor='k',
                markersize=6)

        #Plotting the border points, they still cluster, but they are on the outer edges of the cluster 
        #(can't tell difference between cluster and border in the graph)
        xy = X[class_member_mask & ~core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
                markeredgecolor='k',
                markersize=6)

    plt.title('number of clusters: %d' % n_clusters_)
    plt.show()

def main():
    file = "Data/np_all_data_csv_epsi.csv"  # Replace with your actual file path
    epsi_vars, epsi_data = load_data(file)
    fuzzy_clustering(epsi_data, epsi_vars)
    GMM_clustering(epsi_data, epsi_vars)
    DBSCAN_clustering(epsi_data,epsi_vars)

if __name__ == "__main__":
    main()

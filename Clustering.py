from sklearn.cluster import KMeans
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

dataset = pd.read_csv('Workbook5CleanClustering.csv')
#dataset_without_drafted = dataset
dataset_without_drafted = dataset[dataset.Drafted == 1]

KM = KMeans(n_clusters=3, init='k-means++', random_state=170)

KM = KM.fit(dataset_without_drafted)

print("The cluster centroids are: \n", KM.cluster_centers_)
print("Cluster", KM.labels_)
print("Sum of distances of samples to their closest cluster center: ", KM.inertia_)

#colors = ['black','red']
#plt.scatter(dataset_with_drafted.ASTp, dataset_with_drafted.BLKp, c=KM.labels_, cmap=matplotlib.colors.ListedColormap(colors), s=75)
#plt.show()

fig = plt.figure(1, figsize=(10, 8))
ax = Axes3D(fig, elev=-150, azim=110)

colormap = np.array(['indigo', 'teal', 'gold'])
ax.scatter(dataset_without_drafted.TRBp, dataset_without_drafted.FG, dataset_without_drafted.ASTp,  c=colormap[KM.labels_], s=100)
plt.show()

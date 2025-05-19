import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

X=np.array([[30,50000],[35,60000],[40,80000],[25,30000],[45,100000],[20,20000],
            [50,120000],[55,150000],[60,140000],[28,40000]])

Kmeans=KMeans(n_clusters=3)
Kmeans.fit(X)

labels=Kmeans.labels_
centers=Kmeans.cluster_centers_

plt.scatter(X[:,0],X[:,1],c=labels)
plt.scatter(centers[:,0],centers[:,1],c='red',marker='x',label='Centroids')
plt.xlabel('Age')
plt.ylabel('Income')
plt.title('K-Means Clustering of Customers')
plt.legend()
plt.show()  

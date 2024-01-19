#data_preprocessing

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importer le dataset
dataset = pd.read_csv('Mall_Customers.csv')


X = dataset.iloc[:, 3:5].values


# determination du  nombre de cluster approprié par la methode elbow
from sklearn.cluster import KMeans
wcss=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i,init="k-means++",random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss,linewidth=5)
plt.title("la méthode Elbow")    
plt.xlabel("nombre de cluster")
plt.ylabel("wcss")
plt.figure(dpi=1000)
plt.show()

#construction du model
from sklearn.cluster import KMeans
kmeans=KMeans(n_clusters=5,init="k-means++",random_state=0)
y_kmeans= kmeans.fit_predict(X)

#visualisation des resultats
plt.scatter(X[y_kmeans==1,0], X[y_kmeans==1,1],color="red",label="cluster 1")
plt.scatter(X[y_kmeans==2,0], X[y_kmeans==2,1],color="blue",label="cluster 2")
plt.scatter(X[y_kmeans==3,0], X[y_kmeans==3,1],color="green",label="cluster 3")
plt.scatter(X[y_kmeans==4,0], X[y_kmeans==4,1],color="cyan",label="cluster 4")
plt.scatter(X[y_kmeans==0,0], X[y_kmeans==0,1],color="magenta",label="cluster 5")
#plt.scatter(X[y_kmeans==5,0], X[y_kmeans==5,1],color="black",label="cluster 6")
plt.title("cluster des clients")
plt.xlabel("Salaire annuel")
plt.ylabel("pending score")
plt.legend()


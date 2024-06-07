
import clustering
import parameters as params
import numpy as np

params.remains = np.random.randint(0, 121, params.numEdge)


clst = clustering.Clustering()

clst.form_cluster()
clst.visualize_clusters()

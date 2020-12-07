import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt

# for runtime and memory usage
import time
from guppy import hpy

import sklearn.cluster
from sklearn import metrics


def form_clusters(inputs, clustering, labels_true=None):

  score_types = { 
                  "num_clusters" : [],
                  "silhouette" : [], "cal_har" : [], "dav_bould" : [],
                  "adj_rand" : [],  "adj_mut_inf" : [], "v_measure" : [], "fowlk_mall" : []
                }
  K = range(2,21)
  start_time = time.time()
  h = hpy()

  for k in tqdm(K):
    cluster_line = "sklearn.cluster.%s(n_clusters=k).fit(inputs)" % (clustering)
    clusters = eval (cluster_line)
    labels_pred = clusters.labels_

    score_types["num_clusters"].append(k)
    score_types["silhouette"].append(metrics.silhouette_score(inputs, labels_pred, metric='euclidean'))
    score_types["cal_har"].append(metrics.calinski_harabasz_score(inputs, labels_pred))
    score_types["dav_bould"].append(metrics.davies_bouldin_score(inputs, labels_pred))

    if labels_true is not None:
      score_types["adj_rand"].append(metrics.adjusted_rand_score(labels_true, labels_pred))
      score_types["adj_mut_inf"].append(metrics.adjusted_mutual_info_score(labels_true, labels_pred))
      score_types["v_measure"].append(metrics.v_measure_score(labels_true, labels_pred))
      score_types["fowlk_mall"].append(metrics.fowlkes_mallows_score(labels_true, labels_pred))


  print("runtime: ",time.time()-start_time, end="\n\n")
  print("memory consumption:")
  print(h.heap(), end="\n\n")
  plt.plot(K, score_types["silhouette"], 'bx-')
  plt.show() 

  score_types = {k:v for k,v in score_types.items() if v}
  df = pd.DataFrame(score_types)
  df = df.to_string(index=False)
  print (df)
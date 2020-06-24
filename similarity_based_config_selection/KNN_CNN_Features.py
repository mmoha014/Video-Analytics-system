from sklearn.neighbors import NearestNeighbors
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from scipy.spatial import distance_matrix
from sklearn.metrics.pairwise import cosine_distances
from scipy import sparse

data = list()
files = list()
for f in ['1']:#,'2','3','5','6']:#,'VA1','VA2','VA3']:
    fp = open('/home/morteza/Videos/traffic camera/keyframes/newFeatures/'+f+'.p','rb')#VGG19-FEATURES/'+f+'.p','rb')
    tmp_data = pickle.load(fp)
    for i in range(len(tmp_data)):
        data.append(tmp_data[i])
        files.append(f+'_'+str(i))

fp.close()

# num_data = len(data)
# X,y = range(num_data), range(num_data)

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# a=0
# train_data_f = list()
# train_data_kfr = list()
# for i in X_train:
#     train_data_kfr.append(data[i][1])
#     train_data_f.append(data[i][0])
# b=0
"""
var = pd.DataFrame(data).T

clmns= list()
for i in range(len(data)):
    clmns.append('feature_'+str(i))
var.columns = clmns
dist = pd.DataFrame(1 / (1 + distance_matrix(var.T, var.T)),columns=var.columns, index=var.columns)
# print("Fitting k-nearest-neighbour model on training images...")
# knn = NearestNeighbors(n_neighbors=5, metric="cosine")
# knn.fit(E_train_flatten)

fp = open('CNN_similarity_matrix_trafficOnly.p','wb')
pickle.dump(dist,fp)
fp.close()
"""
fp = open('CNN_similarityTest.p','rb')#CNN_similarity_matrix_trafficOnly.p','rb')
matrix = pickle.load(fp)
sorted_rows = list()
fp = open('keyframe_categorization.txt','w')
fp.write('categorizing keyframes based on their similarity\n')
for i in range(len(matrix)):
    row = np.array(matrix['feature_'+str(i)])
    sorted_rows=row.argsort()[::-1][:12]
    fp.write("feature"+str(i)+": "+str(sorted_rows[1:13])+"\n")
a=0
fp.close()
# """
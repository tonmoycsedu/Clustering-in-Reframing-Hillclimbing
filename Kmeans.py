
# coding: utf-8

# In[3]:

from sklearn.preprocessing import MinMaxScaler
import numpy as np

data = np.array([[50],[70],[110],[250],[54],[136],[214],[78],[184],[34],[60],[266],[56],[166],[122]])
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data[:5])
ct = [[ 0.17470606],[ 0.81430069],[ 1.56099155]]
clusters = kmeans(3,data_scaled,ct,1,1)
print(clusters)


# In[1]:

def kmeans(k,dataItems,centroids,maxIter,num_attr):
    old_centroids = centroids
    #print(k,dataItems,centroids,maxInter,num_attr)
    
    groups = []
    for i in range(k):
        groups.append([])
        
    iter = 0
    while(iter < maxIter):
        for item in dataItems:
            row = []
            for centroid in centroids:
                diff = 0
                for i in range(num_attr):
                    diff += abs(item[i] - centroid[i])
                    
                row.append(diff)
                
            idx = row.index(min(row))
            groups[idx].append(item)
                    
            
        iter += 1
        
    return groups
        


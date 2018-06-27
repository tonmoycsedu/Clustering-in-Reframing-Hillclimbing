
# coding: utf-8

# In[1]:

#function to write in csv
import csv
def write_list_in_file(final, name):
    with open(name, "w", newline="",encoding="utf8") as fp:
        a = csv.writer(fp, delimiter=',')
        a.writerows(final)


# In[2]:

#Function to read csv files
from csv import reader
# Load a CSV file\n",
def load_csv(filename):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset


# In[3]:

import numpy as np
from sklearn import preprocessing

#load base data
base_data = load_csv('season1.csv')    
base_data = np.array(base_data[1:])
base_size = len(base_data)

#no of target elements and load target data
elements = [10,20,30,40,50]
target_data = load_csv('season4.csv')
target_data = np.array(target_data[1:])

# #normalize data using a min max normalizer
# mixed = np.concatenate((base_data,target_data))
# scaler = MinMaxScaler()
# mixed_scaled = scaler.fit_transform(mixed)

#base_data = scaler.fit_transform(base_data)

#normalized base and target data
scaler = preprocessing.StandardScaler().fit(base_data)
base_scaled = scaler.transform(base_data)
target_scaled = scaler.transform(target_data)


# In[4]:

# number of attributes
num_attr = base_scaled[0].size


# In[5]:

#learn number of clusters using silhouette_score

from __future__ import print_function

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

num_clusters = 0
max_silhouette = -100

for n_clusters in range(2,10):

    # Initialize the clusterer with n_clusters value and a random generator
    # seed of 10 for reproducibility.
    clusterer = KMeans(n_clusters=n_clusters)
    cluster_labels = clusterer.fit_predict(base_scaled)

    silhouette_avg = silhouette_score(base_scaled, cluster_labels)
    #print("For n_clusters =", n_clusters,"The average silhouette_score is :", silhouette_avg)
    
    if(silhouette_avg > max_silhouette):
        max_silhouette = silhouette_avg
        num_clusters = n_clusters

print("Optimum Number of Clusters: ",num_clusters)


# In[6]:

#learn centers from source/base model
import numpy as np
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=num_clusters)
kmeans.fit(base_scaled)

base_centroids = kmeans.cluster_centers_
#labels = kmeans.labels_
    
#print(kmeans.cluster_centers_)
#print(kmeans.inertia_)


# In[7]:

#base model results
kmeans = KMeans(n_clusters=num_clusters, init=base_centroids, max_iter=1)
kmeans.fit(target_scaled)
print("applying base model in target result: ", kmeans.inertia_)


# In[8]:

#retraining results


for n in elements:

    #Read Available Limited Target Data
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(target_scaled[:n])
    
    #Result on whole Target Data
    kmeans = KMeans(n_clusters=num_clusters, init=kmeans.cluster_centers_, max_iter=1)
    kmeans.fit(target_scaled)

    print("Retraining results on ", n," data ",kmeans.inertia_)


# In[9]:

def cal_centroids(alpha,beta):

    target_centroids = []
    for i in range(num_clusters):
        val = []
        for j in range(num_attr):
            val.append(alpha[j]*base_centroids[i][j] + beta[j])
        
        target_centroids.append(val)
             

    target_centroids = np.array(target_centroids)
    #print("new centroids: ", target_centroids)
    return target_centroids
    #print(target_centroids)


# In[10]:

def kmeans_custom(k,dataItems,centroids,maxIter,num_attr):
    
    #print(k,dataItems,centroids,maxInter,num_attr)
    old_centroids = []
    groups = []
    
    for i in range(k):
        groups.append([])
        
    iter = 0
    while(iter < maxIter):
        old_centroids = centroids
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
        
    ss = 0    
    for i in range(k):
        for item in groups[i]:
            for j in range(num_attr):
            
                diff = abs(item[j]- old_centroids[i][j])
                ss += pow(diff,2)
            
        
    return (groups,ss/2)
        


# In[11]:

def cal_gradient(clusters,reframed_centroids,old_alpha,old_beta):
    
    gradient_alpha = []
    gradient_beta = []
    gradient = 0
    
    for i in range(num_attr):
        gradient_alpha.append(0)
        gradient_beta.append(0)
        
    
    for i in range(num_clusters):
                
        for member in clusters[i]:
                      
            for j in range(num_attr):
                
                gradient =  (reframed_centroids[i][j] - member[j])
                gradient_alpha[j] = gradient_alpha[j] + gradient*reframed_centroids[i][j]
                gradient_beta[j] = gradient_beta[j] + gradient

    #print("******")
    #print(gradient_alpha,gradient_beta)
    
    new_alpha = []
    new_beta = []
    
    for i in range(num_attr):       
        new_alpha.append(old_alpha[i]-.0001*gradient_alpha[i])
        new_beta.append(old_beta[i]-.0001*gradient_beta[i])
        
    return [new_alpha,new_beta]


# In[12]:

def learn_parameters(alpha,beta):
    reframed_centroids =  cal_centroids(alpha,beta)
    km = kmeans_custom(num_clusters, target_scaled, reframed_centroids, 1, num_attr)
    #kmeans.fit(target_scaled)
    best_error = km[1]
    #centroids = kmeans.cluster_centers_
    #labels = kmeans.labels_

    count = 0
    best_alpha = alpha
    best_beta = beta
    while(1):
        #print(centroids)
        #print("best error: ",best_error)

        #clusters = find_members(centroids,labels)

        #reframed_centroids = closest_centroids(reframed_centroids,centroids)

        new_alphabeta = cal_gradient(km[0],reframed_centroids,alpha,beta)

        alpha = new_alphabeta[0]
        beta = new_alphabeta[1]

        #print("new alpha beta", alpha, beta)

        reframed_centroids =  cal_centroids(alpha,beta)

        km = kmeans_custom(num_clusters, target_scaled, reframed_centroids, 1, num_attr)
        #kmeans.fit(target_scaled)
        new_error = km[1]

        #print("compare ",best_error,new_error)
        if(new_error < best_error):
            best_alpha = alpha
            best_beta = beta
            best_error = new_error
            count = 0

        elif(new_error == best_error):
            if(count<5):
                count += 1
                continue
            else:
                break;

        else:
            break; 

        #base_centroids = kmeans.cluster_centers_
        #labels = kmeans.labels_

        #print(kmeans.cluster_centers_)
        #print(old_error,new_error)

    #print("finalparameters", best_alpha,best_beta)
    
    target_cent = cal_centroids(best_alpha,best_beta)
    kmeans = KMeans(n_clusters=num_clusters, init=target_cent, max_iter=1)
    kmeans.fit(target_scaled)
    #print("Reframing results on ",kmeans.inertia_)
    return kmeans.inertia_ 


# In[13]:


for n in elements:
    
    alpha = []
    beta = []
    avg_m = []
    avg_d = []
    sum_d = 0
    for i in range(num_attr):
        sum_m = 0
        for j in range(num_clusters):
            sum_m = sum_m + base_centroids[j][i]

        avg_m.append(sum_m/num_clusters)


    for i in range(num_attr):
        sum_d = 0
        for j in range(len(target_scaled[:n])):
            sum_d = sum_d + target_scaled[j][i]

        avg_d.append(sum_d/len(target_scaled[:n]))


    for i in range(num_attr):
        alpha.append(avg_m[i]/avg_d[i])
        beta.append(0)

    #print(alpha,beta)
    #print("#####")
    result = learn_parameters(alpha,beta)
    print("Reframing results on ",n," data: ",result)



# In[ ]:




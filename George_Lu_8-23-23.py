#!/usr/bin/env python
# coding: utf-8

# In[4]:


'''
rewrite read data from csv according to different data source w/ flexibility in variable

find which combination of factors works best:
HR + BP
tidal volume + RR + positive end expiratory pressure + airway pressure + Spo2 + compliance
population: general, subset
'''

import pandas as pd
import numpy as np
from statistics import median
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from sklearn import svm, metrics

vars_list = ['Heart Rate', 'Blood Pressure', 'Respiration Rate', 'Temperature']


var1 = 'Heart Rate'
var2 = 'Blood Pressure'


#read data from csv
admissions_data = pd.read_csv('/N/project/waveform_mortality/shared/datasets/clinical_datasets/mimic-iii-clinical-database-1.4/ADMISSIONS.csv', usecols = ['HADM_ID', 'DISCHTIME', 'HOSPITAL_EXPIRE_FLAG'])
admissions_data['DISCHTIME'] = pd.to_datetime(admissions_data['DISCHTIME'])
var1_data = pd.read_csv('/N/project/waveform_mortality/georgelu/HeartRate.csv', 
                      usecols = ['HADM_ID', 'CHARTTIME', 'VALUENUM'])
var1_data['CHARTTIME'] = pd.to_datetime(var1_data['CHARTTIME'])
var1_data.rename(columns = {'VALUENUM' : var1}, inplace = True)

var2_data = pd.read_csv('/N/project/waveform_mortality/georgelu/NIBPs.csv', 
                         usecols = ['HADM_ID', 'CHARTTIME', 'VALUENUM'])
var2_data['CHARTTIME'] = pd.to_datetime(var2_data['CHARTTIME'])
var2_data.rename(columns = {'VALUENUM' : var2}, inplace = True)

#combine var1 frames and var2 frames into one dataframe each
var1_admissions_combined = pd.merge(var1_data, admissions_data, on = 'HADM_ID')
var2_admissions_combined = pd.merge(var2_data, admissions_data, on = 'HADM_ID')
var1_admissions_combined['TIME_DIFFERENCE'] = (var1_admissions_combined['DISCHTIME'] - var1_admissions_combined['CHARTTIME'])
var2_admissions_combined['TIME_DIFFERENCE'] = (var2_admissions_combined['DISCHTIME'] - var2_admissions_combined['CHARTTIME'])

# all time variables use hours as a unit
# time defines the end of the time window closest to death or discharge to pull measurements
# time_window_width defines the width of the time window to pull measurements
# max_time defines the time window farthest from death or discharge
# time_step defines the separation between consecutive windows
# ex: if time = 1 and time_window_width = 6, a window of 1 to 7 hours before death or discharge will be used
# ex: if max_time = 6 and time_window_width = 6, a window of 6 to 12 hours before death or discharge will be used
# ex: if time_step = 1, then time will increment by 1 between windows
time = 1
time_window_width = 6
max_time = 12
time_step = 1
AUCs = []
timepoints = []
while time <= max_time:
    timepoints.append(time)
    #def measurement time frame
    t1 = pd.to_timedelta(time, unit = 'h')
    t2 = t1 + pd.to_timedelta(time_window_width, unit = 'h')
    #t2 = pd.to_timedelta(7, unit = 'h')

    #remove measurements outside time frame
    var1_admissions_combined.dropna(inplace = True)
    var2_admissions_combined.dropna(inplace = True)
    var1_clean = var1_admissions_combined.drop(var1_admissions_combined[(var1_admissions_combined.TIME_DIFFERENCE < t1) | (var1_admissions_combined.TIME_DIFFERENCE > t2)].index)
    var2_clean = var2_admissions_combined.drop(var2_admissions_combined[(var2_admissions_combined.TIME_DIFFERENCE < t1) | (var2_admissions_combined.TIME_DIFFERENCE > t2)].index)
    var1_clean.drop(columns = ['CHARTTIME', 'DISCHTIME', 'HOSPITAL_EXPIRE_FLAG', 'TIME_DIFFERENCE'], inplace = True)
    var2_clean.drop(columns = ['CHARTTIME', 'DISCHTIME', 'TIME_DIFFERENCE'], inplace = True)

    data_clean = pd.merge(var1_clean, var2_clean, on = 'HADM_ID')

    #find median HR and NIBPs
    medians_data = pd.DataFrame(columns = ['HADM_ID', var1, var2, 'HOSPITAL_EXPIRE_FLAG'], 
                                dtype = np.float64)
    data_for_given_ID = pd.DataFrame(columns = ['HADM_ID', var1, var2, 'HOSPITAL_EXPIRE_FLAG'],
                                     dtype = np.float64)

    for ID in data_clean['HADM_ID'].drop_duplicates():
        var1_median = median(data_clean.loc[data_clean['HADM_ID'] == ID][var1])
        var2_median = median(data_clean.loc[data_clean['HADM_ID'] == ID][var2])
        #this line below is dumb come up with some other way to pull HOSPITAL_EXPIRE_FLAG for a given id
        is_expired = min(data_clean.loc[data_clean['HADM_ID'] == ID]['HOSPITAL_EXPIRE_FLAG'])
        data_for_given_ID.loc[0] = [ID, var1_median, var2_median, is_expired]
        medians_data = pd.concat([medians_data, data_for_given_ID], ignore_index = True)

    #calculate mortality using k nearest neighbors
    k = 100
    
    dead_points = medians_data[medians_data['HOSPITAL_EXPIRE_FLAG'] == 1.0]  
    dead_points_indices = list(dead_points.index)
    mortality_rate_list = pd.Series([], dtype = np.float64)

    neigh = NearestNeighbors(n_neighbors = k)
    neigh.fit(medians_data[[var1, var2]])
    (distances, indices) = neigh.kneighbors(medians_data[[var1, var2]])

    for index in indices:
        mortality_rate = pd.Series([len(set(dead_points_indices).intersection(index)) / k])
        mortality_rate_list = pd.concat([mortality_rate_list, mortality_rate])
    
    #SVM training
    X_train, X_test, y_train, y_test = train_test_split(medians_data[[var1, var2]], medians_data['HOSPITAL_EXPIRE_FLAG'], 
                                                        test_size = 0.3, random_state = 0)
    
    svc = svm.SVC(probability = True)
    svc.fit(X_train, y_train)
    
    y_pred_proba = svc.predict_proba(X_test)[::, 1]
    fpr, tpr, unused = metrics.roc_curve(y_test, y_pred_proba)
    auc = metrics.roc_auc_score(y_test, y_pred_proba)
    AUCs.append(auc)

    #plots
    
#     #ROC plot
#     plt.plot(fpr, tpr, label = "AUC = " + str(auc))
#     plt.ylabel('True Positive Rate')
#     plt.xlabel('False Positive Rate')
#     plt.legend(loc = 4)
#     plt.title('ROC and AUC for ' + var1 + ' and ' + var2)
#     plt.show()
    
#     #contour plot
#     contour_levels = 10
#     fig, scatter = plt.subplots()
    
#     scatter.tricontour(medians_data[var1], medians_data[var2], mortality_rate_list, 
#                        levels = contour_levels, colors = 'k')
#     cntr = scatter.tricontourf(medians_data[var1], medians_data[var2], mortality_rate_list, 
#                                levels = contour_levels, cmap = 'RdBu_r')

#     fig.colorbar(cntr, ax = scatter, label = 'Mortality Rate')
    
#     scatter.plot(medians_data[var1], medians_data[var2], 'ko', ms = 3, alpha = 0.0)
    
#     plt.ylabel('Median ' + var2)
#     plt.xlabel('Median ' + var1)
#     plt.title('Contours for ' + var1 + ' and ' + var2)
#     plt.show()
    
    time += time_step
    
y = AUCs
x = timepoints

fig, ax = plt.subplots()
ax.plot(x, y)
plt.title('AUC for ' + var1 + ' and ' + var2 + ' at given time points')
plt.show()


# In[ ]:


from sklearn.neighbors import NearestNeighbors
import pandas as pd
import matplotlib.pyplot as plt

points = pd.DataFrame([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]])
dead_points = points[points[2] == 0]
dead_points_indices = list(dead_points.index)
mortality_rate_list = pd.Series([], dtype = 'Float64')
print(dead_points_indices)

neigh = NearestNeighbors(n_neighbors = 3)
neigh.fit(points[[0, 1]])
(distances, indices) = neigh.kneighbors(points[[0, 1]])

for index in indices:
    mortality_rate = pd.Series([len(list(set(dead_points_indices).intersection(index)))/3])
    mortality_rate_list = pd.concat([mortality_rate_list, mortality_rate])
    
print(mortality_rate_list.reset_index(drop = True))

fig, scatter = plt.subplots()
scatter.tricontour(points[0], points[1], mortality_rate_list, levels = 5, colors = 'k')
cntr = scatter.tricontourf(points[0], points[1], mortality_rate_list, levels = 5, cmap = 'RdBu_r')

fig.colorbar(cntr, ax = scatter)
scatter.plot(points[0], points[1], 'ko', ms = 3, alpha = 1.0)
plt.show()
    


# In[3]:


import matplotlib.pyplot as plt

y = [0.7818383661300566, 0.7701423917612442, 0.7958499692474486, 0.7660046581896008, 0.7758455892060308, 0.7494075073862307, 0.7677919130067568, 0.7249180363580912, 0.7619646789196961, 0.7226251168796474, 0.7200972921636186, 0.7062371285672255]
x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

fig, ax = plt.subplots()
ax.plot(x, y)
plt.show()


# 

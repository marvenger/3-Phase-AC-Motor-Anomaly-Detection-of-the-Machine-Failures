#!/usr/bin/env python
# coding: utf-8

# In[32]:


# Importing the required libraries

import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
from scipy.signal import find_peaks
import matplotlib.pyplot as plt


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report

from sklearn.model_selection import cross_val_score


# In[2]:





# #### I will use the tsmoothie library package

# In[ ]:





# In[3]:


# Using pip to install the tmsoothie library
get_ipython().system(' pip install --upgrade tsmoothie')


# In[4]:


# importing the required dependancies

from tsmoothie.utils_func import sim_seasonal_data
from tsmoothie.smoother import *


# In[5]:


# setting a random seed
np.random.seed(33)


# In[ ]:





# #### The data is structured. There are a total of 317 text files in the given folder which are named in the sequence of 'data_2.txt', 'data_3.txt', ....... and 'data_318.txt'.
# 
# #### Each file consists of the 5 columns of which the first and the last columns are filled with the null values corresponding to all the rows. So, they are removed while processing.
# 
# #### The other 3 columns are the data based on the 3 phases of the motor.
# 
# #### Each of the file is of length 1000 that depicts 1000 current sensors of a 3-phase induction motor obtained per second

# In[ ]:


# a dictionary is created to store the data for each file
# all the items in the dictionary are stored together in a master dataframe
df={}
data_master=pd.DataFrame()
for i in range(2,319):
    x='data/data'+str(i)+'.txt'
    df[i]=pd.read_csv(x,sep=',',header=None)
    df[i].drop(columns=[0,4],inplace=True)
    data_master=data_master.append(df[i])


# In[ ]:





# In[ ]:





# In[ ]:





# #### I will use different types of Smoothing available under the tsmoothie library.
# #### I will prefer using the sinusoidal smoothing techniques as the data of the sensors in the 3 phase induction motor are generally means to be of sinusoidal waveforms.

# #### Initially I will use the single seasonality smoothing parameters for all the different types of smoothing.

# In[ ]:





# In[ ]:





# ### Exponential Smoothing

# #### Exponential smoothing is a time series forecasting method that uses a weighted average of past observations to make predictions about future values. The method assigns exponentially decreasing weights to past observations, with more recent observations being given greater weight than older observations.

# In[6]:


# operate smoothing
smoother = ExponentialSmoother(window_len=100, alpha=0.3)
smoother.smooth(df[2][1])
# generate intervals
low, up = smoother.get_intervals('sigma_interval')

# plot the first smoothed timeseries with intervals based on the first phase
plt.figure(figsize=(100,20))
plt.plot(smoother.smooth_data[0], linewidth=3, color='blue')
plt.plot(smoother.data[0], '.k')
plt.xlabel('timeline')
plt.fill_between(range(len(smoother.data[0])), low[0], up[0], alpha=0.3)


smoother = ExponentialSmoother(window_len=100, alpha=0.3)
smoother.smooth(df[2][2])
low, up = smoother.get_intervals('sigma_interval')

# plot the first smoothed timeseries with intervals based on the second phase
plt.figure(figsize=(100,20))
plt.plot(smoother.smooth_data[0], linewidth=3, color='orange')
plt.plot(smoother.data[0], '.k')
plt.xlabel('timeline')
plt.fill_between(range(len(smoother.data[0])), low[0], up[0], alpha=0.3,color='orange')


smoother = ExponentialSmoother(window_len=20, alpha=0.3)
smoother.smooth(df[2][3])
low, up = smoother.get_intervals('sigma_interval')

# plot the first smoothed timeseries with intervals based on the third phase
plt.figure(figsize=(100,20))
plt.plot(smoother.smooth_data[0], linewidth=3, color='green')
plt.plot(smoother.data[0], '.k')
plt.xlabel('timeline')
plt.fill_between(range(len(smoother.data[0])), low[0], up[0], alpha=0.3,color='green')


# In[ ]:





# In[ ]:





# ### Convolution Smoothing

# #### Convolution smoothing, also known as moving average smoothing, is a technique used in signal processing and time series analysis to smooth out noisy data. The method involves replacing each data point with the average of the nearby points in the series.
# 
# #### The size of the window used for the moving average determines the level of smoothing: a larger window size will result in more smoothing, while a smaller window size will result in less smoothing. The choice of window size depends on the characteristics of the data being analyzed and the desired level of smoothing.

# In[ ]:





# In[7]:


# operate smoothing
smoother = ConvolutionSmoother(window_len=10, window_type='ones')
smoother.smooth(df[2][1])
# generate intervals
low, up = smoother.get_intervals('sigma_interval')

# plot the first smoothed timeseries with intervals based on the first phase
plt.figure(figsize=(100,20))
plt.plot(smoother.smooth_data[0], linewidth=3, color='blue')
plt.plot(smoother.data[0], '.k')
plt.xlabel('timeline')
plt.fill_between(range(len(smoother.data[0])), low[0], up[0], alpha=0.3)


smoother = ConvolutionSmoother(window_len=10, window_type='ones')
smoother.smooth(df[2][2])
low, up = smoother.get_intervals('sigma_interval')

# plot the first smoothed timeseries with intervals based on the second phase
plt.figure(figsize=(100,20))
plt.plot(smoother.smooth_data[0], linewidth=3, color='orange')
plt.plot(smoother.data[0], '.k')
plt.xlabel('timeline')
plt.fill_between(range(len(smoother.data[0])), low[0], up[0], alpha=0.3,color='orange')


smoother = ConvolutionSmoother(window_len=10, window_type='ones')
smoother.smooth(df[2][3])
low, up = smoother.get_intervals('sigma_interval')


# plot the first smoothed timeseries with intervals based on the third phase
plt.figure(figsize=(100,20))
plt.plot(smoother.smooth_data[0], linewidth=3, color='green')
plt.plot(smoother.data[0], '.k')
plt.xlabel('timeline')
plt.fill_between(range(len(smoother.data[0])), low[0], up[0], alpha=0.3,color='green')


# In[ ]:





# In[ ]:





# ### Spectral Smoothing

# #### Spectral smoothing is a signal processing technique used to smooth out a signal or remove high-frequency noise from a signal in the frequency domain. The method involves applying a spectral filter to the signal, which can be done using various filter types such as low-pass, high-pass, band-pass, or notch filters.

# In[8]:


# operate smoothing
smoother = SpectralSmoother(smooth_fraction=0.2, pad_len=20)
smoother.smooth(df[2][1])
# generate intervals
low, up = smoother.get_intervals('sigma_interval')

# plot the first smoothed timeseries with intervals based on the first phase
plt.figure(figsize=(100,20))
plt.plot(smoother.smooth_data[0], linewidth=3, color='blue')
plt.plot(smoother.data[0], '.k')
plt.xlabel('timeline')
plt.fill_between(range(len(smoother.data[0])), low[0], up[0], alpha=0.3)


smoother = SpectralSmoother(smooth_fraction=0.2, pad_len=20)
smoother.smooth(df[2][2])
low, up = smoother.get_intervals('sigma_interval')

# plot the first smoothed timeseries with intervals based on the second phase
plt.figure(figsize=(100,20))
plt.plot(smoother.smooth_data[0], linewidth=3, color='orange')
plt.plot(smoother.data[0], '.k')
plt.xlabel('timeline')
plt.fill_between(range(len(smoother.data[0])), low[0], up[0], alpha=0.3,color='orange')


smoother = SpectralSmoother(smooth_fraction=0.2, pad_len=20)
smoother.smooth(df[2][3])
low, up = smoother.get_intervals('sigma_interval')

# plot the first smoothed timeseries with intervals based on the third phase
plt.figure(figsize=(100,20))
plt.plot(smoother.smooth_data[0], linewidth=3, color='green')
plt.plot(smoother.data[0], '.k')
plt.xlabel('timeline')
plt.fill_between(range(len(smoother.data[0])), low[0], up[0], alpha=0.3,color='green')


# In[ ]:





# In[ ]:





# ### Spline Smoothing

# #### Spline smoothing is a curve fitting technique used to smooth out a set of data points by fitting a smooth curve to the data. The method involves dividing the data into smaller segments and fitting a piecewise polynomial function, called a spline, to each segment.

# In[9]:


# operate smoothing
smoother = SplineSmoother(n_knots=300, spline_type='natural_cubic_spline')
smoother.smooth(df[2][1])
# generate inntervals
low, up = smoother.get_intervals('prediction_interval')

# plot the first smoothed timeseries with intervals based on the first phase
plt.figure(figsize=(100,20))
plt.plot(smoother.smooth_data[0], linewidth=3, color='blue')
plt.plot(smoother.data[0], '.k')
plt.xlabel('timeline')
plt.fill_between(range(len(smoother.data[0])), low[0], up[0], alpha=0.3)


smoother = SplineSmoother(n_knots=300, spline_type='natural_cubic_spline')
smoother.smooth(df[2][2])
low, up = smoother.get_intervals('prediction_interval')


# plot the first smoothed timeseries with intervals based on the second phase
plt.figure(figsize=(100,20))
plt.plot(smoother.smooth_data[0], linewidth=3, color='orange')
plt.plot(smoother.data[0], '.k')
plt.xlabel('timeline')
plt.fill_between(range(len(smoother.data[0])), low[0], up[0], alpha=0.3,color='orange')


smoother = SplineSmoother(n_knots=300, spline_type='natural_cubic_spline')
smoother.smooth(df[2][3])
low, up = smoother.get_intervals('prediction_interval')

# plot the first smoothed timeseries with intervals based on the third phase
plt.figure(figsize=(100,20))
plt.plot(smoother.smooth_data[0], linewidth=3, color='green')
plt.plot(smoother.data[0], '.k')
plt.xlabel('timeline')
plt.fill_between(range(len(smoother.data[0])), low[0], up[0], alpha=0.3,color='green')


# In[ ]:





# In[ ]:





# ### Lowess Smoothing

# #### Lowess (Locally Weighted Scatterplot Smoothing) is a non-parametric method used for smoothing a scatterplot of data. The method works by fitting a locally weighted regression line to the data points, where the weights are based on a kernel function that assigns higher weights to nearby points and lower weights to more distant points.

# In[10]:


# operate smoothing
smoother = LowessSmoother(smooth_fraction=0.005, iterations=1)
smoother.smooth(df[2][1])
# generate intervals
low, up = smoother.get_intervals('prediction_interval')

# plot the first smoothed timeseries with intervals based on the first phase
plt.figure(figsize=(100,20))
plt.plot(smoother.smooth_data[0], linewidth=3, color='blue')
plt.plot(smoother.data[0], '.k')
plt.xlabel('timeline')
plt.fill_between(range(len(smoother.data[0])), low[0], up[0], alpha=0.3)


smoother = LowessSmoother(smooth_fraction=0.005, iterations=1)
smoother.smooth(df[2][2])
low, up = smoother.get_intervals('prediction_interval')

# plot the first smoothed timeseries with intervals based on the second phase
plt.figure(figsize=(100,20))
plt.plot(smoother.smooth_data[0], linewidth=3, color='orange')
plt.plot(smoother.data[0], '.k')
plt.xlabel('timeline')
plt.fill_between(range(len(smoother.data[0])), low[0], up[0], alpha=0.3,color='orange')


smoother = LowessSmoother(smooth_fraction=0.005, iterations=1)
smoother.smooth(df[2][3])
low, up = smoother.get_intervals('prediction_interval')


# plot the first smoothed timeseries with intervals based on the third phase
plt.figure(figsize=(100,20))
plt.plot(smoother.smooth_data[0], linewidth=3, color='green')
plt.plot(smoother.data[0], '.k')
plt.xlabel('timeline')
plt.fill_between(range(len(smoother.data[0])), low[0], up[0], alpha=0.3,color='green')


# In[ ]:





# In[ ]:





# ### Kalman Smoothing

# #### Kalman smoothing is a statistical method used to estimate the hidden state of a dynamic system based on a sequence of noisy observations. The method uses a mathematical model of the system dynamics to predict the state of the system at each time step, and then updates the state estimate based on the latest observation.

# In[11]:


# operate smoothing
smoother = KalmanSmoother(component='level_season', 
                          component_noise={'level':0.1, 'season':0.1}, 
                          n_seasons=24)
smoother.smooth(df[2][1])
# generate intervals
low, up = smoother.get_intervals('kalman_interval')

# plot the first smoothed timeseries with intervals based on the first phase
plt.figure(figsize=(100,20))
plt.plot(smoother.smooth_data[0], linewidth=3, color='blue')
plt.plot(smoother.data[0], '.k')
plt.xlabel('timeline')
plt.fill_between(range(len(smoother.data[0])), low[0], up[0], alpha=0.3)


smoother = KalmanSmoother(component='level_season', 
                          component_noise={'level':0.1, 'season':0.1}, 
                          n_seasons=24)
smoother.smooth(df[2][2])
low, up = smoother.get_intervals('kalman_interval')

# plot the first smoothed timeseries with intervals based on the second phase
plt.figure(figsize=(100,20))
plt.plot(smoother.smooth_data[0], linewidth=3, color='orange')
plt.plot(smoother.data[0], '.k')
plt.xlabel('timeline')
plt.fill_between(range(len(smoother.data[0])), low[0], up[0], alpha=0.3,color='orange')


smoother = KalmanSmoother(component='level_season', 
                          component_noise={'level':0.1, 'season':0.1}, 
                          n_seasons=24)
smoother.smooth(df[2][3])
low, up = smoother.get_intervals('kalman_interval')

# plot the first smoothed timeseries with intervals based on the third phaseplt.figure(figsize=(100,20))
plt.plot(smoother.smooth_data[0], linewidth=3, color='green')
plt.plot(smoother.data[0], '.k')
plt.xlabel('timeline')
plt.fill_between(range(len(smoother.data[0])), low[0], up[0], alpha=0.3,color='green')


# In[ ]:





# In[ ]:





# #### In addition to the Sinusoidal filtering techniques , I will also make use of the Savitzky-Golay smoothing.

# ### Savitzky-Golay Smoothing

# #### Savitzky-Golay smoothing is a signal processing technique used to smooth out a signal by fitting a polynomial function to a moving window of data points. The method involves convolving the signal with a set of weighted coefficients, which are derived from a polynomial fit to the data within the window.
# 
# #### The Savitzky-Golay filter can be used to smooth out noisy data, remove high-frequency noise, and estimate the derivative of a signal. It is particularly useful in applications such as spectroscopy, chromatography, and time-series analysis.

# In[12]:


#analysing the waveform with smaller window length inorder to examine the pattern of the waveform.
window_length = 11
poly_order = 4

# Apply the Savitzky-Golay filter
y1 = savgol_filter(df[2][1], window_length, poly_order)
y2 = savgol_filter(df[2][2], window_length, poly_order)
y3 = savgol_filter(df[2][3], window_length, poly_order)

plt.figure(figsize=(100,20))
plt.plot(y1,color='blue')
plt.grid()
plt.show()

plt.figure(figsize=(100,20))
plt.plot(y2,color='green')
plt.grid()
plt.show()

plt.figure(figsize=(100,20))
plt.plot(y3,color='orange')
plt.grid()
plt.show()


# In[13]:


# smoothening the waveform further by increasing the window length in a way that all the patterns are represented by a much simpler waveform
window_length = 101
poly_order = 4

# Apply the Savitzky-Golay filter for the first file
y1 = savgol_filter(df[2][1], window_length, poly_order)
y2 = savgol_filter(df[2][2], window_length, poly_order)
y3 = savgol_filter(df[2][3], window_length, poly_order)

# plot the smoothed data for phase 1 
plt.figure(figsize=(100,20))
plt.plot(y1,color='blue')
plt.grid()
plt.show()

# plot the smoothed data for phase 2
plt.figure(figsize=(100,20))
plt.plot(y2,color='green')
plt.grid()
plt.show()

# plot the smoothed data for phase 3
plt.figure(figsize=(100,20))
plt.plot(y3,color='orange')
plt.grid()
plt.show()


# #### Here for each of the smoothing algorithms I have chosen the optimal parameters to see the pattern of the waveforms of all the 3 phases data for the first activity.
# #### From the above plots for all the smoothing techniques we can see that all the waveforms follow a particular pattern if the optimal parameters for each of the smoothing are chosen. Hence any changes in between are the sign of an activity.  This means that the activity of collecting the 1000 instances in a second may or may not be a defect.

# In[ ]:





# In[ ]:





# #### I will define all the smoothing functions used earlier for transforming the data of each activity for all the 3 phases separately

# In[14]:


def exp_smoth(value):
    smoother = ExponentialSmoother(window_len=100, alpha=0.3)
    
    y1=smoother.smooth(value[1]).smooth_data[0]  
    y2=smoother.smooth(value[2]).smooth_data[0]
    y3=smoother.smooth(value[3]).smooth_data[0]  
    
    return y1,y2,y3


# In[15]:


def conv_smoth(value):
    smoother = ConvolutionSmoother(window_len=10, window_type='ones')
    
    y1=smoother.smooth(value[1]).smooth_data[0]  
    y2=smoother.smooth(value[2]).smooth_data[0]
    y3=smoother.smooth(value[3]).smooth_data[0]  
    
    return y1,y2,y3


# In[16]:


def spect_smoth(value):
    smoother = SpectralSmoother(smooth_fraction=0.2, pad_len=20)
    
    y1=smoother.smooth(value[1]).smooth_data[0]  
    y2=smoother.smooth(value[2]).smooth_data[0]
    y3=smoother.smooth(value[3]).smooth_data[0]  
    
    return y1,y2,y3


# In[17]:


def spline_smoth(value):
    smoother = SplineSmoother(n_knots=30, spline_type='natural_cubic_spline')
    
    y1=smoother.smooth(value[1]).smooth_data[0]  
    y2=smoother.smooth(value[2]).smooth_data[0]
    y3=smoother.smooth(value[3]).smooth_data[0]  
    
    return y1,y2,y3


# In[18]:


def lowess_smoth(value):
    smoother = LowessSmoother(smooth_fraction=0.05, iterations=1)
    
    y1=smoother.smooth(value[1]).smooth_data[0]  
    y2=smoother.smooth(value[2]).smooth_data[0]
    y3=smoother.smooth(value[3]).smooth_data[0]  
    
    return y1,y2,y3


# In[19]:


def kalman_smoth(value):
    smoother = KalmanSmoother(component='level_season', 
                          component_noise={'level':0.1, 'season':0.1}, 
                          n_seasons=24)
    
    y1=smoother.smooth(value[1]).smooth_data[0]  
    y2=smoother.smooth(value[2]).smooth_data[0]
    y3=smoother.smooth(value[3]).smooth_data[0]  
    
    return y1,y2,y3


# In[20]:


#function to smoothen the waveform
def sg_smoth(value):
    window_length = 101
    poly_order = 4
    # Apply the Savitzky-Golay filter
    y1 = savgol_filter(value[1], window_length, poly_order)
    y2 = savgol_filter(value[2], window_length, poly_order)
    y3 = savgol_filter(value[3], window_length, poly_order)
    return y1,y2,y3


# In[ ]:





# In[ ]:





# #### Now, for all the activities the smoothed values are stored under three dictionaries corresponding to each phase data using one of the smoothing functions defined earlier
# #### Currently I will choose the Savitzky-Golay Smoothing

# In[21]:


a,b,c={},{},{} #just like 1,2,3 ==> a,b,c are the three phases for smoothen graph.
for i in range(2,319):
    a[i],b[i],c[i]=sg_smoth(df[i])


# In[ ]:





# #### The function to determine the indices of the peaks and crests of the smoothed values are estimated for each phase data of each activity using the 'find_peaks' function from the scipy library is defined

# In[22]:


#function to find peaks and crests of the smoothened waveform.
def minima_maxima(value):
    peak_indices, _= find_peaks(value)
    minima_indices, _= find_peaks(-value)
    return peak_indices,minima_indices


# In[ ]:





# #### Now, the indices of the peaks and crests for each activity are stored in the 6 dictionaries of which 3 belong to estimated peak values of each of the 3 phase data and the other 2 belong to the estimated crest values of each of the 3 phase data.

# In[23]:


# from here we have maximas and minimas for all the phases
maxima_a={}
minima_a={}
maxima_b={}
minima_b={}
maxima_c={}
minima_c={}
for i in range(2,319):
    maxima_a[i],minima_a[i]=minima_maxima(a[i])
    maxima_b[i],minima_b[i]=minima_maxima(b[i])
    maxima_c[i],minima_c[i]=minima_maxima(c[i])


# In[ ]:





# #### Using the Interquartile range of the estimation of the outliers, the outliers are determined for each of the 3 phase data for each activity

# In[24]:


#function to calculate the upper limit and lower limit. Anything outside this range will be an outlier or in our case will be a anomaly
def outlier(value):
    q1=np.quantile(value,0.25)
    q3=np.quantile(value,0.75)
    iqr=q3-q1
    ul=q3+(1.5*iqr)
    ll=q1-(1.5*iqr)
    return ul,ll


# In[ ]:





# #### This is to check for all the datafiles for the first phase data whether there are any datafiles or the activites that contain outliers

# In[25]:


#To check out of all the datafiles which actually have outliers (Please note: no smoothening is done here. It is for the general or given values)
uplimit={}
lowlimit={}
for i in range(2,319):
    uplimit[i],lowlimit[i]=outlier(a[i])
     #checking outliers for all the files
    if (a[i]>uplimit[i]).any() or (a[i]<lowlimit[i]).any():
        print(i,'.txt file has outliers')


# #### It is observed that the activity corresponding to the datafile numbered 318 has outliers.
# #### Similarly for the other 2 phase data, same process is carried out and it is found that the same datafile numbered 318 has outliers in both the cases.

# In[ ]:





# #### Now, the waveforms of the datafile numbered 318 are observed for all the three phases using the Savotzky-Golay Smoothing

# In[26]:


# smoothening the waveform further by increasing the window length in a way that all the patterns are represented by a much simpler waveform
window_length = 11
poly_order = 4

# Apply the Savitzky-Golay filter
y1 = savgol_filter(df[318][1], window_length, poly_order)
y2 = savgol_filter(df[318][2], window_length, poly_order)
y3 = savgol_filter(df[318][3], window_length, poly_order)

plt.figure(figsize=(100,10))
plt.plot(y1,color='blue')
plt.grid()
plt.show()

plt.figure(figsize=(100,10))
plt.plot(y2,color='green')
plt.grid()
plt.show()

plt.figure(figsize=(100,10))
plt.plot(y3,color='orange')
plt.grid()
plt.show()


# #### We can see that the waveform somewhat becomes linear at the end. This may indicate failure or may be another activity that does not correspond to this data and hence an anomaly datafile

# In[ ]:





# #### For the next step I define two functions, one for the estimated indices of the peak values such that the outliers are checked using the earlier defined function for the values that only belong corresponding to the indices of the peak values and the other for the estimated indices of the crest values such that the outliers are checked using the earlier defined function for the values that only belong corresponding to the indices of the crest values.
# #### The values that are concluded as outliers are assigned the value of 1 and remaining values are assigned the value of 0 for both the functions.
# #### These engineered values are stored in a dataframe with two features for the first funnction, one feature belonging to the peak values estimated and the other feature corresponding to being outlier or not. Similarly for the other function corresponding to crest values same process is carried out,

# In[27]:


#The outliers in the smoothened peaks and crests are labeled as defects here.

def phase_defected_peaks(value):
    u_peak,l_peak=outlier(value)
    x=np.zeros(len(value))
    x[np.where(value<l_peak)]=1
    x[np.where(value>u_peak)]=1
    df_peak=pd.DataFrame(data=[value,x]).T
    df_peak.columns=['Phase_Peak','Fault']
    return df_peak


def phase_defected_crests(value):
    
    u_crest,l_crest=outlier(value)
    y=np.zeros(len(value))
    y[np.where(value>u_crest)]=1
    y[np.where(value>u_crest)]=1
    df_crest=pd.DataFrame(data=[value,y]).T
    df_crest.columns=['Phase_Crest','Fault']
    
    return df_crest


# In[ ]:





# #### Now using the above two defined functions the dataframes resulted for each activity are stored in the 6 dictionaries of which 3 belong to estimated peak values of each of the 3 phase data and the other 2 belong to the estimated crest values of each of the 3 phase data.
# #### 6 master dataframes are also defined to store the data of all the activities of peaks and crest data of each of the 3 phase data in a single dataframe.

# In[28]:


#calculating the peaks and crest for all the files and compiling them in a master data frame for each phase and type of peaks.
d_peak_a={}
d_crest_a={}
d_peak_master_a=pd.DataFrame()
d_crest_master_a=pd.DataFrame()
d_peak_b={}
d_crest_b={}
d_peak_master_b=pd.DataFrame()
d_crest_master_b=pd.DataFrame()
d_peak_c={}
d_crest_c={}
d_peak_master_c=pd.DataFrame()
d_crest_master_c=pd.DataFrame()
for i in range(2,319):
    d_peak_a[i]=phase_defected_peaks(a[i][maxima_a[i]])
    d_crest_a[i]=phase_defected_crests(a[i][minima_a[i]])
    d_peak_master_a=d_peak_master_a.append(d_peak_a[i])
    d_crest_master_a=d_crest_master_a.append(d_crest_a[i])
    
    d_peak_b[i]=phase_defected_peaks(b[i][maxima_b[i]])
    d_crest_b[i]=phase_defected_crests(b[i][minima_b[i]])
    d_peak_master_b=d_peak_master_b.append(d_peak_b[i])
    d_crest_master_b=d_crest_master_b.append(d_crest_b[i])
    
    d_peak_c[i]=phase_defected_peaks(c[i][maxima_c[i]])
    d_crest_c[i]=phase_defected_crests(c[i][minima_c[i]])
    d_peak_master_c=d_peak_master_c.append(d_peak_c[i])
    d_crest_master_c=d_crest_master_c.append(d_crest_c[i])


# In[ ]:





# #### Next, the function to create the training and testing data is defined

# In[29]:


#function to make train data sample and test data sample
def train_test(data):
    train_data = data.sample(frac=0.8, random_state=1)
    test_data = data.drop(train_data.index)
    train_features = train_data.drop('Fault', axis=1)
    train_labels = train_data['Fault']
    test_features = test_data.drop('Fault', axis=1)
    test_labels = test_data['Fault']
    
    return train_features,train_labels,test_features,test_labels


# #### The peak and crest dataframes for each phase data are concated and the final 3 master dataframes are obtained.
# #### For each 3 master dataframes obtained now, three sets of training and testimg data are defined

# In[30]:


#compiling all the peaks and ccrests for each phase in a master dataframe
d1_a=pd.concat([d_crest_master_a['Phase_Crest'],d_peak_master_a['Phase_Peak']])
d2_a=pd.concat([d_crest_master_a['Fault'],d_peak_master_a['Fault']])

d1_b=pd.concat([d_crest_master_b['Phase_Crest'],d_peak_master_b['Phase_Peak']])
d2_b=pd.concat([d_crest_master_b['Fault'],d_peak_master_b['Fault']])

d1_c=pd.concat([d_crest_master_c['Phase_Crest'],d_peak_master_c['Phase_Peak']])
d2_c=pd.concat([d_crest_master_c['Fault'],d_peak_master_c['Fault']])

master_dataset_a=pd.concat([d1_a,d2_a],axis=1)
master_dataset_b=pd.concat([d1_b,d2_b],axis=1)
master_dataset_c=pd.concat([d1_c,d2_c],axis=1)


# In[ ]:





# #### I will start with the Random Forest Classifier for dealing with this problem statement.
# #### So, I have formed a semi supervised machine learning binary classfication problem statement.

# In[31]:


# create the Random Forest classifier
rf= RandomForestClassifier(n_estimators=100, random_state=1)


# In[ ]:





# #### The 3 sets of training and testing data are defined for each of the 3 phase data
# #### The earlier defined model is used to train for 3 training data individually.

# In[158]:


train_features_a,train_labels_a,test_features_a,test_labels_a=train_test(master_dataset_a)
train_features_a,train_labels_a,test_features_a,test_labels_a=train_test(master_dataset_a)
# train the model
rf_a=rf.fit(train_features_a, train_labels_a)
# make predictions on the test set
predictions_a = rf_a.predict(test_features_a)
# evaluate the model performance
print(confusion_matrix(test_labels_a, predictions_a))
print(classification_report(test_labels_a, predictions_a))


# #### The confusion matrix shows that there are 116 instances of the negative class (0) and 18 instances of the positive class (1) in the dataset. The model correctly predicted 114 instances of the negative class and 18 instances of the positive class. There were two instances of the negative class that were incorrectly classified as positive.
# 
# #### The classification report shows that the model has high precision and recall for both classes, with a weighted average F1 score of 0.99. Precision is the ratio of true positives to the total number of predicted positive instances. Recall is the ratio of true positives to the total number of actual positive instances. F1 score is the harmonic mean of precision and recall. The support column indicates the number of instances in each class.
# 
# #### Overall, the model has high accuracy and performs well in predicting both classes, with slightly higher performance for the negative class. However, further analysis of the data and model performance would be necessary to assess the reliability and usefulness of the model for its intended purpose.

# In[ ]:





# In[159]:


train_features_b,train_labels_b,test_features_b,test_labels_b=train_test(master_dataset_b)
train_features_b,train_labels_b,test_features_b,test_labels_b=train_test(master_dataset_b)

rf_b=rf.fit(train_features_b, train_labels_b)

predictions_b = rf_b.predict(test_features_b)

print(confusion_matrix(test_labels_b, predictions_b))
print(classification_report(test_labels_b, predictions_b))


# #### The confusion matrix shows that there are 91 instances of the negative class (0) and 17 instances of the positive class (1) in the dataset. The model correctly predicted 88 instances of the negative class and 17 instances of the positive class. There were three instances of the negative class that were incorrectly classified as positive.
# 
# #### The classification report shows that the model has high precision and recall for both classes, with a weighted average F1 score of 0.97. Precision is the ratio of true positives to the total number of predicted positive instances. Recall is the ratio of true positives to the total number of actual positive instances. F1 score is the harmonic mean of precision and recall. The support column indicates the number of instances in each class.
# 
# #### Overall, the model has high accuracy and performs well in predicting both classes, with slightly higher performance for the negative class. However, further analysis of the data and model performance would be necessary to assess the reliability and usefulness of the model for its intended purpose.

# In[ ]:





# In[160]:


train_features_c,train_labels_c,test_features_c,test_labels_c=train_test(master_dataset_c)
train_features_c,train_labels_c,test_features_c,test_labels_c=train_test(master_dataset_c)

rf_c=rf.fit(train_features_c, train_labels_c)

predictions_c = rf_c.predict(test_features_c)

print(confusion_matrix(test_labels_c, predictions_c))
print(classification_report(test_labels_c, predictions_c))


# #### The Random Forest classifier performed very well on the test dataset. The confusion matrix shows that there were 87 true positives and 15 true negatives, and no false positives or false negatives. This means that the classifier correctly predicted all of the samples in the test dataset.
# 
# #### The classification report shows that the precision, recall, and F1-score for both classes (0 and 1) are all 1.00, which indicates perfect classification performance. The overall accuracy is also 1.00, which further confirms that the classifier performed extremely well on this dataset.

# In[ ]:





# In[ ]:





# ### Cross-validation to evaluate the performance of the classifier on multiple test sets

# In[ ]:





# In[161]:


scores_a = cross_val_score(rf_a, train_features_a, train_labels_a, cv=5)


# In[162]:


scores_b = cross_val_score(rf_b, train_features_b, train_labels_b, cv=5)


# In[163]:


scores_c = cross_val_score(rf_c, train_features_c, train_labels_c, cv=5)


# In[164]:


scores_a


# #### Each score in the array represents the performance of the model on a different fold of the data, where the model was trained on the remaining folds. In this case, the array contains five scores, with values ranging from 0.915 to 0.920.
# 
# #### Based on these scores, it seems that the model is performing relatively consistently across the different folds, with scores that are fairly close together. This suggests that the model is not overfitting to the training data and is likely to generalize well to new, unseen data.

# In[ ]:





# In[165]:


scores_b


# #### Each score in the array represents the performance of the model on a different fold of the data, where the model was trained on the remaining folds. In this case, the array contains five scores, with values ranging from 0.914 to 0.921.
# 
# #### Based on these scores, it seems that the model is performing relatively consistently across the different folds, with scores that are fairly close together. This suggests that the model is not overfitting to the training data and is likely to generalize well to new, unseen data.

# In[ ]:





# In[166]:


scores_c


# #### The output of cross_val_score is an array of scores, where each score corresponds to the performance of the classifier on a different fold of the data. In this case, the output shows that the classifier achieved a score of around 0.92 on each of the 5 folds of the data.
# 
# #### These cross-validation scores can be useful for evaluating the generalization performance of the classifier, as they provide an estimate of how well the classifier is likely to perform on new, unseen data. By performing cross-validation, we can get a more robust estimate of the classifier's performance than we would by simply evaluating it on a single test set.

# In[ ]:





# #### The model trained is working well as per the chosen parameters for all the 3 engineered data using the Sovatzky-Golay Smoothing.
# #### The other models can be trained using the other smoothing algorithms defined as well as the other classfication algorithms, but as the present model is performing perfect, I will not train any alternative classifiers.

# In[ ]:





# In[ ]:





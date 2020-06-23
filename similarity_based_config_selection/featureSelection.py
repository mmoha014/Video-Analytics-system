from feature_selector import FeatureSelector
import pandas as pd
import numpy as np


# generate related variables
from numpy import mean
from numpy import std
from numpy.random import randn
from numpy.random import seed
from matplotlib import pyplot
# seed random number generator
seed(1)
# prepare data
data1 = 20 * randn(1000) + 100
data2 = data1 + (10 * randn(1000) + 50)
# summarize
print('data1: mean=%.3f stdv=%.3f' % (mean(data1), std(data1)))
print('data2: mean=%.3f stdv=%.3f' % (mean(data2), std(data2)))
# plot
# pyplot.scatter(data1, data2)
# pyplot.show()
covariance = np.cov(data1,data2)
"""
The diagonal of the matrix contains the covariance between each variable and itself. The other values in the matrix represent the covariance between the two variable
The sign of the covariance can be interpreted as whether the two variables change in the same direction (positive) or change in different directions (negative). The magnitude of the covariance is not easily interpreted. A covariance value of zero indicates that both variables are completely independent.
"""
print("covariance", covariance)

from scipy.stats import pearsonr
from scipy.stats import spearmanr

# calculate Pearson's correlation
corr, _ = pearsonr(data1, data2)
print('Pearsons correlation: %.3f' % corr)
corr, _ = spearmanr(data1, data2)
print('Spearmans correlation: %.3f' % corr)
####################################################################################
# import numpy as np
# import pandas as pd
"""
import matplotlib.pyplot as plt
data = pd.read_csv('top_configs.csv')#'https://www.dropbox.com/s/4jgheggd1dak5pw/data_visualization.csv?raw=1', index_col=0)
data = data.drop(columns=['segment'])
# train.drop(columns = ['objectiveFunction_value'])
corr = data.corr()
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(corr,cmap='coolwarm', vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,len(data.columns),1)
ax.set_xticks(ticks)
plt.xticks(rotation=90)
ax.set_yticks(ticks)
ax.set_xticklabels(data.columns)
ax.set_yticklabels(data.columns)
plt.show()
"""
import seaborn as sns
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
np.random.seed(123)
data = pd.read_csv('top_configs.csv')#'https://www.dropbox.com/s/4jgheggd1dak5pw/data_visualization.csv?raw=1', index_col=0)
data = data.drop(columns=['segment'])
label_encoder = LabelEncoder()
data.iloc[:,0] = label_encoder.fit_transform(data.iloc[:,0]).astype('float64')
corr = data.corr()
sns.heatmap(corr,annot=True,cmap="RdYlGn")
plt.show()
#################### store data as csv file in panda.dataframe  format ############
# dataformat = {
#     'segment':[],
#     'frame_size':[],
#     'frame_rate':[],
#     'model':[],
#     'execution_time':[],
#     'average_accuracy':[],
#     'fraction': [],
#     'objectiveFunction_value':[]
# }
# fp = open('/home/morteza/Videos/traffic camera/profile/CAM822_Virginia/profile_on_all_configs/new_top_configs_gt1.txt')
# line_counter = 0
# for line in fp.readlines():
#     if line_counter % 6 == 0:
#         line_counter += 1
#         continue

#     data = line.split(',')
#     line = int(data[0][1:])
#     segm = int(data[1])
#     fr_size = int(data[2][2:-1])
#     fr_rate = int(data[3][2:-1])
#     model = data[4]
#     extime = float(data[5])
#     avg_acc = float(data[6])
#     frac = float(data[7])
#     objFunc = float(data[8][1:-2])
#     dataformat['segment'].append(segm)
#     dataformat['frame_size'].append(fr_size)
#     dataformat['frame_rate'].append(fr_rate)
#     dataformat['model'].append(model)
#     dataformat['execution_time'].append(extime)
#     dataformat['average_accuracy'].append(avg_acc)
#     dataformat['fraction'].append(frac)
#     dataformat['objectiveFunction_value'].append(objFunc)
#     print(line)

#     line_counter += 1

# ##### save data
# df = pd.DataFrame(dataformat)
# compression_opts = dict(method='zip', archive_name='out.csv')  
# df.to_csv('out.zip', index=False, compression=compression_opts) 
################### analyzing data ###############################################
"""
# train = pd.read_csv('/home/morteza/Documents/FeatureSelector/home-credit-default-risk/application_train.csv')
# train_labels = train['TARGET']
# train = train.drop(columns = ['TARGET'])
train = pd.read_csv('top_configs.csv')
train_labels = train['objectiveFunction_value']
train = train.drop(columns = ['objectiveFunction_value'])
print(train.head())

fs = FeatureSelector(data = train, labels = train_labels)
fs.identify_missing(missing_threshold=0.6)
missing_features = fs.ops['missing']
print("missing features: ",missing_features[:10])
fs.plot_missing()
print("statistics of missing features: ",fs.missing_stats.head(10))

######## 2. Single Unique Value #######
fs.identify_single_unique()
single_unique = fs.ops['single_unique']
print('features with a single unique value', single_unique)
fs.plot_unique()

#### we can access a dataframe with the number of unique values per feature
fs.unique_stats.sample(5)

######## 3. Collinear (highly correlated) Features #######
#This method finds pairs of collinear features based on the Pearson correlation coefficient.
#correlations are only calculated between numeric columns
fs.identify_collinear(correlation_threshold=0.975)
correlated_features = fs.ops['collinear']
print("features with a correlation magnitude greater than 0.97", correlated_features)#[:5]
fs.plot_collinear()
fs.plot_collinear(plot_all=True)
"""
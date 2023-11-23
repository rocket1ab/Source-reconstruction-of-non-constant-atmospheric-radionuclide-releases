#%%
import scipy.io
import numpy as np
import pandas as pd
from sklearn import preprocessing
import random


########### 构造时序特征指标 ###########
## 构造时域特征指标
# 波动率
def cal_wave_rate(data):
    # 先对数据进行标准归一化
    data = data.reshape(-1,1)
    min_max_scaler = preprocessing.MinMaxScaler()
    data_minmax = min_max_scaler.fit_transform(data)
    #print(data_minmax)
    wave_rate = np.percentile(data_minmax,90)-np.percentile(data_minmax,10)
    return wave_rate

# 平均值
def average(data):
    data_aver = np.mean(data)
    return data_aver

# 中位数
def median(data):
    data_median = np.median(data)
    return data_median

# 样本熵
# SampEn  计算时间序列U的样本熵
# 输入：U是数据一维数组array
#       m重构维数，一般选择1或2，优先选择2，一般不取m>2
#       r 阈值大小，一般选择r=0.1~0.25*Std(data)
# 输出：样本熵值大小
def SampEn(U, m, r):
     def _maxdist(x_i, x_j):
           return max([abs(ua - va) for ua, va in zip(x_i, x_j)])
     def _phi(m):
          x = [[U[j] for j in range(i, i + m - 1 + 1)] for i in range(N - m + 1)]
          B = [(len([1 for x_j in x if _maxdist(x_i, x_j) <= r]) - 1.0) / (N - m) for x_i in x]
          return (N - m + 1.0)**(-1) * sum(B)
     N = len(U)
     return -np.log(_phi(m+1) / _phi(m))

## 构造频域特征指标
class Feature_fft(object):
    def __init__(self, sequence_data):
        self.data = sequence_data
        fft_trans = np.abs(np.fft.fft(sequence_data))
        self.dc = fft_trans[0]
        self.freq_spectrum = fft_trans[1:int(np.floor(len(sequence_data) * 1.0 / 2)) + 1]
        self._freq_sum_ = np.sum(self.freq_spectrum)

    def fft_mean(self):
        return np.mean(self.freq_spectrum)

    def fft_shape_mean(self):
        shape_sum = np.sum([x * self.freq_spectrum[x]
                            for x in range(len(self.freq_spectrum))])
        return 0 if self._freq_sum_ == 0 else shape_sum * 1.0 / self._freq_sum_


    def fft_all(self):
        '''
        Get all fft features in one function
        :return: All fft features in one list
        '''
        feature_all = list()
        feature_all.append(self.fft_shape_mean())
        feature_all.append(self.fft_mean())
        return feature_all

########### 初始化参数 ###########
window_size = 5
window = window_size-1 # 滑动窗口大小
Sample_amount = 1000 # 样本数量
Sta_num = 4 # 站点个数
Scaling_factor = 100000 # 模拟数据和观测数据缩放因子
start_time = 11 # 选取时间段的起始时间
end_time = 40 # 选取时间段的截止时间
mu = 0 # 高斯噪声均值

########### 初始化空列表存储特征集合 ###########
# 时域指标
wave_rate_all = [] # 波动率
data_aver = [] # 平均值
data_median = [] # 中位数
SampEn_all = [] # 样本熵
#频域指标
fft_shape_mean = [] # 功率谱密度函数的形状统计指标
fft_mean = [] # 功率谱密度函数的幅度统计指标
for i in range(Sta_num):
    wave_rate_all.append([])
    data_aver.append([])
    data_median.append([])
    SampEn_all.append([])
    fft_shape_mean.append([])
    fft_mean.append([])
    
########### Read data file ###########
source_data = './datafile.mat'
data_info = scipy.io.loadmat(source_data)
sim_data = data_info['sim_data']
Sample_amount = sim_data.shape[0]
obs_data = data_info['obs_data']
Scaling_factor = data_info['Scaling_factor']
release_loc = data_info['release_loc']

#####      滑动均值滤波函数   #####
def rolling(data):
    data = data.rolling(window_size).sum()
    return data

data_ce = obs_data
std_ce = []
max_ce = []
median_ce = []
aver_ce = []

# 测点1
data_ce1 = data_ce[:,0]
data_ce1 = data_ce1[start_time:end_time] # 去恒定释放率的那部分
data_ce1_max = max(data_ce1) # 找到第一个测点的最大值
data_ce1_median = median(data_ce1)
data_ce1_aver = average(data_ce1)
max_ce.append(data_ce1_max)
std_ce.append(np.std(data_ce1))
median_ce.append(data_ce1_median)
aver_ce.append(data_ce1_aver)

# 测点2
data_ce2 = data_ce[:,1]
data_ce2 = data_ce2[start_time:end_time] # 去恒定释放率的那部分
data_ce2_max = max(data_ce2) # 找到第二个测点的最大值
data_ce2_median = median(data_ce2)
data_ce2_aver = average(data_ce2)
max_ce.append(data_ce2_max)
std_ce.append(np.std(data_ce2))
median_ce.append(data_ce2_median)
aver_ce.append(data_ce2_aver)

# 测点3
data_ce3 = data_ce[:,2]
data_ce3 = data_ce3[start_time:end_time] # 去恒定释放率的那部分
data_ce3_max = max(data_ce3) # 找到第二个测点的最大值
data_ce3_median = median(data_ce3)
data_ce3_aver = average(data_ce3)
max_ce.append(data_ce3_max)
std_ce.append(np.std(data_ce3))
median_ce.append(data_ce3_median)
aver_ce.append(data_ce3_aver)

# 测点4
data_ce4 = data_ce[:,3]
data_ce4 = data_ce4[start_time:end_time] # 去恒定释放率的那部分
data_ce4_max = max(data_ce4) # 找到第二个测点的最大值
data_ce4_median = median(data_ce4)
data_ce4_aver = average(data_ce4)
max_ce.append(data_ce4_max)
std_ce.append(np.std(data_ce4))
median_ce.append(data_ce4_median)
aver_ce.append(data_ce4_aver)


# 获取释放率量级因子
def getproportion(sim_data, median_ce):
    median_ce = np.array(median_ce)
    location_aver = sim_data.mean(axis=0)
    location_aver = location_aver[start_time:end_time,:]
    location_aver_median = np.median(location_aver,axis=0)
    proportion = median_ce/location_aver_median
    proportion = proportion.mean()
    return proportion

proportion_element = getproportion(sim_data, median_ce)

# 模拟数据预处理与特征提取
for index_col in  range(Sta_num):
    wave_rate_all_temp = []
    data_aver_temp = []
    data_median_temp = []
    SampEn_all_temp = []
    fft_shape_mean_temp = []
    fft_mean_temp = []
    for k in range(Sample_amount):
        df_all = pd.DataFrame(sim_data[k,:,:]) # 读取模拟数据
        # 滑动均值滤波处理
        df = df_all #保存数据到df
        df_all = rolling(df_all)
        df = Scaling_factor*np.array(df[0:window]) #取出前3行数据用于补充后面的NAN值
        data_all = df_all.values * Scaling_factor
        data_all = (1/window_size)*data_all[window:]
        data_all = np.vstack((df,data_all))

        # 乘以释放率量级因子
        data = data_all[:,index_col]
        data = proportion_element*np.array(data)
        data = data[start_time:end_time]
        
        # 添加高斯噪声
        if(std_ce[index_col]<np.std(data)):
            sign = -1
        if(std_ce[index_col]>=np.std(data)):
            sign = 1
        sigma = abs(std_ce[index_col]-np.std(data))

        for i in range(data.size):
            #random.seed(i)
            data[i] = data[i] + sign*random.gauss(mu,sigma)
            if(data[i]<=0):
                data[i]=0
        
        # 提取特征
        wave_rate_all_temp.append(cal_wave_rate(data))
        data_aver_temp.append(average(data))
        data_median_temp.append(median(data))
        SampEn_all_temp.append(SampEn(data,2,np.std(data)))

        a = list(data)
        feature_fft = Feature_fft(a).fft_all()
        fft_shape_mean_temp.append(feature_fft[0])

        fft_mean_temp.append(feature_fft[1])

    wave_rate_all[index_col] =  wave_rate_all_temp
    data_aver[index_col] = data_aver_temp
    data_median[index_col] = data_median_temp
    SampEn_all[index_col] = SampEn_all_temp
    fft_shape_mean[index_col] = fft_shape_mean_temp
    fft_mean[index_col] = fft_mean_temp

#%%
Feature_results = {
                   'release_num': list(range(1,Sample_amount+1)),
                   'wave_rate1':wave_rate_all[0],'aver1':data_aver[0],'median1':data_median[0],'fft_shape_mean1':fft_shape_mean[0],'fft_mean1':fft_mean[0],'SamEn1':SampEn_all[0],
                   'wave_rate2':wave_rate_all[1],'aver2':data_aver[1],'median2':data_median[1],'fft_shape_mean2':fft_shape_mean[1],'fft_mean2':fft_mean[1],'SamEn2':SampEn_all[1],
                   'wave_rate3':wave_rate_all[2],'aver3':data_aver[2],'median3':data_median[2],'fft_shape_mean3':fft_shape_mean[2],'fft_mean3':fft_mean[2],'SamEn3':SampEn_all[2],   
                   'wave_rate4':wave_rate_all[3],'aver4':data_aver[3],'median4':data_median[3],'fft_shape_mean4':fft_shape_mean[3],'fft_mean4':fft_mean[3],'SamEn4':SampEn_all[3],              
                   'x': list(release_loc[:,0]),
                   'y': list(release_loc[:,1])
                  }
Feature_results = pd.DataFrame(Feature_results)
Feature_results.to_csv('./Training_datasets/train_data.csv',index=None)

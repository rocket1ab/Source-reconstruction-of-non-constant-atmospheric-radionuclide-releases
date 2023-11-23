import numpy as np
import pandas as pd
import scipy.io
from sklearn import preprocessing

##################建立feature_fft这个类####################

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



#####        (3)构造波动率指标        #####
def cal_wave_rate(data):
    # 先对数据进行标准归一化
    data = data.reshape(-1,1)
    min_max_scaler = preprocessing.MinMaxScaler()
    data_minmax = min_max_scaler.fit_transform(data)
    #print(data_minmax)
    wave_rate = np.percentile(data_minmax,90)-np.percentile(data_minmax,10)
    return wave_rate


#####        (5)构造平均剂量率指标  #####
def average(data):
    data_aver = np.mean(data)
    return data_aver


#####        (7)构造中位数剂量率指标  #####
def median(data):
    data_median = np.median(data)
    return data_median

#####        (8)构造样本熵指标     ####
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

window_size = 5
#####      (9)对数据进行滑动窗口处理   #####
def rolling(data):
    data = data.rolling(window_size).sum()
    return data

window = window_size-1
wave_rate_all = []
data_aver = []
data_median = []
SampEn_all = []
#fft的一些频域指标
fft_shape_mean = []
fft_mean = []

########### Read data file ###########
source_data = './datafile.mat'
data_info = scipy.io.loadmat(source_data)
obs_data = data_info['obs_data']
Sta_num = obs_data.shape[1]
Scaling_factor = data_info['Scaling_factor']

df_all = pd.DataFrame(obs_data)
df = df_all #保存数据到df
df_all = rolling(df_all)
df = Scaling_factor*np.array(df[0:window]) #取出前3行数据用于补充后面的NAN值
data_all = df_all.values * Scaling_factor
data_all = (1/window_size)*data_all[window:]
data_all = np.vstack((df,data_all))

start_time = 11
end_time = 40

for index_col in range(Sta_num):

    data = data_all[:,index_col]
    data = np.array(data)
    data = data[start_time:end_time]
    
    #Hurst_exponent_all.append(hurst(data))
    wave_rate_all.append(cal_wave_rate(data))
    data_aver.append(average(data))
    data_median.append(median(data))
    SampEn_all.append(SampEn(data,2,np.std(data)))
    
    a = list(data)
    feature_fft = Feature_fft(a).fft_all()
    fft_shape_mean.append(feature_fft[0])
    fft_mean.append(feature_fft[1])

Feature_results = {
                   'release_num': 1,
                   'wave_rate1':wave_rate_all[0],'aver1':data_aver[0],'median1':data_median[0],'fft_shape_mean1':fft_shape_mean[0],'fft_mean1':fft_mean[0],'SamEn1':SampEn_all[0],
                   'wave_rate2':wave_rate_all[1],'aver2':data_aver[1],'median2':data_median[1],'fft_shape_mean2':fft_shape_mean[1],'fft_mean2':fft_mean[1],'SamEn2':SampEn_all[1],
                   'wave_rate3':wave_rate_all[2],'aver3':data_aver[2],'median3':data_median[2],'fft_shape_mean3':fft_shape_mean[2],'fft_mean3':fft_mean[2],'SamEn3':SampEn_all[2],   
                   'wave_rate4':wave_rate_all[3],'aver4':data_aver[3],'median4':data_median[3],'fft_shape_mean4':fft_shape_mean[3],'fft_mean4':fft_mean[3],'SamEn4':SampEn_all[3],              
                  }
Feature_results = pd.DataFrame(Feature_results,index=[0])
Feature_results.to_csv('./Testing_datasets/test_data.csv',index=None)



    
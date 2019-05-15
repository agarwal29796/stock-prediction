
# coding: utf-8

# ##  Assignment 10 : 
# Submitted by : 2018201051<br>

# In[1]:


import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
import os
import numpy as np
import tensorflow as tf 
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM,Dense


# In[2]:


df = pd.read_csv('GoogleStocks.csv')
df = df.sort_values('date')
df.head()


# In[3]:


df = df.drop(df.index[0])
df[["close", "volume" , "open" , "high" , "low"]] = df[["close", "volume" , "open" , "high" , "low"]].apply(pd.to_numeric)
df['avg'] = (df['low'] + df['high'])/2
df = df.reset_index(drop = True)
print(df.shape)
df.head()


# In[4]:


plt.figure(figsize = (18,9))
plt.plot(range(df.shape[0]),(df['avg']))
# plt.plot(range(df.shape[0]),(df['low']) , color =  'red')
# plt.plot(range(df.shape[0]),(df['high']) , color = 'green')
# plt.plot(range(df.shape[0]),(df['open']) , color = 'blue')
# plt.plot(range(df.shape[0]),(df['close']), color = 'black')

plt.xticks(range(0,df.shape[0],50),df['date'].loc[::50],rotation=45)
plt.xlabel('Date',fontsize=18)
plt.ylabel('Mid Price',fontsize=18)
plt.show()

plt.plot(range(df.shape[0]),(df['volume']))
plt.xticks(range(0,df.shape[0],50),df['date'].loc[::50],rotation=45)
plt.xlabel('date',fontsize=18)
plt.ylabel('volume',fontsize=18)
plt.show()


# In[4]:


train_df = df.iloc[: 550, [2,6]]
val_df = df.iloc[550: , [2,6]]
print(train_df.shape)
print(val_df.shape)
train_df.head()


# Due to the observation you made earlier, that is, different time periods of data have different value ranges, you normalize the data by splitting the full series into windows. If you don't do this, the earlier data will be close to 0 and will not add much value to the learning process. Here you choose a window size of 125.
# 
# 

# In[5]:


# Train the Scaler with training data and smooth data
train_data = train_df.as_matrix()
scaler = MinMaxScaler()
smoothing_window_size = 125
for i in range(0,train_data.shape[0],smoothing_window_size):
    scaler.fit(train_data[i:i+smoothing_window_size,:])
    train_data[i:i+smoothing_window_size,:] = scaler.transform(train_data[i:i+smoothing_window_size,:])


# In[7]:


print(train_data)


# In[6]:


val_data = val_df.as_matrix() 
val_data = scaler.transform(val_data)


# ###  Standard Averaging : 

# $$ x_{t+1} = 1/N \sum_{i = t-N}^{t} x_i $$

# In other words, you say the prediction at t+1 is the average value of all the stock prices you observed within a window of t to t−N.

# In[9]:


# Now perform exponential moving average smoothing
# So the data will have a smoother curve than the original ragged data
EMA = 0.0
gamma = 0.1
for ti in range(550):
    EMA = gamma*train_data[ti,1] + (1-gamma)*EMA
    train_data[ti,1] = EMA

# Used for visualization and test purposes
all_data = np.concatenate([train_data,val_data],axis=0)


# In[10]:


window_size = 20
N = len(train_data)
print(N)
std_avg_predictions = []
std_avg_x = []
mse_errors = []

for pred_idx in range(window_size,N):
    if pred_idx >= N:
        date = dt.datetime.strptime(k, '%Y-%m-%d').date() + dt.timedelta(days=1)
    else:
        date = df.loc[pred_idx,'date']

    std_avg_predictions.append(np.mean(train_data[pred_idx-window_size:pred_idx]))
    mse_errors.append((std_avg_predictions[-1]-train_data[pred_idx])**2)
    std_avg_x.append(date)

print('MSE error for standard averaging: %.5f'%(0.5*np.mean(mse_errors)))


# In[11]:


plt.figure(figsize = (18,9))
plt.plot(range(df.shape[0]), all_data[:,1],color='b',label='True')
plt.plot(range(window_size,N),std_avg_predictions,color='orange',label='Prediction')
#plt.xticks(range(0,df.shape[0],50),df['Date'].loc[::50],rotation=45)
plt.xlabel('Date')
plt.ylabel('avg Price')
plt.legend(fontsize=18)
plt.show()


# ###  Eponential  Moving Average : 

# In the exponential moving average method, you calculate$ x_{t+1} $ as,
# 
# $$ x_{t+1} = EMA_t = γ × EMA_{t-1} + (1-γ) x_t $$ where$ EMA_0$ = 0 and EMA is the exponential moving average value you maintain over time.
# 

# In[12]:


window_size = 20
N = len(train_data)
run_avg_predictions = []
run_avg_x = []

mse_errors = []

running_mean = 0.0
run_avg_predictions.append(running_mean)

decay = 0.5

for pred_idx in range(1,N):

    running_mean = running_mean*decay + (1.0-decay)*train_data[pred_idx-1 ,1]
    run_avg_predictions.append(running_mean)
    mse_errors.append((run_avg_predictions[-1]-train_data[pred_idx, 1])**2)
    run_avg_x.append(date)
print('MSE error for EMA averaging: %.5f'%(0.5*np.mean(mse_errors)))


# In[13]:


plt.figure(figsize = (18,9))
plt.plot(range(df.shape[0]),all_data[:,1],color='b',label='True')
plt.plot(range(0,N),run_avg_predictions,color='orange', label='Prediction')
#plt.xticks(range(0,df.shape[0],50),df['Date'].loc[::50],rotation=45)
plt.xlabel('Date')
plt.ylabel('Mid Price')
plt.legend(fontsize=18)
plt.show()


# ####  Where EMA goes wrong ?

# try to make predictions in windows (say you predict the next 2 days window, instead of just the next day)<br>
# some solutions : Momentum based algorithm(considers volume field)
# 

# ## Recurrent Neural Network based prediction  :  

# In[8]:


def normalize_data(data):
    min_max_scaler = MinMaxScaler()
    data[: , :] = min_max_scaler.fit_transform(data[: , :])
    return data  


# In[9]:


data = df.iloc[: , [2,6]]
data = data.as_matrix()
data = normalize_data(data)
print(data.shape)
time_step = 20


# In[6]:


class rnn(object) :
    def __init__(self, layers , nodes , time_step , epochs = 100):
        self.layers = layers
        self.nodes = nodes
        self.time_step = time_step
        self.epochs = epochs
        self.history = None
        
    def data_process(self,  data):
        X,Y = [],[]
        for i in range(len(data)-self.time_step-1):
            X.append(data[i:(i + self.time_step),:])
            Y.append(data[(i+self.time_step),-1])
        return np.array(X),np.array(Y)
        
    def fit_and_predict(self , data):
        X,y = self.data_process(data)
        X_train,X_test = X[:int(X.shape[0]*0.80) , :],X[int(X.shape[0]*0.80):, :]
        y_train,y_test = y[:int(y.shape[0]*0.80)],y[int(y.shape[0]*0.80):]
        
        model = Sequential()
        model.add(LSTM(self.nodes, return_sequences=True,input_shape=(self.time_step, X_train.shape[-1])))
        
        for i in range(self.layers-2) :
            model.add(LSTM(self.nodes, return_sequences=True))
        model.add(LSTM(self.nodes))
        model.add(Dense(1 , activation =  'relu'))
        model.compile(optimizer='adam',loss='mse')

        self.history = model.fit(X_train,y_train,epochs=self.epochs,validation_data=(X_test,y_test),shuffle=False)
        
        Xt = model.predict(X_test)
        plt.plot(y_test.reshape(-1,1))
        plt.plot(Xt)
        
        #input data shape: (batch_size, timesteps, data_dim)

        


# ### 1. hidden layers = 2 , cells  = 30  , time_steps = 20

# In[48]:


rnn1 = rnn(2, 30, 20,50)
rnn1.fit_and_predict(data)


# In[49]:


plt.plot(rnn1.history.history['loss'] , color = 'green')
plt.plot(rnn1.history.history['val_loss']  , color = 'red')
plt.legend(['training loss' , 'val loss'])


# ### 2. hidden layers = 2 , cells  = 30  , time_steps = 50

# In[50]:


rnn2 = rnn(2, 30, 50, 50)
rnn2.fit_and_predict(data)


# ### 3. hidden layers = 2 , cells  = 30  , time_steps = 75

# In[52]:


rnn3 = rnn(2, 30, 75 , 50)
rnn3.fit_and_predict(data)


# ### 4. hidden layers = 2 , cells  = 50  , time_steps = 20

# In[53]:


rnn4 = rnn(2, 50, 20 , 50)
rnn4.fit_and_predict(data)


# ### 5. hidden layers = 2 , cells  = 50  , time_steps = 50

# In[58]:


rnn5 = rnn(2, 50, 50 , 50)
rnn5.fit_and_predict(data)


# ### 6. hidden layers = 2 , cells  = 50  , time_steps = 75

# In[55]:


rnn6 = rnn(2, 50, 75 , 50)
rnn6.fit_and_predict(data)


# ### 7. hidden layers = 2 , cells  = 80  , time_steps = 20

# In[60]:


rnn7 = rnn(2, 80, 20 , 50)
rnn7.fit_and_predict(data)


# ### 8. hidden layers = 2 , cells  = 80  , time_steps = 50

# In[61]:


rnn8 = rnn(2, 80, 50 , 50)
rnn8.fit_and_predict(data)


# ### 9. hidden layers = 2 , cells  = 80  , time_steps = 75

# In[62]:


rnn9 = rnn(2, 80, 75 , 50)
rnn9.fit_and_predict(data)


# ### 10 . hidden layers = 3 , cells  = 30  , time_steps = 20
# 

# In[63]:


rnn10 = rnn(3, 30, 20,50)
rnn10.fit_and_predict(data)


# ### 11. hidden layers = 3 , cells  = 30  , time_steps = 50
# 

# In[65]:


rnn11 = rnn(3, 30,50 ,50)
rnn11.fit_and_predict(data)


# ### 12 . hidden layers = 3 , cells  = 30  , time_steps = 75

# In[66]:


rnn12 = rnn(3, 30 ,  75 ,50)
rnn12.fit_and_predict(data)


# ### 13 . hidden layers = 3 , cells  = 50  , time_steps = 20
# 

# In[69]:


rnn13 = rnn(3, 50 , 20 ,50)
rnn13.fit_and_predict(data)


# ### 14 . hidden layers = 3 , cells  = 50  , time_steps = 50
# 

# In[70]:


rnn14 = rnn(3, 50 , 50 ,50)
rnn14.fit_and_predict(data)


# ### 15 . hidden layers = 3 , cells  = 50  , time_steps = 75
# 

# In[71]:


rnn15 = rnn(3, 50 , 75 ,50)
rnn15.fit_and_predict(data)


# ### 16 . hidden layers = 3 , cells  = 80  , time_steps = 20
# 

# In[73]:


rnn16 = rnn(3 , 80 , 20 ,50)
rnn16.fit_and_predict(data)


# ### 17 . hidden layers = 3 , cells  = 80  , time_steps = 50
# 

# In[74]:


rnn17 = rnn(3 , 80 , 50 ,50)
rnn17.fit_and_predict(data)


# ### 18 . hidden layers = 3 , cells  = 80  , time_steps = 75
# 

# In[7]:


rnn18 = rnn(3 , 80 , 75 ,50)
rnn18.fit_and_predict(data)


# ###  HMM based prediction : 

# In[116]:


from hmmlearn.hmm import GaussianHMM
from hmmlearn.hmm import MultinomialHMM
from matplotlib import cm, pyplot as plt
from matplotlib.dates import YearLocator, MonthLocator
import math
import warnings
warnings.filterwarnings('ignore')


# ##  Question 2 : 

# In[120]:



states = ('E', '5' , 'I' )

observations = ('A' , 'C' , 'G' , 'T')

start_probability = {'E': 1, '5': 0 ,'I' : 0}
end_probability = {'E': 0, '5': 0 ,'I' : 0.1}

transition_probability = {
  'E' : {'E': 0.9, '5': 0.1 , 'I' : 0},
  '5' : {'E': 0, '5': 0 , 'I' :1 },
  'I' : {'E': 0, '5': 0 , 'I' : 0.9}
   }

emission_probability = {
  'E' : {'A': 0.25, 'C': 0.25, 'G': 0.25,  'T' : 0.25},
  '5' : {'A': 0.05, 'C': 0, 'G': 0.95,  'T' :0 },
  'I' : {'A': 0.4, 'C': 0.1, 'G': 0.1,  'T' :0.4}
   }

model = MultinomialHMM(n_components=3)
model.startprob_ = np.array([1,0,0])
model.endprob_ = np.array([0,0,0.1])

model.transmat_ = np.array([[0.9, 0.1, 0],
                           [0, 0, 1] ,
                           [0, 0, 1]])
model.emissionprob_ = np.array([[0.25, 0.25, 0.25, 0.25],
                               [0.05, 0, 0.95,  0] , 
                              [0.4, 0.1 ,0.1 , 0.4]])


# In[121]:


#"CTTCATGTGAAAGCAGACGTAAGTCA" A = 0 , C = 1 , G = 2 , T = 3
sequence = [1,3,3,1,0,3,2,3,2,0,0,0,2,1,0,2,0,1,2,3,0,0,2,3,1,0]

logprob, seq = model.decode(np.array([sequence]).transpose())
print(logprob)
print(seq)
# E = 0 ,  5 = 1 , I = 2
print("following sequence correspond to :")
print("EEEEEEEEEEEEEEEEEE5IIIIIII")


# ##  Question 1.2  : HMM 
# #### based on the following paper : 
# https://editorialexpress.com/cgi-bin/conference/download.cgi?db_name=SILC2016&paper_id=38

# In[90]:


# Calculating Mean Absolute Percentage Error of predictions
def calc_mape(predicted_data, true_data):
    return np.divide(np.sum(np.divide(np.absolute(predicted_data - true_data), true_data), 0), true_data.shape[0])


# In[145]:


class HMM(object):
    def __init__(self , states , time_step):
        self.states = states
        self.time_step = time_step
        self.num_calib = 100

    def fit_and_predict(self, dataset):
        
        predicted_stock_data = np.empty([0,dataset.shape[1]])

        for idx in range(self.num_calib):
            train_dataset = dataset[idx : idx + self.time_step:]
            test_data = dataset[idx + self.time_step,:] 
            if idx == 0:
#                 n_components=4, covariance_type="diag", n_iter=100
                model = GaussianHMM(n_components=self.states, covariance_type='full',verbose = True,  n_iter=100, init_params='stmc')
            else:
                # Retune the model by using the HMM paramters from the previous iterations as the prior
                model = GaussianHMM(n_components=self.states, covariance_type='full',verbose = True, n_iter=100, init_params='')
                model.transmat_ = transmat_retune_prior 
                model.startprob_ = startprob_retune_prior
                model.means_ = means_retune_prior
                model.covars_ = covars_retune_prior

            model.fit(train_dataset)
            
            print(model.transmat_)
            
            transmat_retune_prior = model.transmat_
            startprob_retune_prior = model.startprob_
            means_retune_prior = model.means_
            covars_retune_prior = model.covars_

            if model.monitor_.iter == 100:
                print('Increase number of iterations')
                sys.exit(1)

            iters = 1;
            past_likelihood = []
            K = self.time_step
            curr_likelihood = model.score(train_dataset[0:K , :])
            num_examples = train_dataset.shape[0]
            
            iters = num_examples
            
            while iters > 0 :
                past_likelihood = np.append(past_likelihood, model.score(train_dataset[0:iters, :]))
                iters = iters - 1

            
            likelihood_diff_idx = np.argmin(np.absolute(past_likelihood - curr_likelihood))
            
            predicted_change = train_dataset[likelihood_diff_idx,:] - train_dataset[likelihood_diff_idx + 1,:]
            
            predicted_stock_data = np.vstack((predicted_stock_data, dataset[idx + self.time_step-1,:] + predicted_change))
            
            mape = calc_mape(predicted_stock_data, np.flipud(dataset[range(100),:]))
            print('MAPE is ',mape)
            print(predicted_stock_data)
#             plt.plot(range(100), predicted_stock_data);
#             plt.title('Predicted stock prices')
#             plt.legend(iter(hdl_p), ('Close','Open','High','Low'))
#             plt.xlabel('Time steps')
#             plt.ylabel('Price')
#             plt.figure()
#             plt.plot(range(100),dataset[range(100),:])
#             plt.title('Actual stock prices')
#             plt.legend(iter(hdl_p), ('Close','Open','High','Low'))
#             plt.xlabel('Time steps')
#             plt.ylabel('Price')

        


# In[146]:


data_new = data[:,0]
hmm1 = HMM(4,20)
hmm1.fit_and_predict(data_new.reshape(-1,  1))


# ### Question 1.3 

# Hidden Markov Models (HMMs) are much simpler than Recurrent Neural Networks (RNNs), and rely on strong assumptions which may not always be true. However, if the assumptions are true then you may see better performance from an HMM since it is less finicky to get working. On the other hand, even if the assumptions roughly hold true, if you have gobs of data than an RNN may still perform better since the extra complexity can take better advantage of the information in your data.<br>

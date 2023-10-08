#!/usr/bin/env python
# coding: utf-8

# # BHARAT INTERNSHIP
# 
# # NAME-VALLELA MANASA
# 
# # TASK 1-STOCK PREDICTION
# 
# In this we will use the Nestle India -Historical Stock Price Dataset for STOCK PREDICTION
# 
#   
# 

# # Importing Libraries

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense 


# # Loading data from a CSV file

# In[3]:


df=pd.read_csv(r"C:\Users\Vallela Manasa\Downloads\nestle.csv")


# In[4]:


df.head()


# In[5]:


df.tail()


# # SHAPE OF THE DATA

# In[6]:


df.shape


# # Gathering information about data

# In[7]:


df.info()


# In[8]:


df.describe()


# In[9]:


df.dtypes


# # cleaning the data

# In[10]:


df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values(by='Date', ascending=True)
df = df[['Date', 'Close Price']]


# In[11]:


df.columns


# In[12]:


df.head()


# In[13]:


df.tail()


# # Normalize the Close prices

# In[14]:


scaler = MinMaxScaler()
df['Close Price'] = scaler.fit_transform(df['Close Price'].values.reshape(-1, 1))


# # split the data into train and test sets

# In[15]:


train_size = int(len(df) * 0.8)
train_data = df.iloc[:train_size]
test_data = df.iloc[train_size:]
print(train_data)
print(test_data)


# # create sequences and labels for training and testing

# In[ ]:





# In[16]:


# Function to create sequences and labels
def create_sequences(df, seq_length):
    sequences, labels = [], []
    for i in range(len(df) - seq_length):
        seq = df['Close Price'].values[i:i+seq_length]
        label = df['Close Price'].values[i+seq_length]
        sequences.append(seq)
        labels.append(label)
    return np.array(sequences), np.array(labels)

# Define sequence length
seq_length = 10 


X_train, y_train = create_sequences(train_data, seq_length)
X_test, y_test = create_sequences(test_data, seq_length)
print(X_train,y_train,X_test,y_test)


# # Reshape the data for LSTM

# In[17]:


X_train = X_train.reshape(X_train.shape[0], seq_length, 1)
X_test = X_test.reshape(X_test.shape[0], seq_length, 1)

print(X_train.shape)
print(X_test.shape)


# # Build and train the LSTM model

# In[18]:


model = Sequential()
model.add(LSTM(50, input_shape=(seq_length, 1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=50, batch_size=32)


# # Make predictions

# In[19]:


y_pred = model.predict(X_test)
y_pred = scaler.inverse_transform(y_pred)
y_test = scaler.inverse_transform(y_test.reshape(-1, 1))


# # Calculating RMSE

# 
# 

# In[21]:


rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Root Mean Squared Error: {rmse}")


# # Plot the true vs predicted prices

# In[22]:


plt.figure(figsize=(12, 6))
plt.plot(y_test, label='True Price', color='green')
plt.plot(y_pred, label='Predicted Price', color='red')
plt.legend()
plt.show()


# In[ ]:





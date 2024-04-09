# pip install numpy pandas matplotlib keras scikit-learn tensorflow
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras import optimizers
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
import warnings
plt.style.use('ggplot')

# Setting up an early stop
earlystop = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=40,  verbose=0, mode='min')
callbacks_list = [earlystop]

def data_preprocess(path):
    '''
    read data from path, and preprocess it
    calculate stock price supposing price is 1 at the beginning
    '''
    df = pd.read_csv(path, header=None)
    df.fillna(method="ffill", inplace=True)
    # rename columns
    df.columns = ['label', 'open', 'high', 'low', 'close']
    # calculate price from return
    df['close_price'] = (1+df['close']).cumprod()
    df['high_price'] = (1+df['high'])*df['close_price'].shift(1)
    df['low_price'] = (1+df['low'])*df['close_price'].shift(1)
    df['open_price'] = (1+df['open'])*df['close_price'].shift(1)
    df.fillna(1, inplace=True)
    df['label2'] = df['close_price'].shift(-1)
    # ffill for label2
    df.fillna(method="ffill", inplace=True)
    return df

#Build and train the model
def fit_model_univariate(train,val,timesteps,hl,lr,batch,epochs):
    '''
    The first method, Univariate LSTM, only use close price as input
    '''
    X_train = []
    Y_train = []
    X_val = []
    Y_val = []
    # Loop for training data
    for i in range(timesteps,train.shape[0]):
        X_train.append(train[i-timesteps:i])
        Y_train.append(train[i])
    X_train,Y_train = np.array(X_train),np.array(Y_train)
  
    # Loop for val data
    for i in range(timesteps,val.shape[0]):
        X_val.append(val[i-timesteps:i])
        Y_val.append(val[i])
    X_val,Y_val = np.array(X_val),np.array(Y_val)
  
    # Adding Layers to the model
    model = Sequential()
    model.add(LSTM(1,input_shape = (X_train.shape[1],1),return_sequences = True))
    for i in range(len(hl)-1):        
        model.add(LSTM(hl[i],return_sequences = True))
        model.add(Dropout(0.2))
    model.add(LSTM(hl[-1]))
    model.add(Dense(1))
    model.compile(optimizer = optimizers.Adam(learning_rate=lr), loss = 'mean_squared_error')
  
    # Training the data
    history = model.fit(X_train,Y_train,epochs = epochs, batch_size = batch, validation_data = (X_val, Y_val),
                        verbose = 0, shuffle = False, callbacks=callbacks_list)
    # model.reset_states()
    return model, history.history['loss'], history.history['val_loss']

#Build and train the model
def fit_model_multivariate(train,val,timesteps,hl,lr,batch,epochs):
    '''
    The second method, Multivariate LSTM, use all features as input
    '''
    X_train = []
    Y_train = []
    X_val = []
    Y_val = []
  
    # Loop for training data
    for i in range(timesteps,train.shape[0]+1):
        X_train.append(train[i-timesteps:i, 0:4])
        Y_train.append(train[i-1][4])
    X_train,Y_train = np.array(X_train),np.array(Y_train)
  
    # Loop for val data
    for i in range(timesteps,val.shape[0]+1):
        X_val.append(val[i-timesteps:i,0:4])
        Y_val.append(val[i-1][4])
    X_val,Y_val = np.array(X_val),np.array(Y_val)
    
    # Adding Layers to the model
    model = Sequential()
    model.add(LSTM(X_train.shape[2],input_shape = (X_train.shape[1],X_train.shape[2]),return_sequences = True))
    for i in range(len(hl)-1):        
        model.add(LSTM(hl[i],return_sequences = True))
        # model.add(Dropout(0.1))
    model.add(LSTM(hl[-1]))
    model.add(Dense(1))
    model.compile(optimizer = optimizers.Adam(learning_rate = lr), loss = 'mean_squared_error')
  
    # Training the data
    history = model.fit(X_train,Y_train,epochs = epochs,batch_size = batch,validation_data = (X_val, Y_val),verbose = 0,
                        shuffle = False, callbacks=callbacks_list)
    # model.reset_states()
    return model, history.history['loss'], history.history['val_loss']


# Evaluating the model
def evaluate_model_univariate(model,test,timesteps):
    '''
    Evaluate the model
    main use here is to calculate the out of sample r2 score
    '''
    X_test = []
    Y_test = []

    # Loop for testing data
    for i in range(timesteps,test.shape[0]):
        X_test.append(test[i-timesteps:i])
        Y_test.append(test[i])
    X_test,Y_test = np.array(X_test),np.array(Y_test)
  
    Y_hat = model.predict(X_test, verbose=0)
    r2 = r2_score(Y_test,Y_hat)
    return r2

def evaluate_model_multivariate(model,test,timesteps):
    '''
    Evaluate the model
    main use here is to calculate the out of sample r2 score
    '''
    X_test = []
    Y_test = []

    # Loop for testing data
    for i in range(timesteps,test.shape[0]+1):
        X_test.append(test[i-timesteps:i, 0:4])
        Y_test.append(test[i-1][4])
    X_test,Y_test = np.array(X_test),np.array(Y_test)
  
    Y_hat = model.predict(X_test, verbose=0)
    r2 = r2_score(Y_test,Y_hat)
    return r2
  


def generate_output_univariate(model, all_data, timesteps, sc):
    '''
    Give output of Price for Univariate LSTM of the whole dataset
    '''
    X_all = []
    Y_all = []
    for i in range(timesteps,all_data.shape[0]):
        X_all.append(all_data[i-timesteps:i])
        Y_all.append(all_data[i])
    X_all,Y_all = np.array(X_all),np.array(Y_all)
    Y_hat = model.predict(X_all, verbose=0)
    Y_hat = np.array(Y_hat)
    Y_hat = sc.inverse_transform(Y_hat)
    Y_all = sc.inverse_transform(Y_all)
    c = np.concatenate((Y_all[:timesteps].reshape(-1), Y_hat.reshape(-1)))
    return c

# Evaluating the model
def generate_output_multivariate(model,all_data,timesteps, sc):
    '''
    Give output of Price for Multivariate LSTM of the whole dataset
    '''
    X_test = []
    Y_test = []

    # Loop for testing data
    for i in range(timesteps,all_data.shape[0]+1):
        X_test.append(all_data[i-timesteps:i, 0:4])
        Y_test.append(all_data[i-1][4])
    X_test,Y_test = np.array(X_test),np.array(Y_test)


    Y_hat = model.predict(X_test, verbose=0)
    # create a new array, first timesteps is all_data[:timesteps, 4], then Y_hat
    c = np.concatenate((all_data[:timesteps-1, 3], Y_hat.reshape(-1)))

    Y_hat_tmp= np.zeros_like(all_data)
    Y_hat_tmp[:len(all_data),4] = c.reshape(-1)
    Y_hat_tmp = sc.inverse_transform(Y_hat_tmp)[:,4]

    return Y_hat_tmp
  

def split_data(series, train_percent, val_percent, test_percent):
    '''
    Split data into train, val, test
    '''
    train_data = series[:int(train_percent*len(series))]
    val_data = series[int(train_percent*len(series)):int((train_percent+val_percent)*len(series))]
    test_data = series[int((train_percent+val_percent)*len(series)):]
    return train_data, val_data, test_data

def transform_data(scalar, all, train, val, test):
    '''
    Normalize data
    '''
    train = scalar.fit_transform(train)
    val = scalar.transform(val)
    test = scalar.transform(test)
    all = scalar.transform(all)
    return train, val, test, all

def run_univariate(df, timesteps, hl, lr, batch_size, num_epochs, train_percent, val_percent, test_percent):
    '''
    Run univariate LSTM model
    '''
    # Extracting the series
    series = df['close_price']
    series2 = series.values.reshape(-1,1)
    train_data, val_data, test_data = split_data(series2, train_percent, val_percent, test_percent)
    # Normalization
    sc_uni = MinMaxScaler()
    train, val, test, all_data = transform_data(sc_uni, series2, train_data, val_data, test_data)
    model_uni,train_error,val_error = fit_model_univariate(train,val,timesteps,hl,lr,batch_size,num_epochs)
    
    r2_uni = evaluate_model_univariate(model_uni, test, timesteps)
    Y_hat = generate_output_univariate(model_uni, all_data,timesteps, sc_uni)
    del model_uni
    return Y_hat, r2_uni

def run_multivariate(df, timesteps, hl, lr, batch_size, num_epochs, train_percent, val_percent, test_percent):
    '''
    Run multivariate LSTM model
    '''
    # Extracting the series
    series = df[['open_price', 'high_price', 'low_price', 'close_price', 'label2']]
    series2 = series.values
    # split data for training
    train_data, val_data, test_data = split_data(series2, train_percent, val_percent, test_percent)
    # Normalization
    sc_multi = MinMaxScaler()
    train, val, test, all_data = transform_data(sc_multi, series2, train_data, val_data, test_data)
    model,train_error,val_error = fit_model_multivariate(train,val,timesteps,hl,lr,batch_size,num_epochs)
    r2_multi = evaluate_model_multivariate(model, test, timesteps)
    Y_hat = generate_output_multivariate(model, all_data,timesteps, sc_multi)
    del model
    return Y_hat, r2_multi

def convert_price_to_return(Y_hat, r0):
    '''
    convert price to return
    '''
    df2 = pd.DataFrame({'Predicted': Y_hat.flatten()})
    df2['predicted_return'] = df2['Predicted'].pct_change()
    df2.fillna(r0, inplace=True)
    return df2


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    # read from args
    if len(sys.argv) != 2:
        # print("Usage: python final1.py <input_file>")
        sys.exit(1)
    file_path = sys.argv[1]
    # file_path = "dataset-20230501.csv"
    # read data
    data = data_preprocess(file_path)

    # set hyperparameters
    timesteps = 40
    hl = [40,35] # hidden layers
    lr = 0.0005
    batch_size = 64
    num_epochs = 150
    train_percent = 0.7
    val_percent = 0.15
    test_percent = 0.15

    Y_uni, r2_uni = run_univariate(data, timesteps, hl, lr, batch_size, num_epochs, train_percent, val_percent, test_percent)
    Y_multi, r2_multi = run_multivariate(data, timesteps, hl, lr, batch_size, num_epochs, train_percent, val_percent, test_percent)
    if (r2_multi > r2_uni):
        Y = Y_multi
    else:
        Y = Y_uni
    returns_df = convert_price_to_return(Y, data['close'][0])

    # output
    for i in range(len(returns_df)):
        print(i+1, ",", returns_df['predicted_return'][i])
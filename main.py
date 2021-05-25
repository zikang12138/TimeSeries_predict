
from re import X
import numpy as np
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,SimpleRNN,Conv1D,MaxPool1D,Flatten
import matplotlib.pyplot as plt
import time
from tensorflow.keras.models import load_model
import pandas as pd


class Time_Predict:
    '''
    用于时间序列预测的一个类
    '''
    def __init__(self,data_name,seq_len,label_len,teach_forecast=False,n_features=1):
        #初始化
        #data_name:原始数据
        #seq_len 输入长度
        #label——len 预测步长
        #teach_forecast 是否使用时间滑窗
        self.data_name=data_name
        self.seq_len=seq_len
        self.label_len=label_len
        self.teach_forecast=teach_forecast
        self.n_features=n_features

    def load_data(self):
        f = open(self.data_name, 'r').read()  # 读取文件中的数据
        data2 = f.split('\n')  # split() 方法用于把一个字符串分割成字符串数组，这里就是换行分割
        data = []
        for n in data2:
            if (n == ''):
                continue
            data.append(float(n))
        sequence_lenghth = self.seq_len + self.label_len  # #得到长度为seq_len+1的向量，最后一个作为label
        result = []
        for index in range(len(data) - sequence_lenghth):
            result.append(data[index: index + sequence_lenghth])  # 制作数据集，从data里面分割数据
        result = np.array(result)  # shape (4121,51) 4121代表行，51是seq_len+1
        row = round(0.9 * result.shape[0])  # round() 方法返回浮点数x的四舍五入值
        train = result[:int(row), :]  # 取前90%

        x_train = train[:, :-self.label_len]  # 取前50列，作为训练数据
        y_train = train[:, -self.label_len:]  # 取最后一列作为标签
        x_test = result[int(row):, :-self.label_len]  # 取后10% 的前50列作为测试集
        y_test = result[int(row):, -self.label_len:]  # 取后10% 的最后一列作为标签
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))  # 最后一个维度1代表一个数据的维度
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
        return [x_train, y_train, x_test, y_test]

    def rnn(self,x_train, y_train, model_save,  ep):#model_save 模型文件保存名 ep循环次数
        model = Sequential()
        model.add(SimpleRNN(100, return_sequences=False, input_shape=(self.seq_len, self.n_features)))
        model.add(Dense(50,activation='tanh'))
        model.add(Dense(50,activation='tanh'))
        model.add(Dense(50,activation='tanh'))
        if self.teach_forecast:
            model.add(Dense(1))
        else:
            model.add(Dense(self.label_len))
        model.add(Activation('linear'))
        model.compile(loss='mse', optimizer='adam')
        start = time.time()
        model.fit(x_train, y_train, batch_size=64, epochs=ep, validation_split=0.05)
        print('rnn compilation time : ', time.time() - start)
        model.save(model_save)


    def lstm(self,x_train, y_train, model_save, ep):
        model = Sequential()
        model.add(LSTM(100, return_sequences=False, input_shape=(self.seq_len, self.n_features)))
        model.add(Dropout(0.2))
        model.add(LSTM(100, return_sequences=False, input_shape=(self.seq_len, self.n_features)))
        model.add(Dropout(0.2))
        model.add(Dense(50))
        if self.teach_forecast:
            model.add(Dense(1))
        else:
            model.add(Dense(self.label_len))
        model.add(Activation('linear'))
        model.compile(loss='mse', optimizer='adam')
        start = time.time()
        model.fit(x_train, y_train, batch_size=64, epochs=ep, validation_split=0.05)
        print('compilation time : ', time.time() - start)
        model.save(model_save)     

    def cnn(self,x_train, y_train, model_save, ep):
        model = Sequential()
        model.add(Conv1D(filters=64, kernel_size=2, activation='tanh', input_shape=(self.seq_len, self.n_features)))
        model.add(MaxPool1D(pool_size=2))
        model.add(Flatten())
        model.add(Dense(50, activation='tanh'))
        if self.teach_forecast:
            model.add(Dense(1))
        else:
            model.add(Dense(self.label_len))
        model.compile(loss='mse', optimizer='adam')
        start = time.time()
        model.fit(x_train, y_train, batch_size=64, epochs=ep, validation_split=0.05)
        print('compilation time : ', time.time() - start)
        model.save(model_save)


    def plot_training(self,model_save,plot_save):
        history=load_model(model_save)
        plt.plot(history.history['loss'], label='loss')
        plt.plot(history.history['val_loss'], label='val_loss')
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('epoch')
        plt.legend()
        plt.savefig(plot_save)

    def predict_result(self,model_save, x_test):#model——save模型保存名 
        model=load_model(model_save)#读取模型
        if self.teach_forecast:
            y_hat=x_test
            predicteds=[]
            for i in range(self.label_len):
                m=y_hat[:,i:,:]#取数据的i列到最后
                predicted=model.predict(m)#预测一行
                predicteds.append(predicted)#加入到预测list中
                predicted_rever=np.reshape(predicted,(predicted.shape[1],predicted.shape[0]))#转置
                insert_post=y_hat.shape[1]
                y_hat_temp=np.insert(y_hat[:,:,0], insert_post, predicted_rever,axis=1)#插入到预测输入中
                y_hat=np.reshape(y_hat_temp, (y_hat_temp.shape[0],y_hat_temp.shape[1],1))#升维
                # y_hat.append(predicted)
            predictedss=np.array(predicteds)#
            predictedss=predictedss.transpose(1,0,2)
            predictedss=np.reshape(predictedss, (predictedss.shape[0],predictedss.shape[1]))
        else:
            predictedss = model.predict(x_test)  # 输入测试集的全部数据进行全部预测，
        #predicted = np.reshape(predicted, (predicted.size,))
        return predictedss

    def plot_results(self,predicted_data, true_data, save_name,picture_name):#save_name 图形保存名 picture name 图形标题
        fig = plt.figure(facecolor='white')
        ax = fig.add_subplot(111)
        ax.plot(true_data, label='True Data')
        plt.plot(predicted_data, label='Prediction')
        plt.title(picture_name)
        plt.legend()
        plt.savefig(save_name)

    def AVE(self,y_true, y_predict):
        m = y_true - y_predict
        ave=np.average(abs(m))
        return ave

    def evalute(self,predicted_data,y_test,plot_result_name,picture_name):
        n=len(y_test)
        m=abs(y_test-predicted_data)
        k=[]
        for j in range(n):
            k.append(sum(m[j]))
        i=k.index(min(k))
        self.plot_results(predicted_data[i], y_test[i], plot_result_name,picture_name)
        ave=self.AVE(y_test,predicted_data)
        print("loss: ")
        print(ave)



class multi_Time_Predict(Time_Predict):
    def load_data(self,forecast_num=0):
        dataset = pd.read_csv(self.data_name)
        x_1 = dataset['x1']
        y = dataset['y']
        x_1 = x_1.values
        y = y.values
        n=len(x_1)
        x_1 = x_1.reshape((n, 1))
        y = y.reshape((n, 1))
        sequences = np.hstack((x_1[forecast_num:,:], y[:n-forecast_num,:]))
        split=round(0.9*sequences.shape[0])
        X, Y = list(), list()
        for i in range(len(sequences)):
            # find the end of this pattern
            end_ix = i + self.seq_len
            out_end_ix = end_ix + self.label_len-1
            # check if we are beyond the dataset
            if out_end_ix > len(sequences):
                break
            # gather input and output parts of the pattern
            seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix-1:out_end_ix, -1]
            X.append(seq_x)
            Y.append(seq_y)
        X=np.array(X)
        Y=np.array(Y)
        Y=np.reshape(Y,(Y.shape[0],Y.shape[1],1))
        x_train , y_train = X[:split, :] , Y[:split, :]
        x_test , y_test = X[split:, :] , Y[split:, :]
        return x_train,y_train,x_test,y_test
    


'''
代码运行实例 单变量
'''
TP=Time_Predict(data_name='data/4class.csv',seq_len=100,label_len=10,teach_forecast=False,n_features=1)#定义一个time_predict类 
[xtrain,ytrain,xtest,ytest]=TP.load_data()#获取数据
TP.cnn(x_train=xtrain,y_train=ytrain,model_save='model/cnn.h5',ep=10)#cnn模型训练并生成训练文件 cnn.h5
tp=TP.predict_result(model_save='model/cnn.h5',x_test=xtest)#读取模型文件并生成预测值
TP.evalute(predicted_data=tp,y_test=ytest,plot_result_name='picture/cnn.png',picture_name='cnn')#评估模型， 生成预测曲线和实际曲线，图名为cnn 文件名为cnn.png

'''
代码运行实例 多变量
'''
MTP=multi_Time_Predict(data_name='data/mutil.csv',seq_len=100,label_len=10,teach_forecast=False,n_features=2)
[xtrain2,ytrain2,xtest2,ytest2]=MTP.load_data(forecast_num=0)
MTP.rnn(x_train=xtrain2,y_train=ytrain2,model_save='model/mutil_rnn_forecast10_300_100to10.h5',ep=300)
MTP.cnn(x_train=xtrain2,y_train=ytrain2,model_save='model/mutil_cnn_forecast10_300_100to10.h5',ep=300)
MTP.lstm(x_train=xtrain2,y_train=ytrain2,model_save='model/mutil_lstm_forecast10_300_100to10.h5',ep=300)

mtp1=MTP.predict_result(model_save='model/mutil_rnn_forecast10_300_100to10.h5',x_test=xtest2)
mtp2=MTP.predict_result(model_save='model/mutil_cnn_forecast10_300_100to10.h5',x_test=xtest2)
mtp3=MTP.predict_result(model_save='model/mutil_lstm_forecast10_300_100to10.h5',x_test=xtest2)

MTP.evalute(predicted_data=mtp1,y_test=np.reshape(ytest2,(ytest2.shape[0],ytest2.shape[1])),plot_result_name='picture/mutil_rnn_forecast10_300_100to10.png',picture_name='mutil_rnn_forecast10_300_100to10')
MTP.evalute(predicted_data=mtp2,y_test=np.reshape(ytest2,(ytest2.shape[0],ytest2.shape[1])),plot_result_name='picture/mutil_cnn_forecast10_300_100to10.png',picture_name='mutil_cnn_forecast10_300_100to10')
MTP.evalute(predicted_data=mtp3,y_test=np.reshape(ytest2,(ytest2.shape[0],ytest2.shape[1])),plot_result_name='picture/mutil_lstm_forecast10_300_100to10.png',picture_name='mutil_lstm_forecast10_300_100to10')
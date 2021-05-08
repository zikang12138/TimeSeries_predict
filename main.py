import numpy as np
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM,SimpleRNN,Conv1D,MaxPool1D,Flatten
import matplotlib.pyplot as plt
import time
from tensorflow.keras.models import load_model
import random

class Time_Predict:
    '''
    用于时间序列预测的一个类
    '''
    def __init__(self,data_name,seq_len,label_len,teach_forecast):
        #初始化
        #data_name:原始数据
        #seq_len 输入长度
        #label——len 预测步长
        #teach_forecast 是否使用时间滑窗
        self.data_name=data_name
        self.seq_len=seq_len
        self.label_len=label_len
        self.teach_forecast=teach_forecast

    def normalise_windows(self,window_data):  # 数据全部除以最开始的数据再减一
        normalised_data = []
        for window in window_data:
            normalised_window = [((float(p) / float(window[1]))) for p in window]
            normalised_data.append(normalised_window)
        return normalised_data

    def load_data(self, normalise_window=False):
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
        if normalise_window:
            result = self.normalise_windows(result)
        result = np.array(result)  # shape (4121,51) 4121代表行，51是seq_len+1
        row = round(0.9 * result.shape[0])  # round() 方法返回浮点数x的四舍五入值
        train = result[:int(row), :]  # 取前90%
        if normalise_window:
            np.random.shuffle(train)  # shuffle() 方法将序列的所有元素随机排序。
        x_train = train[:, :-self.label_len]  # 取前50列，作为训练数据
        y_train = train[:, -self.label_len:]  # 取最后一列作为标签
        x_test = result[int(row):, :-self.label_len]  # 取后10% 的前50列作为测试集
        y_test = result[int(row):, -self.label_len:]  # 取后10% 的最后一列作为标签
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))  # 最后一个维度1代表一个数据的维度
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
        return [x_train, y_train, x_test, y_test]

    def rnn(self,x_train, y_train, model_save,  ep):#model_save 模型文件保存名 ep循环次数
        model = Sequential()
        model.add(SimpleRNN(100, return_sequences=True, input_shape=(self.seq_len, 1)))
        model.add(Dropout(0.2))
        model.add(SimpleRNN(100, return_sequences=False))
        model.add(Dropout(0.2))
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
        model.add(LSTM(100, return_sequences=True, input_shape=(self.seq_len, 1)))
        model.add(Dropout(0.2))
        model.add(LSTM(100, return_sequences=False))
        model.add(Dropout(0.2))
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
        model.add(Conv1D(filters=64, kernel_size=2, activation='tanh', input_shape=(self.seq_len, 1)))
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

    def predict_point_by_point(self,model_save, x_test):#model——save模型保存名 
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
            predictedss = model.predict(x_test)  # 输入测试集的全部数据进行全部预测，（412，1）
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
        n = len(y_true)
        m = y_true - y_predict
        mse=np.average(abs(m))
        return mse

    def evalute(self,predicted_data,y_test,plot_result_name,picture_name,model_name):
        n=len(y_test)
        i=random.randint(0, n-1)
        self.plot_results(predicted_data[i], y_test[i], plot_result_name,picture_name)
        ave=self.AVE(y_test,predicted_data)
        print(model_name,end=" ")
        print(ave)


tm=Time_Predict ('data/4class.csv',seq_len=100,label_len=10,teach_forecast=True) 
[x_train,y_train,x_test,y_test]=tm.load_data()
# tm.rnn(x_train, y_train, 'model/4class_rnn2_300_10.h5', ep=300)
p=tm.predict_point_by_point('model/4class_rnn2_300_10.h5', x_test)
tm.evalute(predicted_data=p, y_test=y_test, plot_result_name='picture/4class_rnn2_300_10.png', picture_name='4class2_rnn2_300_10',model_name='rnn2')

#外部依赖
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
    def __init__(self,data_name,seq_len,label_len):
        self.data_name=data_name
        self.seq_len=seq_len
        self.label_len=label_len


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

    def rnn(self,x_train, y_train, model_save,  ep):
        model = Sequential()
        model.add(SimpleRNN(100, return_sequences=True, input_shape=(self.seq_len, 1)))
        model.add(Dropout(0.2))
        model.add(SimpleRNN(100, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(1))
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
        model.add(Dense(1))
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
        model.add(Dense(10))
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

    def predict_point_by_point(self,model_save, x_test):
        model=load_model(model_save,compile=False)
        predicted = model.predict(x_test)  # 输入测试集的全部数据进行全部预测，（412，1）
        #predicted = np.reshape(predicted, (predicted.size,))
        return predicted

    def plot_results(self,predicted_data, true_data, save_name):
        fig = plt.figure(facecolor='white')
        ax = fig.add_subplot(111)
        ax.plot(true_data, label='True Data')
        plt.plot(predicted_data, label='Prediction')
        plt.legend()
        plt.savefig(save_name)

    def MSE(self,y_true, y_predict):
        n = len(y_true)
        m = y_true - y_predict
        mse=np.std(m)
        return mse

    def evalute(self,predicted_data,y_test,plot_result_name,model_name):
        n=len(y_test)
        i=random.randint(0, n-1)
        self.plot_results(predicted_data[i], y_test[i], plot_result_name)
        mse=self.MSE(y_test,predicted_data)
        print(model_name,end=" ")
        print(mse)


tm=Time_Predict ('data/4class.csv', 100, 10) 
[x_train,y_train,x_test,y_test]=tm.load_data()
tm.cnn(x_train, y_train, 'model/4class_cnn_300_10.h5', ep=300)
# p=tm.predict_point_by_point('model/4class_cnn_300_10.h5', x_test)

# tm.evalute(p, y_test, 'picture/cnn_300_10.png', 'cnn')
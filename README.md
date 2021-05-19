# 代码文档
## 输入要求

## 函数说明
### Time_Predict类
这是一个时间序列的类（单变量预测)  
#### Time_Predict(data_name,seq_len,label_len,teach_forecast=False,n_features=1) 
data_name : 所读取的csv文件  
seq_len：输入时间序列长度  
label_len: 预测长度  
teach_forecast:为True：预测一个点，将这个点作为输入预测下一个点，默认为False  
n_feature:在单变量预测中，默认为1,**请不要修改**    
#### load_data()  
返回值 x_train y_train x_test y_test 分别为训练集输入 训练集输出 测试集输入 测试集输出    
#### rnn(x_train,y_train,model_save,ep)  
构建rnn模型并构建  
x_train: 训练集输入   
y_train: 训练集输出  
model_save:模型文件保存，保存为.h5文件 如rnn.h5  
ep:迭代次数  
#### cnn(x_train,y_train,model_save,ep) 
构建cnn模型并训练  
#### lstm(x_train,y_train,model_save,ep) 
构建lstm并训练  
#### plot_training(model_save,plot_save)
生成训练图像  
model_save：读取保存的模型文件  
plot_save: 保存训练图像

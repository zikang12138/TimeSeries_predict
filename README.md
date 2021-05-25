# 代码文档
## 环境要求
python 3.8  
tensorflow 2.4.1  
Keras 2.4.3  
numpy 1.19.5  
matplotlib 3.4.1  
pandas  1.2.4  
ide pycharm或vscode  
## 组织结构
data 存放数据  
model 存放模型文件  
picture 存放图片  
main.py 主程序  
## 输入要求
单变量输入请以4class.csv为例，其中数据为船舶运动  
多变量请以mutil.csv为例，其中x1为波浪数据，y为船舶运动  
## 函数说明
### Time_Predict类
这是一个时间序列的类（单变量预测)  
#### Time_Predict(data_name,seq_len,label_len,teach_forecast=False,n_features=1) 
data_name : 所读取的csv文件  
seq_len：输入时间序列长度  
label_len: 预测长度  
teach_forecast:为True：预测一个点，将这个点作为输入预测下一个点，默认为False  
n_feature:在单变量预测中，默认为1,**请不要修改**    
#### load_data(self)  
返回值： x_train y_train x_test y_test 分别为训练集输入 训练集输出 测试集输入 测试集输出    
#### rnn(self,x_train,y_train,model_save,ep)  
构建rnn模型并构建  
x_train: 训练集输入   
y_train: 训练集输出  
model_save:模型文件保存，保存为.h5文件 如rnn.h5  
ep:迭代次数  
#### cnn(self,x_train,y_train,model_save,ep) 
构建cnn模型并训练  
#### lstm(self,x_train,y_train,model_save,ep) 
构建lstm并训练  
#### plot_training(model_save,plot_save)
生成训练图像  **该函数目前有错误，暂时无法使用**
model_save：读取保存的模型文件  
plot_save: 保存训练图像
#### predict_result(model_save, x_test)
生成预测数据  
model_save:读取保存的模型文件   
x_test: 测试集输入    
返回值 根据x_test及模型产生的预测值   
#### plot_results(self,predicted_data, true_data, save_name,picture_name)  
生成预测数据和真实数据的对比图 **不建议单独使用,因为在evluate函数中调用该函数**  
predicted_data：预测数据的一行  
true_data：真实的输出的一行  
save_name:设定图片文件名称 如cnn.png  
picture_name：图片标题名字  
#### AVE(self,y_true, y_predict)  
评估真实数据与预测数据的误差 公式  $$L=\frac{1}{n}\sum_{i=0}^n\frac{|y\_true-y\_predict|}{|y\_true|}$$  
**不建议单独使用,因为在evluate函数中调用该函数**  
#### evalute(self,predicted_data,y_test,plot_result_name,picture_name)  
输出预测图像及误差  
predicted_data：预测数据  
y_test：测试集输出  
plot_result_name：保存的图片名  
picture_name：图片题目  
**注意本函数输入的图片是拟合效果最好的，并不代表全部**
### multi_Time_Predict  
继承自Time_Predict类，完成多输入预测  
参数与Time_Predict相同，但是**n_feature=2(2以上不清楚，请自行探索)**  
该类函数使用方法与Time_Predict相同  
主要在evalute中 MTP.evalute(predicted_data=mtp,**y_test=np.reshape(ytest2,(ytest2.shape[0],ytest2.shape[1]))**,plot_result_name='picture/rnn.png',picture_name='rnn')
注意加粗部分，对y_test做变形操作  
#### load_data(forecast)
forecast 论文（海洋平台。。。）中w的值
## 实例
在main.py中，类定义后面已经加入如下的示意代码 
```
#单变量
TM=Time_Predict(data_name='data/4class.csv',seq_len=100,label_len=10,teach_forecast=False,n_features=1)#定义一个time_predict类 
[xtrain,ytrain,xtest,ytest]=TM.load_data()#获取数据
TM.cnn(x_train=xtrain,y_train=ytrain,model_save='model/cnn.h5',ep=10)#cnn模型训练并生成训练文件 cnn.h5
tmp=TM.predict_result(model_save='model/cnn.h5',x_test=xtest)#读取模型文件并生成预测值
TM.evalute(predicted_data=tmp,y_test=ytest,plot_result_name='picture/cnn.png',picture_name='cnn')#评估模型， 生成预测曲线和实际曲线，图名为cnn 文件名为cnn.png

MTP=multi_Time_Predict(data_name='data/mutil.csv',seq_len=100,label_len=10,teach_forecast=False,n_features=2)
[xtrain2,ytrain2,xtest2,ytest2]=MTP.load_data()
MTP.rnn(x_train=xtrain2,y_train=ytrain2,model_save='model/rnn.h5',ep=10)
# MTP.plot_training('model/rnn.h5','picture/rnn_training.png')
mtp=MTP.predict_result(model_save='model/rnn.h5',x_test=xtest2)
MTP.evalute(predicted_data=mtp,y_test=ytest2,plot_result_name='picture/rnn.png',picture_name='rnn')
```
输出图像见picture内文件
## 参考资料
https://blog.csdn.net/weixin_39059031/article/details/82419728  
https://blog.csdn.net/weixin_39653948/article/details/105332534  
https://zhuanlan.zhihu.com/p/191211602  
https://zhuanlan.zhihu.com/p/51812293  
https://www.jianshu.com/p/de2008093115  

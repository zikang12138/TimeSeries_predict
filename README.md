# 代码文档
## 输入要求

## 函数说明
### Time_Predict类
这是一个时间序列的类（单变量预测）
**Time_Predict(data_name,seq_len,label_len,teach_forecast=False,n_features=1)**
data_name : 所读取的csv文件
seq_len：输入时间序列长度
label_len: 预测长度
teach_forecast:为True：预测一个点，将这个点作为输入预测下一个点，默认为False
n_feature:在单变量预测中，默认为1,请不要修改
**load_data()**

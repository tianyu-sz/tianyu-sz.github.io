### 这篇是用 paddlepaddle 写房价预测

> 原始链接：https://github.com/tianyu-sz/MachineLearning/blob/master/用PaddlePaddle写双层神经网络波士顿房价预测.ipynb

```python

#加载飞桨、Numpy和相关类库

import matplotlib.pyplot as plt
import numpy as np
import paddle.fluid as fluid
import paddle.fluid.dygraph as dygraph
from paddle.fluid.dygraph import FC

# 数据加载工具类
from util.housing_util import load_data, load_one_example

# 加载数据后的统计值变量，在 load_data 后设置，为最大，最小，平均。用于计算归一化值
global all_stastic



```


### 1.数据预处理 ，直接加载之前的工具类housing_util.py，一致的
### 2.模型设计

这里设计双层网络。一层输入层，一层隐层





```python

class Regressor(fluid.dygraph.Layer):
    
    def __init__(self, name_scope) :
        super(Regressor,self).__init__(name_scope)
        name_scope = self.full_name()
        
        # 定义全连接层，这里没加 bias。和设计的算法有点出入，后面看下API 说明加上
        self.fc1 = FC(name_scope, size=13, act=None)
        self.fc2 = FC(name_scope, size=1, act=None)
        
        
    # 网络前向计算函数
    def forward(self, inputs):
        a_h = self.fc1(inputs)
        y = self.fc2(a_h)
        
        return y
    
    

        
  
 
```

### 3.训练配置




```python
with fluid.dygraph.guard():
    model = Regressor("Regressor")
    
    # 开启模型训练模式
    model.train()
    
    
    # 加载数据
    training_data, test_data , my_all_stastic = load_data()
    
    all_stastic = my_all_stastic
    print(all_stastic)
    # 学习率设置为 0.01
    opt = fluid.optimizer.SGD(learning_rate=0.005)




```

    [array([ 88.9762, 100.    ,  25.65  ,   1.    ,   0.871 ,   8.78  ,
           100.    ,  12.1265,  24.    , 666.    ,  22.    , 396.9   ,
            37.97  ,  50.    ]), array([6.3200e-03, 0.0000e+00, 4.6000e-01, 0.0000e+00, 3.8500e-01,
           3.5610e+00, 2.9000e+00, 1.1296e+00, 1.0000e+00, 1.8700e+02,
           1.2600e+01, 7.0800e+01, 1.7300e+00, 5.0000e+00]), array([1.91589931e+00, 1.42326733e+01, 9.50232673e+00, 8.66336634e-02,
           5.31731931e-01, 6.33310891e+00, 6.44274752e+01, 4.17421361e+00,
           6.78960396e+00, 3.52910891e+02, 1.80262376e+01, 3.79971757e+02,
           1.13549505e+01, 2.41757426e+01])]



### 4.训练并保存模型

这里用的 mini-batch 方式直接copy 过来的





```python

# 这里定义一个训练过程绘图的函数
iter=0
iters=[]
train_costs=[]

def draw_train_process(iters_param, train_costs_param):
    title="training cost"
    plt.title(title, fontsize=24)
    plt.xlabel("iter", fontsize=14)
    plt.ylabel("cost", fontsize=14)
    plt.plot(iters_param, train_costs_param,color='red',label='training cost') 
    plt.grid()
    plt.show()
    
    
# 开始训练
with dygraph.guard():
    EPOCH_NUM = 10000   # 设置外层循环次数
    BATCH_SIZE = 404  # 设置batch大小
    
    # 定义外层循环
    for epoch_id in range(EPOCH_NUM):
        # 在每轮迭代开始之前，将训练数据的顺序随机的打乱
        np.random.shuffle(training_data)
        # 将训练数据进行拆分，每个batch包含BATCH_SIZE条数据
        mini_batches = [training_data[k:k+BATCH_SIZE] for k in range(0, len(training_data), BATCH_SIZE)]
        # 定义内层循环
        for iter_id, mini_batch in enumerate(mini_batches):
            x = np.array(mini_batch[:, :-1]).astype('float32') # 获得当前批次训练数据
            y = np.array(mini_batch[:, -1:]).astype('float32') # 获得当前批次训练标签（真实房价）
            # 将numpy数据转为飞桨动态图variable形式
            house_features = dygraph.to_variable(x)
            prices = dygraph.to_variable(y)
            
            # print(house_features)
            # 前向计算
            
            predicts = model(house_features)
            
            # 计算损失
            loss = fluid.layers.square_error_cost(predicts, label=prices)
            avg_loss = fluid.layers.mean(loss)
            # if iter_id%20==0:
            #     print("epoch: {}, iter: {}, loss is: {}".format(epoch_id, iter_id, avg_loss.numpy()))
            
            
            iter=iter+1
            iters.append(iter)
            train_costs.append(avg_loss.numpy()[0])
            
            
            # 反向传播
            avg_loss.backward()
            # 最小化loss,更新参数
            opt.minimize(avg_loss)
            # 清除梯度
            model.clear_gradients()
            
    # 保存模型
    fluid.save_dygraph(model.state_dict(), 'LR_model')
    print("以参数 epoch: {0}, batch_size: {1} 训练完成.模型保存成功，模型参数保存在LR_model中".format(EPOCH_NUM,BATCH_SIZE))
    
draw_train_process(iters,train_costs)
    
```

    以参数 epoch: 10000, batch_size: 404 训练完成.模型保存成功，模型参数保存在LR_model中



![训练图](12/paddle_paddle_housing_data.png)


### 5. 测试模型，对所有结果进行预测并画出图




```python
with dygraph.guard():
    #所有预测结果
    results = np.zeros(len(test_data))
    labels = np.zeros(len(test_data))
    model_dict, _ = fluid.load_dygraph("LR_model")
    model.load_dict(model_dict)
    # 设置到预测状态
    model.eval()
    for j in range(0,len(test_data)):
        current_test_data, label = load_one_example(j,test_data)
        current_test_data = dygraph.to_variable(current_test_data)
        result = model(current_test_data)
        max_values = all_stastic[0]
        min_values = all_stastic[1]
        avg_values = all_stastic[2]
        
        result = result * (max_values[-1] - min_values[-1]) + avg_values[-1]
        results[j] = result.numpy()
        label = label * (max_values[-1] - min_values[-1]) + avg_values[-1]
        labels[j] = label
        
        print("预测结果 is {}, 标签结果 is {}".format(result.numpy(), label))


```

    预测结果 is [[8.063351]], 标签结果 is 8.5
    预测结果 is [[9.703434]], 标签结果 is 5.0
    预测结果 is [[5.968216]], 标签结果 is 11.9
    预测结果 is [[19.963814]], 标签结果 is 27.9
    预测结果 is [[12.429203]], 标签结果 is 17.2
    预测结果 is [[20.81434]], 标签结果 is 27.5
    预测结果 is [[16.703356]], 标签结果 is 15.0
    预测结果 is [[18.021067]], 标签结果 is 17.2
    预测结果 is [[0.04165268]], 标签结果 is 17.9
    预测结果 is [[11.143707]], 标签结果 is 16.3
    预测结果 is [[-4.4441757]], 标签结果 is 7.0
    预测结果 is [[11.30804]], 标签结果 is 7.199999999999999
    预测结果 is [[15.454807]], 标签结果 is 7.5
    预测结果 is [[7.2800655]], 标签结果 is 10.4
    预测结果 is [[9.168554]], 标签结果 is 8.8
    预测结果 is [[17.289066]], 标签结果 is 8.4
    预测结果 is [[21.396982]], 标签结果 is 16.7
    预测结果 is [[19.32442]], 标签结果 is 14.2
    预测结果 is [[18.431263]], 标签结果 is 20.8
    预测结果 is [[13.882022]], 标签结果 is 13.4
    预测结果 is [[14.697031]], 标签结果 is 11.7
    预测结果 is [[11.241514]], 标签结果 is 8.300000000000002
    预测结果 is [[16.821507]], 标签结果 is 10.2
    预测结果 is [[16.717302]], 标签结果 is 10.9
    预测结果 is [[15.630312]], 标签结果 is 11.0
    预测结果 is [[14.65843]], 标签结果 is 9.5
    预测结果 is [[18.882334]], 标签结果 is 14.5
    预测结果 is [[19.954376]], 标签结果 is 14.1
    预测结果 is [[22.571402]], 标签结果 is 16.1
    预测结果 is [[19.414867]], 标签结果 is 14.3
    预测结果 is [[18.281849]], 标签结果 is 11.7
    预测结果 is [[16.014956]], 标签结果 is 13.4
    预测结果 is [[17.419338]], 标签结果 is 9.6
    预测结果 is [[11.028912]], 标签结果 is 8.7
    预测结果 is [[6.3057537]], 标签结果 is 8.4
    预测结果 is [[13.634687]], 标签结果 is 12.8
    预测结果 is [[13.915702]], 标签结果 is 10.500000000000002
    预测结果 is [[19.06953]], 标签结果 is 17.1
    预测结果 is [[20.240389]], 标签结果 is 18.4
    预测结果 is [[19.905542]], 标签结果 is 15.4
    预测结果 is [[12.833725]], 标签结果 is 10.8
    预测结果 is [[14.482351]], 标签结果 is 11.800000000000002
    预测结果 is [[19.532452]], 标签结果 is 14.9
    预测结果 is [[19.912434]], 标签结果 is 12.6
    预测结果 is [[18.840103]], 标签结果 is 14.1
    预测结果 is [[18.91743]], 标签结果 is 13.0
    预测结果 is [[19.499859]], 标签结果 is 13.4
    预测结果 is [[21.349045]], 标签结果 is 15.2
    预测结果 is [[20.053133]], 标签结果 is 16.1
    预测结果 is [[25.401875]], 标签结果 is 17.8
    预测结果 is [[18.329922]], 标签结果 is 14.9
    预测结果 is [[18.3993]], 标签结果 is 14.1
    预测结果 is [[14.761218]], 标签结果 is 12.7
    预测结果 is [[15.219011]], 标签结果 is 13.5
    预测结果 is [[19.293585]], 标签结果 is 14.9
    预测结果 is [[20.048498]], 标签结果 is 20.0
    预测结果 is [[21.468227]], 标签结果 is 16.4
    预测结果 is [[21.897223]], 标签结果 is 17.7
    预测结果 is [[21.622017]], 标签结果 is 19.5
    预测结果 is [[24.66067]], 标签结果 is 20.2
    预测结果 is [[21.590042]], 标签结果 is 21.4
    预测结果 is [[18.712383]], 标签结果 is 19.9
    预测结果 is [[15.932061]], 标签结果 is 19.0
    预测结果 is [[16.62816]], 标签结果 is 19.1
    预测结果 is [[17.104801]], 标签结果 is 19.1
    预测结果 is [[18.439224]], 标签结果 is 20.1
    预测结果 is [[20.307102]], 标签结果 is 19.9
    预测结果 is [[22.78922]], 标签结果 is 19.6
    预测结果 is [[22.92166]], 标签结果 is 23.2
    预测结果 is [[27.078156]], 标签结果 is 29.8
    预测结果 is [[15.510456]], 标签结果 is 13.8
    预测结果 is [[15.810659]], 标签结果 is 13.3
    预测结果 is [[20.95625]], 标签结果 is 16.7
    预测结果 is [[10.614654]], 标签结果 is 12.0
    预测结果 is [[19.41286]], 标签结果 is 14.6
    预测结果 is [[22.322279]], 标签结果 is 21.4
    预测结果 is [[23.64771]], 标签结果 is 23.0
    预测结果 is [[27.928967]], 标签结果 is 23.7
    预测结果 is [[29.825233]], 标签结果 is 25.0
    预测结果 is [[21.045368]], 标签结果 is 21.8
    预测结果 is [[19.859436]], 标签结果 is 20.6
    预测结果 is [[23.363604]], 标签结果 is 21.2
    预测结果 is [[20.145649]], 标签结果 is 19.1
    预测结果 is [[21.54454]], 标签结果 is 20.6
    预测结果 is [[14.368862]], 标签结果 is 15.2
    预测结果 is [[10.538194]], 标签结果 is 7.0
    预测结果 is [[5.47645]], 标签结果 is 8.099999999999998
    预测结果 is [[16.914381]], 标签结果 is 13.6
    预测结果 is [[19.414948]], 标签结果 is 20.1
    预测结果 is [[20.654835]], 标签结果 is 21.8
    预测结果 is [[20.85927]], 标签结果 is 24.5
    预测结果 is [[16.926046]], 标签结果 is 23.1
    预测结果 is [[13.492403]], 标签结果 is 19.7
    预测结果 is [[19.393959]], 标签结果 is 18.3
    预测结果 is [[21.648792]], 标签结果 is 21.2
    预测结果 is [[18.172852]], 标签结果 is 17.5
    预测结果 is [[20.79073]], 标签结果 is 16.8
    预测结果 is [[24.602003]], 标签结果 is 22.4
    预测结果 is [[22.786432]], 标签结果 is 20.6
    预测结果 is [[29.085186]], 标签结果 is 23.9
    预测结果 is [[27.46927]], 标签结果 is 22.0
    预测结果 is [[22.803867]], 标签结果 is 11.9


### 如何评价测试结果呢？

#### 1.最常用的是“均方误差” 


```python
import math
def get_mse(records_real, records_predict):
    """
    均方误差 估计值与真值 偏差
    """
    if len(records_real) == len(records_predict):
        return sum([(x - y) ** 2 for x, y in zip(records_real, records_predict)]) / len(records_real)
    else:
        return None
def get_rmse(records_real, records_predict):
    """
    均方根误差：是均方误差的算术平方根
    """
    mse = get_mse(records_real, records_predict)
    if mse:
        return math.sqrt(mse)
    else:
        return None

mse = get_mse(labels,results)
rmse = get_rmse(labels,results)
print ("mse: {0},  rmse: {1}".format(mse, rmse))
```

    mse: 21.60603418305056,  rmse: 4.648229144851893


#### 2.展示训练程度与泛化误差、偏差之间的关系
> todo

#### 3.问题点

* 预测后有负值，未细调查原因



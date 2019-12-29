### 这篇是用 paddlepaddle 写房价预测





```python

#加载飞桨、Numpy和相关类库

import paddle.fluid as fluid
import paddle.fluid.dygraph as dygraph
from paddle.fluid.dygraph import FC
import numpy as np

# 数据加载工具类
from  util.housing_util import load_data, load_one_example
# 加载数据后的统计值变量，在 load_data 后设置，为最大，最小，平均。用于计算归一化值
global all_stastic



```


### 1.数据预处理 ，直接加载之前的工具类，一致的
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
    opt = fluid.optimizer.SGD(learning_rate=0.01)


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
with dygraph.guard():
    EPOCH_NUM = 100   # 设置外层循环次数
    BATCH_SIZE = 50  # 设置batch大小
    
    # 定义外层循环
    for epoch_id in range(EPOCH_NUM):
        # 在每轮迭代开始之前，将训练数据的顺序随机的打乱
        np.random.shuffle(training_data)
        # 将训练数据进行拆分，每个batch包含10条数据
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
            if iter_id%20==0:
                print("epoch: {}, iter: {}, loss is: {}".format(epoch_id, iter_id, avg_loss.numpy()))
            
            # 反向传播
            avg_loss.backward()
            # 最小化loss,更新参数
            opt.minimize(avg_loss)
            # 清除梯度
            model.clear_gradients()
    # 保存模型
    fluid.save_dygraph(model.state_dict(), 'LR_model')
    print("模型保存成功，模型参数保存在LR_model中")
    
    
```

    epoch: 0, iter: 0, loss is: [0.1481473]
    epoch: 1, iter: 0, loss is: [0.10623512]
    epoch: 2, iter: 0, loss is: [0.06507152]
    epoch: 3, iter: 0, loss is: [0.10395487]
    epoch: 4, iter: 0, loss is: [0.07486193]
    epoch: 5, iter: 0, loss is: [0.09647383]
    epoch: 6, iter: 0, loss is: [0.0592256]
    epoch: 7, iter: 0, loss is: [0.05568671]
    epoch: 8, iter: 0, loss is: [0.05602317]
    epoch: 9, iter: 0, loss is: [0.0466639]
    epoch: 10, iter: 0, loss is: [0.04839917]
    epoch: 11, iter: 0, loss is: [0.05812053]
    epoch: 12, iter: 0, loss is: [0.05793015]
    epoch: 13, iter: 0, loss is: [0.06924678]
    epoch: 14, iter: 0, loss is: [0.05832703]
    epoch: 15, iter: 0, loss is: [0.0427563]
    epoch: 16, iter: 0, loss is: [0.03139522]
    epoch: 17, iter: 0, loss is: [0.02521168]
    epoch: 18, iter: 0, loss is: [0.04452838]
    epoch: 19, iter: 0, loss is: [0.03508262]
    epoch: 20, iter: 0, loss is: [0.04956186]
    epoch: 21, iter: 0, loss is: [0.04176604]
    epoch: 22, iter: 0, loss is: [0.04395108]
    epoch: 23, iter: 0, loss is: [0.01881038]
    epoch: 24, iter: 0, loss is: [0.04877205]
    epoch: 25, iter: 0, loss is: [0.04403654]
    epoch: 26, iter: 0, loss is: [0.02880949]
    epoch: 27, iter: 0, loss is: [0.03630515]
    epoch: 28, iter: 0, loss is: [0.04419984]
    epoch: 29, iter: 0, loss is: [0.04238167]
    epoch: 30, iter: 0, loss is: [0.03977687]
    epoch: 31, iter: 0, loss is: [0.02600098]
    epoch: 32, iter: 0, loss is: [0.03183369]
    epoch: 33, iter: 0, loss is: [0.01639062]
    epoch: 34, iter: 0, loss is: [0.02203416]
    epoch: 35, iter: 0, loss is: [0.03184173]
    epoch: 36, iter: 0, loss is: [0.02381719]
    epoch: 37, iter: 0, loss is: [0.05137089]
    epoch: 38, iter: 0, loss is: [0.04110888]
    epoch: 39, iter: 0, loss is: [0.039241]
    epoch: 40, iter: 0, loss is: [0.03647748]
    epoch: 41, iter: 0, loss is: [0.03187416]
    epoch: 42, iter: 0, loss is: [0.0309484]
    epoch: 43, iter: 0, loss is: [0.04838527]
    epoch: 44, iter: 0, loss is: [0.02272531]
    epoch: 45, iter: 0, loss is: [0.02989025]
    epoch: 46, iter: 0, loss is: [0.02336251]
    epoch: 47, iter: 0, loss is: [0.02699223]
    epoch: 48, iter: 0, loss is: [0.02866812]
    epoch: 49, iter: 0, loss is: [0.02402335]
    epoch: 50, iter: 0, loss is: [0.04171531]
    epoch: 51, iter: 0, loss is: [0.04470729]
    epoch: 52, iter: 0, loss is: [0.02488734]
    epoch: 53, iter: 0, loss is: [0.01800954]
    epoch: 54, iter: 0, loss is: [0.02715453]
    epoch: 55, iter: 0, loss is: [0.01856082]
    epoch: 56, iter: 0, loss is: [0.022078]
    epoch: 57, iter: 0, loss is: [0.02643769]
    epoch: 58, iter: 0, loss is: [0.03292308]
    epoch: 59, iter: 0, loss is: [0.02866376]
    epoch: 60, iter: 0, loss is: [0.02374008]
    epoch: 61, iter: 0, loss is: [0.02626836]
    epoch: 62, iter: 0, loss is: [0.04346812]
    epoch: 63, iter: 0, loss is: [0.01654934]
    epoch: 64, iter: 0, loss is: [0.01168822]
    epoch: 65, iter: 0, loss is: [0.01541619]
    epoch: 66, iter: 0, loss is: [0.03037399]
    epoch: 67, iter: 0, loss is: [0.01603233]
    epoch: 68, iter: 0, loss is: [0.02457377]
    epoch: 69, iter: 0, loss is: [0.01566416]
    epoch: 70, iter: 0, loss is: [0.03497444]
    epoch: 71, iter: 0, loss is: [0.02841765]
    epoch: 72, iter: 0, loss is: [0.02007245]
    epoch: 73, iter: 0, loss is: [0.01647839]
    epoch: 74, iter: 0, loss is: [0.02841564]
    epoch: 75, iter: 0, loss is: [0.03026055]
    epoch: 76, iter: 0, loss is: [0.01914282]
    epoch: 77, iter: 0, loss is: [0.02681017]
    epoch: 78, iter: 0, loss is: [0.02798228]
    epoch: 79, iter: 0, loss is: [0.01753144]
    epoch: 80, iter: 0, loss is: [0.01593197]
    epoch: 81, iter: 0, loss is: [0.03224338]
    epoch: 82, iter: 0, loss is: [0.0199582]
    epoch: 83, iter: 0, loss is: [0.02490276]
    epoch: 84, iter: 0, loss is: [0.01723043]
    epoch: 85, iter: 0, loss is: [0.01921878]
    epoch: 86, iter: 0, loss is: [0.02397879]
    epoch: 87, iter: 0, loss is: [0.02429887]
    epoch: 88, iter: 0, loss is: [0.01738256]
    epoch: 89, iter: 0, loss is: [0.02251818]
    epoch: 90, iter: 0, loss is: [0.02398938]
    epoch: 91, iter: 0, loss is: [0.00912767]
    epoch: 92, iter: 0, loss is: [0.01622294]
    epoch: 93, iter: 0, loss is: [0.0202231]
    epoch: 94, iter: 0, loss is: [0.01655493]
    epoch: 95, iter: 0, loss is: [0.01540987]
    epoch: 96, iter: 0, loss is: [0.02027803]
    epoch: 97, iter: 0, loss is: [0.01916204]
    epoch: 98, iter: 0, loss is: [0.02039809]
    epoch: 99, iter: 0, loss is: [0.01872693]
    模型保存成功，模型参数保存在LR_model中


### 5. 测试模型




```python

with dygraph.guard():
    model_dict, _ = fluid.load_dygraph("LR_model")
    model.load_dict(model_dict)
    # 设置到预测状态
    model.eval()
    
    # 
    one_test_data, label = load_one_example()
    print(one_test_data,label)
    
    one_test_data = dygraph.to_variable(one_test_data)
    results = model(one_test_data)
    max_values = all_stastic[0]
    min_values = all_stastic[1]
    avg_values = all_stastic[2]

    results = results * (max_values[-1] - min_values[-1]) + avg_values[-1]
    
    print("预测结果 is {}, 标签结果 is {}".format(results.numpy(), label))

```

    [[-0.01827921 -0.14232673  0.00745031 -0.08663366  0.10960508 -0.18070683
       0.08725566 -0.12509103 -0.03433061  0.07951798  0.12486834  0.0519112
       0.2700069 ]] 19.7
    预测结果 is [[16.939106]], 标签结果 is 19.7


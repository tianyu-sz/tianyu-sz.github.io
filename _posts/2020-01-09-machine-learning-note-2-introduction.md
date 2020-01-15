---
layout: post
title: '吴恩达《机器学习》训练营作业笔记-2 Introduction'
subtitle: '机器学习概述'
date: 2020-01-09
categories: 机器学习 人工智能
cover: '/assets/img/post/2020_01/automatic-helicop.jpg'
tags: 机器学习 人工智能 笔记
---

# 吴恩达《机器学习》训练营作业笔记-2 Introduction

### 机器学习的定义：
机器学习有多种定义方式：
* Machine learning is the science of getting computers to learn, without being explicitly programmed
* Tom Mitchell给的定义: "A computer program is said to learn from experience E with respect to some class of tasks T and performance measure P, 
if its performance at tasks in T, as measured by P, improves with experience E."
* 针对目前流行的技术，我个人的理解：有一个实际任务(**T**)，以及可以用于训练的经验数据集 (**E**)，通过机器学习的算法设计并训练出模型，能在新的数据上达到比较好的预测结果。
预测结果的评价指标(**P**)可能有多种形式，如准确度等

定义的例子：以垃圾邮件过滤的问题为例：
* 实际任务T：对邮件是否是垃圾邮件进行判断分类
* 经验数据集E：我标记的垃圾邮件或者由别人标记的海量垃圾邮件
* 预测结果评价指标P：正确/错误分类的邮件数

### 机器学习
这里课程提到了一个观点是机器学习给了计算机新的capability（"New capability of computers"），这个观点细想起来确实是这样的。因为我们目前使用的计算机、手机设备其实都是依赖
固定编写好的程序进行工作。APP里写了这段对应代码，才有对应功能。而对于机器学习来说，并没有进行显示的编程，而是通过数据拟合的方式（目前流行的是这种形式）进行工作。可以看做扩大
了计算机的使用范围，处理一些无法使用手动编程解决的问题。
#### Example
##### 数据挖掘（Database mining）：从互联网、物联网(automation/web)中生成的海量数据
* 自主机器人
* 电子医疗记录：将医疗档案转换成医学知识，进而更有效的识别疾病，个人、群体
* 计算生物学：分析人类DNA，这里举了一个无监督学习的例子，分析不同人具有的相同基因片段，进而将人分组分类，提前不知道基因片段的具体作用，图:
![分析人类DNA](/assets/img/post/2020_01/genes.jpg)

##### 无法手工编程的任务
* 自动飞行的飞机 图
![自动飞行的飞机](/assets/img/post/2020_01/automatic-helicop.jpg)
* 手写字识别，邮政里有用到，用于识别包裹上邮单地址
* 自然语言处理、机器视觉


##### 个性化定制的功能：每个用户都不同，无法通过固定的一段代码编程实现
* "千人千面"这种功能
* Amazon，Netflix的产品推荐

##### 理解人类如何学习：目前机器学习和人类智慧是"形似"，通过研究机器学习算法，给研究人类智慧，人类是如何学习的提供了参考。有一门学科叫"认知心理学"

### 机器学习的分类

一种分类方式
* 监督学习：训练集中，已经给出了正确答案（"right answer" given）
* 非监督学习：!监督学习咯

另一种分类方式：
* 回归：预测结果是连续值
* 分类：预测结果是离散值

课程这里举了一个癌症识别的例子，并且提到使用支持向量机以及一些简单的数学trick，可以处理无限多的特征属性。

### 无监督学习：将数据进行cluster分类。这里举了一些有意思的例子：
* 谷歌新闻的聚类
* 基因分析：分析每个个体是否具有特定基因，进而将人根据基因进行分组分类
* 大型计算机集群中，找到哪些机器tends to work together， 进而organize computing clusters，使集群工作更高效
* Social network analysis。通过分析email来往，facebook，google+的朋友，分析出哪些是好友组，哪些是仅仅认识
* Market Segment：通过分析客户信息，找出细分市场
* 天文学：关于星系是如何诞生的理论

### 鸡尾就会算法：很惊艳
![鸡尾酒会问题图片](/assets/img/post/2020_01/cocktail-party-problem.png)
* 一个是从两个人同时说话的音频中分离出每个人单独的说话内容。一个是从讲话中分离出讲话内容和背景音。（这种用普通编程是做不到的）
* 公式如下：
[W,s,v] = svd((repmat(sum(x.*x,1),size(x,1),1).*x)*x');

### 笔记手稿
![手稿2-1](/assets/img/post/2020_01/note-handcraft-2-1.jpg){:height="200px" width="150px"}
![手稿2-2](/assets/img/post/2020_01/note-handcraft-2-2.jpg){:height="200px" width="150px"}
![手稿2-3](/assets/img/post/2020_01/note-handcraft-2-3.jpg){:height="200px" width="150px"}
![手稿2-4](/assets/img/post/2020_01/note-handcraft-2-4.jpg){:height="200px" width="150px"}
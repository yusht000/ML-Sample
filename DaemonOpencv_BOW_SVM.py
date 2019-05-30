# -*- coding: utf-8 -*-
"""
Created on Thu May 23 11:01:24 2019
@author: Echo

"""

import numpy as np
import cv2 as cv

class BOW(object):
    
    def __init__(self,):
        
        # 创建一个 sift 对象  用于关键点提取
        # sift 算法构建 尺度空间 
        # 尺度空间 是根据 人类观察事物的 有远及近的角度出发 远处可以看到整体轮廓 近处可以看到局部的细节
        # 根据尺度空间 来获取图像关键点（是尺度空间的极值点 获取其位置 尺度 旋转不变量 ）
        # 关键点 与影像的大小 和 旋转无关
        self.feature_detector= cv.xfeatures2d.SIFT_create()
        # 创建一个 sift 对象  用于关键点描述符提取
        # 关键点描述符 在提取关键点位置的基础上 在尺度金字塔附近进行小区域内梯度直方图统计
        # 将其归一化形成最终的描述向量 
        # 对图像中所有的关键点进行归一化形成图像的描述向量
        self.descriptor_extractor = cv.xfeatures2d.SIFT_create()
   
   
    def path(self,cls,i):
        
        # 用于获取图片的全路径
        return '%s/%s.%d.jpg' %(self.train_path,cls,i + 1)


    def fit(self,train_path,k):

        self.train_path = train_path
        # train_path 训练的参数
        # k k_means 参数        
        # 近似快速匹配
        # kd树的建立 就是在高纬度的空间上 对数据进行建立索引 
        # 目的 减少计算的复杂度 可以快速查找 空间上与目标最近的几个点 减少遍历空间来进行查找；
        flann_params= dict(algorithm=1,tree=5)
        flann = cv.FlannBasedMatcher(flann_params,{})
        
        # k-means 属于无监督聚类算法
        # 关键点是 K 的指定 （可以结合其他算法来优化 k 值得指定）
        # 内容 ： 形成欧式距离空间矩阵 进行加分组 根据质心形成新的矩阵与分组
        # 停止条件 是 质心不再改变；
        #处理好的特征数据进行全部合并 利用聚类把特征词分成若干类 每一类相当于于一个视觉词汇
        bow_kmeans_trainer = cv.BOWKMeansTrainer(k)
        
        pos = 'dog'
        neg = 'cat'

        #指定用于提取词汇字典的样本数
        length = 10
        # 合并特征数据 每个类从数据集中读取length张图片, 通过聚类创建视觉词汇
        # 每一幅都具有自己的特征数据 通过对十幅图像的特征数据 利用 k-means 无监督聚类算法
        # 将图像相同或相近的部分进行 聚类 形成基本的词典词汇
        for i in range(length):
            bow_kmeans_trainer.add(self.sift_descriptor_extractor(self.path(pos,i)))
            bow_kmeans_trainer.add(self.sift_descriptor_extractor(self.path(neg,i)))
        # 进行 k-means 聚类 返回词汇字典 也是聚类中心；
        voc = bow_kmeans_trainer.cluster()
        #输出词汇字典 基本词汇一个128行向量
        print(type(voc),voc.shape)
        
        
        
        # 初始化bow提取器（设置词汇字典），用于提取每一个张图像的Bow特征描述
        # 构造器 方法中 传递参数 sift 对象 和 进行 图像匹配的算法；
        self.bow_img_descriptor_extractor = cv.BOWImgDescriptorExtractor(self.descriptor_extractor,flann)
        
        # 给这个对象设置基础词汇向量 
        self.bow_img_descriptor_extractor.setVocabulary(voc)
        
        #创建两个数组 分别对应训练数据和标签
        # 对图像词频特征向量数据 已经把图像特征向量进行了转化 进行标签注明
        traindata,trainlabels =[] ,[]
        for i in range(400):    
            traindata.extend(self.bow_descriptor_extractor(self.path(pos,i)))
            trainlabels.append(1)
            traindata.extend(self.bow_descriptor_extractor(self.path(neg,i)))
            trainlabels.append(-1)
            
            
        #创建一个 svm 对象
        self.svm = cv.ml.SVM_create()
        # 使用训练数据和标签进行训练
        self.svm.train(np.array(traindata),cv.ml.ROW_SAMPLE,np.array(trainlabels))
       
    def predict(self,img_path):
        
        data = self.bow_descriptor_extractor(img_path)
        res = self.svm.predict(data)
        
        print(img_path,'\t',res[1][0][0])
        
        if res[1][0][0] == 1.0 :
           return True
        else :
           return False
       
    def sift_descriptor_extractor(self,img_path): 
        # 特征提取： 提取数据集中中每一幅图像的特征点 ，然后提取特征描述符 形成特征数据
        im = cv.imread(img_path,0)
        return self.descriptor_extractor.compute(im,self.feature_detector.detect(im))[1]
        
        
        
    def bow_descriptor_extractor(self,img_path):
        
        # 对图像进行bow编码 提取图像bow特征描述 利用视觉词袋子量化图像特特征 相当于换了一种描述图像特征的向量 是词频出现个数
        im = cv.imread(img_path,0)
        return self.bow_img_descriptor_extractor.compute(im,self.feature_detector.detect(im))
        
if __name__ == '__main__' :
    
  test_samples = 100
  test_results = np.zeros(test_samples,dtype=np.bool)
 
  # 训练集图片路径 猫和狗两类
  train_path = 'E:/ML_Data_Img/train'

  bow = BOW()
  bow.fit(train_path,40)
  
  # 指定测试图像路径
  for index in range (test_samples):
      dog = 'E:/ML_Data_Img/train/dog.{0}.jpg'.format(index)
      dog_img = cv.imread(dog)
      # 预测
      dog_predict = bow.predict(dog)
      test_results[index] = dog_predict
      
      
      
  # 计算准确率
  accuracy = np.mean(test_results.astype(dtype=np.float32))
  print("测试准确率：", accuracy)
  
  # 可视化最后一个
  font = cv.FONT_HERSHEY_SIMPLEX
  if test_results[0]:
    cv.putText(dog_img, 'Dog Detected', (10, 30), font, 1, (0, 255, 0), 2, cv.LINE_AA)
 
  cv.imshow('dog_img', dog_img)
 
  cv.waitKey(0)
  cv.destroyAllWindows()



    
     
            
            
            
        
        
        
        
        
        
        
        
        
        
            
            
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
            
    
    
    
    
    
      

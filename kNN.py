#coding:utf8
#导入科学计算包和运算符模块
from numpy import *
import operator
import matplotlib
import matplotlib.pyplot as plt
# 创建训练数据集
def creatDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ["A","A","B","B"]
    return group,labels

# 接受group和labels值
group,labels = creatDataSet()
#print group,labels

# 计算测试数据
def classify0(inX,dataSet,labels,k):
    dataSetSize = dataSet.shape[0]
    # 计算距离
    diffMat = tile(inX,(dataSetSize,1))-dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort()
    classCount = {}
    # 选择距离最小的K个点
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    sortedClassCount = sorted(classCount.iteritems(),key = operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]


# knn改善约会网站配对效果
def file2matrix(filename):
    fr = open(filename)
    # 按照行读取文件
    arraryOLines = fr.readlines()
    # 查看总行数
    numberOfLines = len(arraryOLines)
    # 声明一个同行数0矩阵
    returnMat = zeros((numberOfLines,3))
    # 声明标签
    classLabelVector = []
    # 修改矩阵标志
    index = 0
    for line in arraryOLines:
        line = line.strip()
        listFromLine = line.split('\t')
        # 更新矩阵对应样本数据
        returnMat[index,:] = listFromLine[0:3]
        # 添加标签
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    fr.close()
    return returnMat,classLabelVector

datingDataMat,datingLabels = file2matrix('E:\knn\datingTestSet.txt')
# fig = plt.figure()
# ax = fig.add_subplot(111)
# # 第一个参数为x轴,第二个参数为Y轴
# ax.scatter(datingDataMat[:,0],datingDataMat[:,1],15.0*array(datingLabels),15.0*array(datingLabels))
# plt.show()

# 归一化数值
def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals,(m,1))
    normDataSet = normDataSet/tile(ranges,(m,1))
    return normDataSet,ranges,minVals

# 接收归一化矩阵,差值矩阵,和最小值
# normMat,ranges,minVals = autoNorm(datingDataMat)

# 测试准确率
def datingClassTest():
    hoTatio = 0.10
    datingDataMat,datingLabels = file2matrix('E:\knn\datingTestSet.txt')
    normMat,ranges,minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoTatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],4)
        print "预测值为:%d,真正值为:%d"%(classifierResult,datingLabels[i])
        if (classifierResult != datingLabels[i]):
            errorCount += 1.0
    print '错误率为:%f'%(errorCount/float(numTestVecs))
#datingClassTest()

# 预测函数
def classfyPerson():
    resultList = ['不喜欢','一般','喜欢']
    percentTats = float(raw_input('游戏时间占比:'))
    ffMiles = float(raw_input('飞行公里数:'))
    iceCream = float(raw_input('每周消费冰淇淋公升数:'))
    datingDataMat,datingLabels = file2matrix('.\datingTestSet.txt')
    normMat,ranges,minVals = autoNorm(datingDataMat)
    inArr = array([ffMiles,percentTats,iceCream])
    classifierResult = classify0((inArr-minVals)/ranges,normMat,datingLabels,3)
    print '你可能喜欢这个人的程度为:',resultList[classifierResult-1]
classfyPerson()
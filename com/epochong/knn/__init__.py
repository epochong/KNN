import random
import csv
# 读取
with open('Prostate_Cancer.csv', 'r') as file:
    reader = csv.DictReader(file)
    data_set = [row for row in reader]

# 分组
random.shuffle(data_set)
n = len(data_set) // 3
test_set = data_set[0:n]
train_set = data_set[n:]


# KNN
# 距离
def distance(d1, d2):
    res = 0
    for key in("radius", "texture", "perimeter", "area", "smoothness", "compactness", "symmetry", "fractal_dimension"):
        res += (float(d1[key]) - float(d2[key])) ** 2
    return res ** 0.5


def knn(each_group,K):
    """
    KNN算法预测前列腺癌症危险性
    :param each_group: 测试集每一行的数据
    :return: 危险性 {B,M}
    """
    # 1.距离
    # 计算出传过来的测试集的
    # a.训练集的每组对应癌症程度
    # b.一组数据和训练集的每组数据中的每个数据的距离总和
    res = [
        {
            "train_real_result": train_each_group['diagnosis_result'],
            "distance": distance(each_group, train_each_group)
        }
        for train_each_group in train_set
    ]

    # 2.排序
    res = sorted(res, key=lambda item: item['distance'])

    # 3.取前K个
    k_nums = res[0:K]
    # 总距离
    _sum = 0
    for i in k_nums:
        _sum += i['distance']

    # 4.加权平均
    result = {'B': 0, 'M': 0}
    # B,M的对应的
    for i in k_nums:
        result[i['train_real_result']] += 1 - i['distance'] / _sum
    if result['B'] > result['M']:
        return 'B'
    else:
        return 'M'


# 测试阶段
correct_count = 0
for k in range(1, 11):
    for test_each_line in test_set:
        real_result = test_each_line['diagnosis_result']
        predict_result = knn(test_each_line,k)
        if real_result == predict_result:
            correct_count += 1
    print("K取值为：",k ,",准确率：{:.2f}%".format(100 * correct_count / len(test_set)))
    correct_count = 0

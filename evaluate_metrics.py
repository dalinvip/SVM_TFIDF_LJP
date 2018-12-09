# @Author : bamtercelboo
# @Datetime : 2018/12/9 8:30
# @File : evaluate metrics.py
# @Last Modify Time : 2018/12/9 8:30
# @Contact : bamtercelboo@{gmail.com, 163.com}

"""
    FILE :  evaluate metrics.py
    FUNCTION : None
"""

import os
import sys
import time
import json
from svm import *
from predictor.predictor import Predictor
from judger.judger import Judger

class Metrics(object):
    """
        Metrics
    """

    def __init__(self, **kwargs):
        self.train_path = kwargs["train_path"]
        self.valid_path = kwargs["valid_path"]
        self.test_path = kwargs["test_path"]
        self.accusation_path = kwargs["accusation_path"]
        self.law_path = kwargs["law_path"]
        self.judger = Judger(accusation_path=self.accusation_path, law_path=self.law_path)
        self.train(train_path=train_path)
        self.calculate_metrics(path=self.valid_path)

    def train(self, train_path):
        """
        :param train_path:
        :return:
        """
        print("reading...")
        alltext, accu_label, law_label, time_label = read_trainData(train_path)
        print("cut text...")
        train_data = cut_text(alltext)
        print("train tfidf...")
        tfidf = train_tfidf(train_data)
        vec = tfidf.transform(train_data)
        print('accu SVC')
        accu = train_SVC(vec, accu_label)
        print('law SVC')
        law = train_SVC(vec, law_label)
        print('time SVC')
        time = train_SVC(vec, time_label)

        print('saving model to ./predictor/model/*.model')
        joblib.dump(tfidf, './predictor/model/tfidf.model')
        joblib.dump(accu, './predictor/model/accu.model')
        joblib.dump(law, './predictor/model/law.model')
        joblib.dump(time, './predictor/model/time.model')

    def calculate_metrics(self, path):
        """
        :param path:
        :return:
        """
        predictor = Predictor()
        cnt = 0
        result = [[], [], {}]
        for a in range(0, self.judger.task1_cnt):
            result[0].append({"TP": 0, "FP": 0, "TN": 0, "FN": 0})
        for a in range(0, self.judger.task2_cnt):
            result[1].append({"TP": 0, "FP": 0, "TN": 0, "FN": 0})
        result[2] = {"cnt": 0, "score": 0}

        with open(path, encoding="UTF-8") as f:
            for line in f.readlines():
                line = json.loads(line)
                ground_truth = line["meta"]
                fact = line["fact"]
                ans = predictor.predict(fact)
                cnt += 1
                result = self.judger.gen_new_result(result, ground_truth, ans[0])
                scores = self.judger.get_score(result)
        print(result)
        print(scores)


if __name__ == "__main__":
    train_path = "./Data/cail_0518_jieba_length/data_valid_small.json"
    valid_path = "./Data/cail_0518_jieba_length/data_valid_small.json"
    test_path = "./Data/cail_0518_jieba_length/data_valid_small.json"
    accusation_path = "./accu.txt"
    law_path = "./law.txt"

    Metrics(train_path=train_path, valid_path=valid_path, test_path=test_path,
            accusation_path=accusation_path, law_path=law_path)

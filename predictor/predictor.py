import json
# import thulac
from sklearn.externals import joblib


class Predictor(object):
    def __init__(self):
        self.tfidf = joblib.load('./predictor/model/tfidf.model')
        self.law = joblib.load('./predictor/model/law.model')
        self.accu = joblib.load('./predictor/model/accu.model')
        self.time = joblib.load('./predictor/model/time.model')
        self.batch_size = 1

        # self.cut = thulac.thulac(seg_only=True)

    def predict_law(self, vec):
        y = self.law.predict(vec)
        return [y[0] + 1]

    def predict_accu(self, vec):
        y = self.accu.predict(vec)
        return [y[0] + 1]

    def predict_time(self, vec):

        y = self.time.predict(vec)[0]

        # 返回每一个罪名区间的中位数
        if y == 0:
            return -2
        if y == 1:
            return -1
        if y == 2:
            return 120
        if y == 3:
            return 102
        if y == 4:
            return 72
        if y == 5:
            return 48
        if y == 6:
            return 30
        if y == 7:
            return 18
        else:
            return 6

    def predict(self, content):
        # fact = self.cut.cut(content[0], text=True)
        fact = content

        vec = self.tfidf.transform([fact])
        ans = {}

        ans['accusation'] = self.predict_accu(vec)
        ans['articles'] = self.predict_law(vec)
        ans['imprisonment'] = self.predict_time(vec)

        # print(ans)
        return [ans]


if __name__ == "__main__":
    predictor = Predictor()
    text = "公诉 机关 指控 ， 被告人 李 某某 于 2016 年 1 月 12 日 12 时许 ， 在 长春市 九台 区 工农 街 日杂 商店 附近 ， " \
           "从 被害人 陈 某某 衣兜 内 盗窃 一部 苹果 5s 手机 （ 价值 人民币 1620.00 元 ） 。 被告人 李 某某 于 2016 年 1" \
           " 月 30 日 13 时许 ， 在 九台 区 二道 街 中段 ， 从 被害人 于 某某 外 衣兜 内 盗窃 一部 白色 三星 note3 手机 （ " \
           "价值 人民币 500.00 元 ） 。 被告人 李 某某 于 2016 年 2 月 21 日 14 时许 ， 在 九台 区 第一 中学 附近 ， 从 " \
           "被害人 宋 某某 上 衣兜 内 盗窃 一部 苹果 6s - plus 手机 （ 价值 人民币 4 ， 950.00 元 ） 。 被告人 李 某某 于 " \
           "2016 年 2 月 24 日 14 时许 ， 在 九台 区 二道 街 中段 从孙 某某 上 衣兜 内 盗窃 一部 白色 步步高 vivo 手机 。 " \
           "（ 价值 人民币 300.00 元 ） 。 案发后 ， 赃物 已 返还 被害人 。 被告人 李 某某 于 2016 年 2 月 24 日 14 时许 ，" \
           " 在 九台 区 二道 街 实验 小学 对面 路上 从张 某某 兜内 盗窃 一部 白色 联想 808t 手机 。 （ 价值 人民币 200.00 元 " \
           "） 。 案发后 ， 赃物 已 返还 被害人 。 综上 ， 被告人 李 某某 共 盗窃 五 起 ， 价值 人民币 7570.00 元 。"
    predictor.predict(text)
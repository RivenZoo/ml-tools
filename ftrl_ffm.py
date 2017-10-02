# coding: utf-8

import math
import random

epoch = 4
alpha = 0.5
beta = 1
l1 = 0.01
l2 = 1
cross_field = {0: [1]}


def exp(v):
    v = 20 if v > 20 else v
    v = -20 if v < -20 else v
    return math.exp(v)


def sigmod(v):
    return 1 / (1 + exp(-v))


def ln(v):
    if v <= 0:
        return -999
    return math.log(v)


class Sample:

    def __init__(self, y, features=[]):
        '''
        :param y: 1 or -1
        :param features: FeatureItem list
        '''
        self.features = features
        self.y = y

    class FeatureItem:
        _invalid_field = -1

        def __init__(self, field, column, val):
            self._field = field
            self._column = column
            self._val = val

        def invalid_field(self):
            return self._field == Sample.FeatureItem._invalid_field

        @property
        def field(self):
            return self._field

        @property
        def feature_id(self):
            return self._column

        @property
        def feature_value(self):
            return self._val

        @staticmethod
        def parse(s):
            idx = s.find(':')
            if idx == -1:
                return None
            i = int(s[:idx])
            begin = idx + 1
            idx = s.find(':', idx + 1)
            if idx == -1:
                v = float(s[begin:])
                return Sample.FeatureItem(Sample.FeatureItem._invalid_field,
                                          i, v)
            c = int(s[begin:idx])
            v = float(s[idx + 1:])
            return Sample.FeatureItem(i, c, v)

    @staticmethod
    def parseLine(line):
        '''
        :param line: 1,0:21:0.5 or 0,1:8:1.0
        :return: Sample
        '''
        idx = line.find(',')
        if idx == -1:
            return None
        y = int(line[:idx])
        if y == 0:
            y = -1
        features = []
        begin = idx + 1
        idx = line.find(',', idx + 1)
        while idx != -1:
            item = Sample.FeatureItem.parse(line[begin: idx])
            if item:
                features.append(item)
            begin = idx + 1
            idx = line.find(',', idx + 1)
        if begin < len(line):
            item = Sample.FeatureItem.parse(line[begin:])
            if item:
                features.append(item)
        return Sample(y, features)


def test_sample():
    feature_str = '0:10:0.5'
    item = Sample.FeatureItem.parse(feature_str)
    assert item.field == 0
    assert item.feature_id == 10
    assert item.feature_value == 0.5
    feature_str = '10:0.5'
    item = Sample.FeatureItem.parse(feature_str)
    assert item.invalid_field()
    assert item.feature_id == 10
    assert item.feature_value == 0.5
    sample_str = '0,0:10:0.5,1:11:1'
    sample = Sample.parseLine(sample_str)
    assert len(sample.features) == 2
    assert sample.y == -1


class FtrlFFM:

    def __init__(self, alpha, beta, l1, l2, k=4):
        self.alpha = alpha
        self.beta = beta
        self.l1 = l1
        self.l2 = l2
        self.k = k
        self._z = {}
        self._q = {}
        # cross field f1:[f2,f3]
        self.cross_field = cross_field
        # V_ifj _cross_z[i][fj] = [0,1,...k-1]
        self._cross_z = {}
        self._cross_q = {}

    def save_model(self, output):
        with open(output, 'w') as f:
            for feature_id in self._z:
                w = self.get_weight(feature_id)
                if w != 0:
                    f.write('%s\t%s\n' % (feature_id, w))
        with open(output + '.ffm', 'w') as f:
            for feature_id in self._cross_z:
                v_s = []
                w_sum = 0
                for field in self._cross_z[feature_id]:
                    w = self.get_cross_weight(feature_id, field)
                    w_sum += sum(w)
                    v_s.append(','.join(['%s' % (i,) for i in w]))
                if w_sum != 0:
                    f.write('%s\t%s\n' % (feature_id, '\t'.join(v_s)))

    def calc_weight(self, zi, qi):
        sign = 1
        if zi < 0:
            sign = -1
        if sign * zi < self.l1:
            return 0
        else:
            return (self.l1 * sign - zi) / \
                   (self.l2 + (self.beta + math.sqrt(qi)) / self.alpha)

    def get_weight(self, idx):
        return self.calc_weight(self._z.get(idx, 0), self._q.get(idx, 0))

    def get_cross_weight(self, i, fj):
        zi = self._cross_z.get(i, None)
        if not zi or fj not in zi:
            z = [random.random() for j in range(self.k)]
            q = [random.random() for j in range(self.k)]
            self._cross_z.setdefault(i, {})
            self._cross_z[i].setdefault(fj, z)
            self._cross_q.setdefault(i, {})
            self._cross_q[i].setdefault(fj, q)
            zi = self._cross_z[i]
        z_ifj = zi[fj]
        q_ifj = self._cross_q[i][fj]
        return [self.calc_weight(z_ifj[j], q_ifj[j]) for j in range(self.k)]

    def calc_cross(self, feature_i, feature_j):
        vi = self.get_cross_weight(feature_i.feature_id, feature_j.field)
        vj = self.get_cross_weight(feature_j.feature_id, feature_i.field)
        if not vi or not vj:
            return None
        return sum([vi[i] * vj[i] for i in range(self.k)]) \
            * feature_i.feature_value * feature_j.feature_value

    def y_hat(self, field_features):
        w0 = self.get_weight(0)
        calc = w0
        for field, features in field_features.iteritems():
            for feature in features:
                wi = self.get_weight(feature.feature_id)
                if wi == 0:
                    continue
                calc += wi * feature.feature_value
        for f1, flist in self.cross_field.iteritems():
            feature_i = field_features.get(f1, None)
            if not feature_i:
                continue
            for f2 in flist:
                feature_j = field_features.get(f2, None)
                if not feature_j:
                    continue
                for fi in feature_i:
                    for fj in feature_j:
                        calc += self.calc_cross(fi, fj)
        return calc

    def loss(self, y_hat, y):
        return ln(1 + exp(-y_hat * y))

    def partial_guidance(self, feature=None):
        # None for w0
        if not feature:
            return 1
        # w_l
        return feature.feature_value

    def partial_guidance_cross(self, feature, cross_field_features):
        if feature.invalid_field() or len(cross_field_features) == 0:
            return None
        guidance = [0] * self.k
        for cross_feature in cross_field_features:
            v_s = self.get_cross_weight(
                cross_feature.feature_id, feature.field)
            for k in range(self.k):
                guidance[k] += v_s[k] * cross_feature.feature_value
            guidance = map(lambda x: feature.feature_value * x, guidance)
        return guidance

    def gradient(self, y_hat, y, partial_guidance):
        if type(partial_guidance) == float or type(partial_guidance) == int:
            return y * (sigmod(y_hat * y) - 1) * partial_guidance
        return map(lambda x: x * (y * (sigmod(y_hat * y) - 1)),
                   partial_guidance)

    def update_gradient(self, feature_id, gi):
        self._z.setdefault(feature_id, 0.0)
        self._q.setdefault(feature_id, 0.0)
        w = self.get_weight(feature_id)
        g2 = math.pow(gi, 2)
        sigma = (math.sqrt(self._q[feature_id] + g2) -
                 math.sqrt(self._q[feature_id])) / self.alpha
        self._z[feature_id] += (gi - sigma * w)
        self._q[feature_id] += g2

    def update_cross_gradient(self, feature_id, field_j, g_c):
        self._cross_z.setdefault(feature_id, {})
        self._cross_z[feature_id].setdefault(
            field_j, [random.random() for i in range(self.k)])
        self._cross_q.setdefault(feature_id, {})
        self._cross_q[feature_id].setdefault(
            field_j, [random.random() for i in range(self.k)])
        z = self._cross_z[feature_id][field_j]
        q = self._cross_q[feature_id][field_j]
        w = self.get_cross_weight(feature_id, field_j)
        g_c_2 = map(lambda x: math.pow(x, 2), g_c)
        sigma = [(math.sqrt(q[k] + g_c_2[k]) - math.sqrt(q[k])) /
                 self.alpha for k in range(self.k)]
        for k in range(self.k):
            z[k] += (g_c[k] - sigma[k] * w[k])
            q[k] += g_c_2[k]
        self._cross_z[feature_id][field_j] = z
        self._cross_q[feature_id][field_j] = q

    def score(self, features):
        y_hat = self.y_hat(features)
        return sigmod(y_hat)

    def group_features_by_field(self, features):
        field_features = {}
        for feature in features:
            field_features.setdefault(feature.field, [])
            field_features[feature.field].append(feature)
        return field_features

    def fit(self, samples):
        loss = []
        for sample in samples:
            y = sample.y
            features = sample.features
            field_features = self.group_features_by_field(features)
            y_hat = self.y_hat(field_features)
            loss.append(self.loss(y_hat, y))

            g_0 = self.gradient(y_hat, y, self.partial_guidance())
            g_l = self.gradient(
                y_hat, y, [self.partial_guidance(f) for f in features])
            g_cross = {}
            for f1, flist in self.cross_field.iteritems():
                feature_i = field_features.get(f1, None)
                if not feature_i:
                    continue
                for f2 in flist:
                    feature_j = field_features.get(f2, None)
                    if not feature_j:
                        continue
                    for fi in feature_i:
                        g_cross.setdefault(fi.feature_id, {})
                        pg = self.partial_guidance_cross(fi, feature_j)
                        g_cross[fi.feature_id][
                            f2] = self.gradient(y_hat, y, pg)

            # update gradient
            self.update_gradient(0, g_0)
            for i in range(len(features)):
                self.update_gradient(features[i].feature_id, g_l[i])
            for feature_id, field_gradients in g_cross.iteritems():
                for field_id, g_c in field_gradients.iteritems():
                    self.update_cross_gradient(feature_id, field_id, g_c)
        return loss

    def predict(self, samples):
        ret = []
        for sample in samples:
            ret.append(self.score(sample.features))
        return ret


def test_train():
    ffm = FtrlFFM(alpha, beta, l1, l2)
    sample_str = '0,525243223:0.05059798,384291900:0.07346334'
    sample = Sample.parseLine(sample_str)
    loss = ffm.fit([sample])
    print 'loss ', loss


def train(ffm, sample_file, epoch):
    samples = []
    batch_size = 100
    trained_cnt = 0
    loss_sum = 0
    print 'start train epoch ', epoch
    with open(sample_file) as f:
        for line in f:
            sample = None
            try:
                sample = Sample.parseLine(line)
            except Exception:
                continue
            if sample:
                samples.append(sample)
            if len(samples) > batch_size:
                trained_cnt += batch_size
                loss = ffm.fit(samples)
                loss_sum += sum(loss)
                samples = []
                if trained_cnt > 0 and trained_cnt % 100 == 0:
                    print 'train %s samples, loss %f' % \
                        (trained_cnt, loss_sum / trained_cnt)
        if len(samples) > 0:
            ffm.fit(samples)
    print 'train %s samples, loss %f' % (trained_cnt, loss_sum / trained_cnt)


if __name__ == '__main__':
    import sys
    from sklearn.metrics import roc_auc_score

    sample_file = sys.argv[1]
    test_file = sys.argv[2]
    ffm = FtrlFFM(alpha, beta, l1, l2)
    for e in range(epoch):
        train(ffm, sample_file, e)
    print 'start eval'
    y = []
    pred = []
    with open(test_file) as f:
        for line in f:
            sample = Sample.parseLine(line)
            if sample:
                if sample.y == -1:
                    y.append(0)
                else:
                    y.append(sample.y)
                pred.append(
                    ffm.score(ffm.group_features_by_field(sample.features)))
    print 'test eval, ', roc_auc_score(y, pred)
    ffm.save_model(sample_file + '_model')

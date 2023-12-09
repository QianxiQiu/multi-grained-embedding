
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from copy import deepcopy


class GCForest:
    def __init__(self,   cascade_test_size=0.2, n_cascadeRF=2, n_cascadeRFtree=101, cascade_layer=8, min_sample_cascade=0.05, tolerance=0.0, n_jobs=-1):
        self.cascade_test_size = cascade_test_size
        self.n_cascadeRF = n_cascadeRF
        self.n_cascadeRFtree = n_cascadeRFtree
        self.cascade_layer = cascade_layer
        self.min_sample_cascade = min_sample_cascade
        self.tolerance = tolerance
        self.n_jobs = n_jobs

    def fit(self, X, y):
        self.cascade_forest(X,y)

    def predict_proba(self,X):
        cascade_all_pred_prob = self.cascade_forest(X)
        predict_proba = np.mean(cascade_all_pred_prob, axis=0)
        return predict_proba

    def predict(self,X):
        pred_prob = self.predict_proba(X)
        return np.argmax(pred_prob, axis=1)

    def cascade_forest(self, X, y=None):
        if y is not None:
            self.n_layer = 0
            tol = self.tolerance
            split_per = self.cascade_test_size
            max_layers = self.cascade_layer

            X_train, X_valid, y_train, y_valid = train_test_split(X,y, test_size=split_per)

            self.n_layer += 1
            prf_crf_pred_ref = self._cascade_layer(X_train, y_train)
            accuracy_ref = self._cascade_evaluation(X_valid, y_valid)
            feat_arr = self._create_feat_arr(X_train, prf_crf_pred_ref)

            self.n_layer += 1
            prf_crf_pred_layer = self._cascade_layer(feat_arr, y_train)
            accuracy_layer = self._cascade_evaluation(X_valid, y_valid)

            while accuracy_layer > (accuracy_ref+tol) and self.n_layer <= max_layers:
                accuracy_ref = accuracy_layer
                prf_crf_pred_ref = prf_crf_pred_layer
                feat_arr = self._create_feat_arr(X_train, prf_crf_pred_ref)
                self.n_layer += 1
                prf_crf_pred_layer = self._cascade_layer(feat_arr, y_train)
                accuracy_layer = self._cascade_evaluation(X_valid, y_valid)

            if accuracy_layer < accuracy_ref:
                n_cascadeRF = getattr(self, 'n_cascadeRF')
                for irf in range(n_cascadeRF):
                    delattr(self, '_casprf{}_{}'.format(self.n_layer, irf))
                    delattr(self, '_cascrf{}_{}'.format(self.n_layer, irf))
                self.n_layer -= 1


        elif y is None:
            at_layer = 1
            prf_crf_pred_ref = self._cascade_layer(X, layer=at_layer)
            while at_layer < getattr(self, 'n_layer'):
                at_layer += 1
                feat_arr = self._create_feat_arr(X, prf_crf_pred_ref)
                #print (feat_arr.shape)
                prf_crf_pred_ref = self._cascade_layer(feat_arr, layer=at_layer)

        return prf_crf_pred_ref


    def _cascade_layer(self, X, y=None, layer=0):

        n_tree = getattr(self, 'n_cascadeRFtree')
        n_cascadeRF = getattr(self, 'n_cascadeRF')
        min_samples = getattr(self, 'min_sample_cascade')

        n_jobs = getattr(self, 'n_jobs')

        prf_crf_pred = []
        if y is not None:
            #print('Adding/Training Layer, n_layer={}'.format(self.n_layer))
            for irf in range(n_cascadeRF):


                prf = RandomForestClassifier(n_estimators=n_tree, max_features='sqrt',
                                             min_samples_split=min_samples, oob_score=True, n_jobs=n_jobs
                                             )
                crf = RandomForestClassifier(n_estimators=n_tree, max_features=1,
                                             min_samples_split=min_samples, oob_score=True, n_jobs=n_jobs
                                             )
                prf.fit(X, y)
                crf.fit(X, y)
                setattr(self, '_casprf{}_{}'.format(self.n_layer, irf), deepcopy(prf))
                setattr(self, '_cascrf{}_{}'.format(self.n_layer, irf), deepcopy(crf))
                prf_crf_pred.append(prf.oob_decision_function_)
                prf_crf_pred.append(crf.oob_decision_function_)
        elif y is None:
            for irf in range(n_cascadeRF):
                prf = getattr(self, '_casprf{}_{}'.format(layer, irf))
                crf = getattr(self, '_cascrf{}_{}'.format(layer, irf))
                prf_crf_pred.append(prf.predict_proba(X))
                prf_crf_pred.append(crf.predict_proba(X))

        return prf_crf_pred



    def _cascade_evaluation(self, X_valid, y_valid):
        casc_pred_prob = np.mean(self.cascade_forest(X_valid), axis=0)
        casc_pred = np.argmax(casc_pred_prob, axis=1)
        casc_accuracy = accuracy_score(y_true=y_valid, y_pred=casc_pred)
        return casc_accuracy

    def _create_feat_arr(self, X, prf_crf_pred):
        swap_pred = np.swapaxes(prf_crf_pred, 0, 1)
        add_feat = swap_pred.reshape([np.shape(X)[0], -1])
        feat_arr = np.concatenate([add_feat, X], axis=1)
        return feat_arr


def main(data_train,data_test,target_train,target_test):

    for i, target in enumerate(np.unique(target_train)):
        target_train[target_train == target] = int(i)
        target_test[target_test == target] = int(i)
    target_train=target_train.astype(int).values
    target_test=target_test.astype(int).values

    gcf = GCForest(tolerance=0.0)
    gcf.fit(data_train, target_train)

    pred_y_test = gcf.predict(data_test)
    accuracy = accuracy_score(y_true=target_test, y_pred=pred_y_test)

    return accuracy







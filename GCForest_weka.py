import copy
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.metrics import accuracy_score
import pandas as pd

import jpype
from jpype import *

#the method to generate the header is as follows: merge the training and test sets (if only the training set is used, the header may encounter out-of-bag data), then convert it to .arff format using Weka.
dic_arff_head={
               "tic-tac-toe":["{x,o,b}","{x,o,b}","{x,o,b}","{x,o,b}","{o,b,x}","{o,b,x}","{x,o,b}","{o,x,b}","{o,x,b}","{positive,negative}"],
               "vote":["{n,y}","{y,n}","{n,y}","{y,n}","{y,n}","{y,n}","{n,y}","{n,y}","{n,y}","{y,n}","{n,y}","{y,n}","{y,n}","{y,n}","{n,y}","{y,n}","{republican,democrat}"],
               "balance-scale":["numeric","numeric","numeric","numeric","{B,R,L}"],
               "dermatology":["numeric","numeric","numeric","numeric","numeric", "numeric","numeric","numeric","numeric","numeric","numeric","numeric","numeric","numeric","numeric","numeric","numeric","numeric","numeric","numeric","numeric","numeric","numeric","numeric","numeric","numeric","numeric","numeric","numeric","numeric","numeric","numeric","numeric","numeric","{2a,1a,3a,5a,4a,6a}"],
               "Disease_symptom":["{Yes,No}","{No,Yes}","{Yes,No}","{Yes,No}","numeric","{Female,Male}","{Low,Normal,High}","{Normal,Low,High}","{Positive,Negative}"],
               "jnaERA":["numeric","numeric","numeric","numeric","{4a,2a,7a,6a,5a,3a,9a,8a,1a}"],
               "jnaESL":["numeric","numeric","numeric","numeric","{6a,5a,4a,3a,2a,7a,8a}"],
               "jnaLEV":["numeric","numeric","numeric","numeric","{3a,2a,0a,4a,1a}"],
               "nursery":["numeric","numeric","numeric","numeric","numeric","{convenient,inconv}","numeric","numeric","{priority,not_recom,very_recom,spec_prior}"],
               "car":["numeric","numeric","numeric","numeric","numeric","numeric","{unacc,acc,vgood,good}"],
                "diabetes": ["{Female,Male,Other}", "numeric", "{No,YEs}", "{YEs,No}",
                             "{never,'No_Info',current,former,ever,'not_current'}", "numeric", "numeric", "numeric", "{No,YEs}"],
                "letter": ["numeric", "numeric", "numeric", "numeric", "numeric", "numeric", "numeric", "numeric", "numeric",
                           "numeric", "numeric", "numeric", "numeric", "numeric", "numeric", "numeric",
                           "{Z,P,S,H,F,N,R,M,D,V,A,K,E,O,Q,L,X,Y,I,W,U,T,C,G,B,J}"]

}

is_encoded=None


class GCForest:
    def __init__(self,   cascade_test_size=0.2, cascade_layer=5, min_sample_cascade=0.05, tolerance=0.0, dataname=None):
        self.cascade_test_size = cascade_test_size
        self.cascade_layer = cascade_layer
        self.min_sample_cascade = min_sample_cascade
        self.tolerance = tolerance
        self.dataname=dataname


    def fit(self, X, y):
        self.cascade_forest(X,y,if_y_is_logical_use=True)

    def predict_proba(self,X,y):
        cascade_all_pred_prob,accu_list = self.cascade_forest(X,y,if_y_is_logical_use=False)
        predict_proba = np.mean(cascade_all_pred_prob, axis=0)
        return predict_proba

    def predict(self,X,y):
        pred_prob = self.predict_proba(X,y)
        return np.argmax(pred_prob, axis=1)


    def cascade_forest(self, X, y,if_y_is_logical_use):
        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)
        if if_y_is_logical_use:
            self.n_layer = 0
            tol = self.tolerance
            split_per = self.cascade_test_size
            max_layers = self.cascade_layer

            split=StratifiedShuffleSplit(n_splits=1,test_size=split_per)
            for train_index,valid_index in split.split(X,y):
                X_train=X.loc[train_index]
                X_valid=X.loc[valid_index]
                y_train=y.loc[train_index]
                y_valid=y.loc[valid_index]

            self.n_layer += 1
            prf_crf_pred_ref,accu_list = self._cascade_layer(X_train, y_train,if_y_is_logical_use=True)
            accuracy_ref = self._cascade_evaluation(X_valid, copy.deepcopy(y_valid))
            feat_arr = self._create_feat_arr(X_train, prf_crf_pred_ref)

            self.n_layer += 1
            prf_crf_pred_layer,accu_list = self._cascade_layer(feat_arr, y_train,if_y_is_logical_use=True)
            accuracy_layer = self._cascade_evaluation(X_valid, copy.deepcopy(y_valid))

            while accuracy_layer > (accuracy_ref+tol) and self.n_layer <= max_layers:
                accuracy_ref = accuracy_layer
                prf_crf_pred_ref = prf_crf_pred_layer
                feat_arr = self._create_feat_arr(X_train, prf_crf_pred_ref)
                self.n_layer += 1
                prf_crf_pred_layer,accu_list = self._cascade_layer(feat_arr, y_train,if_y_is_logical_use=True)
                accuracy_layer = self._cascade_evaluation(X_valid, copy.deepcopy(y_valid))

            if accuracy_layer < accuracy_ref:
                self.n_layer -= 1

        else:
            at_layer = 1
            prf_crf_pred_ref,accu_list = self._cascade_layer(X,y ,if_y_is_logical_use=False,layer=at_layer)
            while at_layer < getattr(self, 'n_layer'):
                at_layer += 1
                feat_arr = self._create_feat_arr(X, prf_crf_pred_ref)
                prf_crf_pred_ref,accu_list = self._cascade_layer(feat_arr,y,if_y_is_logical_use=False, layer=at_layer)

        return prf_crf_pred_ref,accu_list


    def csv_to_arff(self,csvpath,df):
        global dic_arff_head
        arffpath=csvpath[:csvpath.find(".csv")]+".arff"
        f=open(arffpath,"w+")
        #Write ARFF file header
        f.write("@relation {}\n\n".format(arffpath[14:-5]))
        index_original_attr=0
        global is_encoded
        for col in range(df.shape[1]):
            #The encoded dataset is sure to have all numeric features except for the label column.
            if is_encoded:
                if col == df.shape[1]-1:
                    # label (should be place in the last column)
                    f.write("@attribute {} {}\n".format(col,dic_arff_head[self.dataname][-1]))
                else:
                    # features
                    f.write("@attribute {} numeric\n".format(col))
            else:
                if col >= (df.shape[1] - len(dic_arff_head[self.dataname])):
                    f.write("@attribute {} {}\n".format(col, dic_arff_head[self.dataname][index_original_attr]))
                    index_original_attr = index_original_attr + 1
                else:
                    f.write("@attribute {} numeric\n".format(col))
        # Write ARFF file data
        f.write("\n@data\n")
        for i in range(df.shape[0]):
            line=df.iloc[i,:].values
            mystr=""
            for j in range(df.shape[1]):
                mytype=type(line[j])
                if isinstance(line,np.str_):
                    mystr = mystr + line[j] + ","
                else:
                    mystr = mystr + np.str_(line[j]) + ","

            mystr=mystr[:-1]
            mystr=mystr+"\n"
            f.write(mystr)
        f.close()
        return arffpath



    def _cascade_layer(self, X, y,if_y_is_logical_use, layer=0):
        X=X.reset_index(drop=True)
        y=y.reset_index(drop=True)
        n_cascadeRF = getattr(self, 'n_cascadeRF')


        prf_crf_pred = []
        accu_list=[]
        if if_y_is_logical_use:
            for irf in range(n_cascadeRF):
                train = pd.concat([X, y], axis=1)
                train.columns = np.arange(train.shape[1])

                path_train="temp_GCForest/temp_train_" +"_"+str(self.n_layer)+"_"+str(irf)+ ".csv"
                path_train=self.csv_to_arff(path_train,train)


                Integer = java.lang.Integer
                opt = JClass("MyRandomForest")

                prf=opt()

                crf=opt()
                crf.setParam_random_seed(Integer(2))

                path_prf_model="model_GCForest_randomForest/model_prf"+"_"+str(self.n_layer)+"_"+str(irf)+".model"
                path_crf_model = "model_GCForest_randomForest/model_crf" + "_" + str(self.n_layer) + "_" + str(irf)+".model"
                prf.fit(path_train,path_prf_model)
                crf.fit(path_train,path_crf_model)


                path_result_prf="temp_GCForest/temp_proba_train_prf_"  +"_"+str(self.n_layer)+"_"+str(irf)+ ".csv"
                path_result_crf = "temp_GCForest/temp_proba_train_crf_"  + "_" +str(self.n_layer)+"_"+str(irf)+ ".csv"

                a=prf.predict_proba(path_train, path_result_prf,path_prf_model)
                b=crf.predict_proba(path_train, path_result_crf,path_crf_model)


                proba_prf = pd.read_csv(path_result_prf,header=None)
                proba_crf = pd.read_csv(path_result_crf,header=None)


                prf_crf_pred.append(proba_prf)
                prf_crf_pred.append(proba_crf)
                accu_list.append(a)
                accu_list.append(b)
        else:

            for irf in range(n_cascadeRF):
                # prf = getattr(self, '_casprf{}_{}'.format(layer, irf))
                # crf = getattr(self, '_cascrf{}_{}'.format(layer, irf))
                Integer = java.lang.Integer
                opt = JClass("MyRandomForest")

                prf = opt()

                crf = opt()
                crf.setParam_random_seed(Integer(2))


                test = pd.concat([X, y], axis=1)
                test.columns = np.arange(test.shape[1])
                path_test = "temp_GCForest/temp_test_"  + "_" +str(layer)+"_"+str(irf)+ ".csv"

                path_test=self.csv_to_arff(path_test, test)

                path_result_prf = "temp_GCForest/temp_proba_test_prf_"  + "_" +str(layer)+"_"+str(irf)+ ".csv"
                path_result_crf = "temp_GCForest/temp_proba_test_crf_"  + "_" +str(layer)+"_"+str(irf)+ ".csv"

                path_prf_model = "model_GCForest_randomForest/model_prf" + "_" + str(layer) + "_" + str(irf)+".model"
                path_crf_model = "model_GCForest_randomForest/model_crf" + "_" + str(layer) + "_" + str(irf)+".model"

                a=prf.predict_proba(path_test, path_result_prf,path_prf_model)
                b=crf.predict_proba(path_test, path_result_crf,path_crf_model)

                proba_prf = pd.read_csv(path_result_prf, header=None)
                proba_crf = pd.read_csv(path_result_crf, header=None)

                prf_crf_pred.append(proba_prf)
                prf_crf_pred.append(proba_crf)
                accu_list.append(a)
                accu_list.append(b)

        return prf_crf_pred,accu_list



    def _cascade_evaluation(self, X_valid, y_valid):
        predict_proba_list,accu_list=self.cascade_forest(X_valid,y_valid,if_y_is_logical_use=False)
        casc_pred_prob = np.mean(predict_proba_list, axis=0)
        casc_pred = np.argmax(casc_pred_prob, axis=1)
        y_valid_int=trans_to_int(y_valid,self.dataname)

        casc_accuracy = accuracy_score(y_true=y_valid_int, y_pred=casc_pred)

        return casc_accuracy

    def _create_feat_arr(self, X, prf_crf_pred):
        swap_pred = np.swapaxes(prf_crf_pred, 0, 1)
        add_feat = swap_pred.reshape([np.shape(X)[0], -1])
        df_add_feat = pd.DataFrame(add_feat, index=None, columns=None)
        X=X.reset_index(drop=True)
        feat_arr = pd.concat([df_add_feat, X], axis=1)
        feat_arr.columns=np.arange(feat_arr.shape[1])

        return feat_arr

def trans_to_int(target_test,dataname):
    global dic_arff_head
    targers_str=dic_arff_head[dataname][-1]
    targers_str=targers_str[1:-1]
    value_list=targers_str.split(",")
    target_test_int=copy.deepcopy(target_test)
    target_test_int.columns=[0]
    for index, value in enumerate(value_list):
        target_test_int.loc[target_test_int.iloc[:, 0] == value, 0] = index
    target_test_int = target_test_int.astype(int).values
    return target_test_int

def main(data_train,data_test,target_train,target_test,dataname,if_is_encoded=False):

    target_train=pd.DataFrame(target_train)
    target_test = pd.DataFrame(target_test)
    global is_encoded
    is_encoded=if_is_encoded

    gcf = GCForest(tolerance=0.0,dataname=dataname)
    gcf.fit(data_train, target_train)

    pred_y_test = gcf.predict(data_test,target_test)

    target_test_int=trans_to_int(target_test,dataname)

    accuracy = accuracy_score(y_true=target_test_int, y_pred=pred_y_test)

    return accuracy







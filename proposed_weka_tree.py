import sys
import numpy as np
import pandas as pd
import random
import torch
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
import copy
from collections import Counter

import jpype
from jpype import *

from torch import nn

mydataname=None

valid_fold=None


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
                "diabetes":["{Female,Male,Other}","numeric","{No,YEs}","{YEs,No}","{never,'No_Info',current,former,ever,'not_current'}","numeric","numeric","numeric","{No,YEs}"],
                "letter":["numeric","numeric","numeric","numeric","numeric","numeric","numeric","numeric","numeric","numeric","numeric","numeric","numeric","numeric","numeric","numeric","{Z,P,S,H,F,N,R,M,D,V,A,K,E,O,Q,L,X,Y,I,W,U,T,C,G,B,J}"]
               }

class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'w')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

#This class is the first part of the proposed encoding method: Divide granularity and calculate probability distribution:
class embed_trans_to_proba:
    def __init__(self,params=None):
        self.params=params
        if params=={}:
            self.params=None


    def find_most_relative_feature(self,now,indexes):
        max_score=0
        max_index=0
        for i in indexes:
            score=metrics.mutual_info_score(self.x_train.iloc[:,now],self.x_train.iloc[:,i])
            if score>max_score:
                max_score=score
                max_index=i
        if max_score==0 :
            max_index=random.choice(indexes)
        return max_index

    def divide(self, indexes):
        # Save the subset corresponding to the current granularity division.
        self.subsets.append(copy.deepcopy(indexes))

        # Recursive termination condition
        if len(indexes) == 1:
            return

        # Split features into two parts based on mutual information:
        s1 = []
        s2 = []
        now = random.choice(indexes)
        s1.append(now)
        flag_s1 = False
        indexes.remove(now)
        while len(indexes) != 0:
            now = self.find_most_relative_feature(now, indexes)
            if flag_s1:
                s1.append(now)
                flag_s1 = False
            else:
                s2.append(now)
                flag_s1 = True
            indexes.remove(now)

        # Recursively partition the two feature subsets
        self.divide(s1)
        self.divide(s2)


    def csv_to_arff(self,csvpath,df,index_subset):
        global dic_arff_head
        global mydataname
        arffpath=csvpath[:csvpath.find(".csv")]+".arff"
        f=open(arffpath,"w+")
        #Write ARFF file header
        f.write("@relation {}\n\n".format(arffpath[28:-5]))
        index_subsets_col=0
        #features
        for col in index_subset:
            f.write("@attribute {} {}\n".format(index_subsets_col, dic_arff_head[mydataname][col]))
            index_subsets_col = index_subsets_col + 1
        # label (should be place in the last column)
        f.write("@attribute {} {}\n".format(index_subsets_col, dic_arff_head[mydataname][-1]))
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

    def fit_one_grain(self,i,subsets):

        global mydataname

        # Merge 'data' and 'target', and save it to a CSV file for convenient conversion to ARFF format.
        train = pd.concat([self.x_train.iloc[:, subsets], self.y_train], axis=1)
        train.columns = np.arange(train.shape[1])
        train = train.reset_index(drop=True)
        path_train = "temp_proposed_optimizedtree/temp_train_" + mydataname + str(i) + ".csv"
        path_train = self.csv_to_arff(path_train, train, subsets)

        # Merge 'data' and 'target', and save it to a CSV file for convenient conversion to ARFF format.
        test = pd.concat([self.x_test.iloc[:, subsets], self.y_test], axis=1)
        test.columns = np.arange(test.shape[1])
        test = test.reset_index(drop=True)
        path_test = "temp_proposed_optimizedtree/temp_test_" + mydataname + str(i) + ".csv"
        path_test = self.csv_to_arff(path_test, test, subsets)

        #Create a java class named MyRandomForest, which is included in the MyRandomForest.jar
        opt = JClass("MyRandomForest")
        myopt = opt()

        if self.params is not None:
            if "num_inside_tree" in self.params.keys():
                myopt.setParam_num_inside_tree(JInt(self.params["num_inside_tree"]))
            if "max_depth" in self.params.keys():
                myopt.setParam_max_depth(JInt(self.params["max_depth"]))

        path_model = "model_multiGrain_randomForest/model" + "_" + str(i) + ".model"

        myopt.fit(JString(path_train),JString(path_model))

        path_result_train = "temp_proposed_optimizedtree/temp_proba_train" + mydataname + str(i) + ".csv"
        path_result_test = "temp_proposed_optimizedtree/temp_proba_test" + mydataname + str(i) + ".csv"

        a = myopt.predict_proba(path_train, path_result_train,path_model)
        b = myopt.predict_proba(path_test, path_result_test,path_model)

        proba_train = pd.read_csv(path_result_train, header=None)
        proba_test = pd.read_csv(path_result_test, header=None)

        return proba_train, proba_test

    def trans_to_proba(self):
        self.subsets_probas_train = []
        self.subsets_probas_test = []

        for i, subsets in enumerate(self.subsets):
            proba_train, proba_test=self.fit_one_grain(i, subsets)
            self.subsets_probas_train.append(proba_train)
            self.subsets_probas_test.append(proba_test)

        self.subsets_probas_train = np.array(self.subsets_probas_train)
        self.subsets_probas_test = np.array(self.subsets_probas_test)



    def get_predict_test(self):
        return self.precision


    def fit_and_transform(self,x_train,x_test,y_train,y_test,num_target):
        self.x_train=x_train
        self.x_test=x_test
        self.y_train=y_train
        self.y_test=y_test

        self.labels_nums=num_target
        self.subsets = []
        cols=x_train.shape[1]
        indexes=np.arange(cols).tolist()

        self.divide(copy.deepcopy(indexes))
        self.trans_to_proba()

        return self.subsets_probas_train, self.subsets_probas_test, self.y_train, self.y_test



#This class is the second part of the proposed encoding method:Adjusting temperature parameter:
class embed_temper_control:
    def __init__(self,iterations=1000,learningRate=1):
        self.iterations=iterations
        self.learningRate=learningRate


    def temper_control(self,subsets_probas_train,subsets_probas_test,y_train,y_test,types_target,lr=None,flag_save=False,parallel="single_gpu"):
        if parallel =="multi_gpu":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            if lr is None:
                lr=self.learningRate

            y_train = torch.tensor(y_train, dtype=torch.long, device=device)
            y_test = torch.tensor(y_test, dtype=torch.long, device=device)

            subsets_probas_train=torch.tensor(subsets_probas_train, device=device)
            subsets_probas_test = torch.tensor(subsets_probas_test, device=device)

            tempers=torch.rand(size=(1,subsets_probas_train.shape[0]),requires_grad=True, device=device)

            model=torch.nn.Sequential(
                torch.nn.Linear(len(subsets_probas_train) * types_target.shape[0], types_target.shape[0]),
                torch.nn.LogSoftmax(dim=1),
            ).to(device)

            if torch.cuda.device_count() > 1:
                model = nn.DataParallel(model)

            cost=torch.nn.NLLLoss()
            optimizer=torch.optim.SGD(model.parameters(),lr=lr)

            for j in range(self.iterations):
                total_temper_softmax=torch.tensor(np.array([]),requires_grad=True,dtype=torch.float, device=device)

                for i,subset in enumerate(subsets_probas_train):
                    subsets_temper_softmax=torch.softmax(subset/tempers[0,i],dim=1)
                    total_temper_softmax=torch.cat((total_temper_softmax,subsets_temper_softmax),dim=1)
                total_temper_softmax = total_temper_softmax.to(torch.float32)
                predict = model(total_temper_softmax)
                loss = cost(predict, y_train)
                loss.backward()

                tempers.data.add_(-(lr * tempers.grad.data))
                optimizer.step()

                tempers.grad.data.zero_()
                optimizer.zero_grad()

                if j == (self.iterations-1) :
                    total_temper_softmax_test = torch.tensor(np.array([]), requires_grad=True, dtype=torch.float, device=device)
                    for i,subset in enumerate(subsets_probas_test):
                        subsets_temper_softmax=torch.softmax(subset/tempers[0,i],dim=1)
                        total_temper_softmax_test=torch.cat((total_temper_softmax_test,subsets_temper_softmax),dim=1)
                    total_temper_softmax_test = total_temper_softmax_test.to(torch.float32)
                    outputs = model(total_temper_softmax_test)
                    a,predict_test=torch.max(outputs,1)
                    array_predict_test=predict_test.data.cpu().detach().numpy()
                    conut_true=np.sum(array_predict_test==y_test.data.cpu().detach().numpy())
                    precision=conut_true/len(array_predict_test)

            if flag_save==True :
                return precision,total_temper_softmax.cpu().detach().numpy(),total_temper_softmax_test.cpu().detach().numpy()
            else:
                return precision

        elif parallel=="cpu":
            if lr is None:
                lr = self.learningRate
            y_train = torch.tensor(y_train, dtype=torch.long)
            y_test = torch.tensor(y_test, dtype=torch.long)

            subsets_probas_train = torch.tensor(subsets_probas_train)
            subsets_probas_test = torch.tensor(subsets_probas_test)

            tempers = torch.rand(size=(1, subsets_probas_train.shape[0]), requires_grad=True)

            model = torch.nn.Sequential(
                torch.nn.Linear(len(subsets_probas_train) * types_target.shape[0], types_target.shape[0]),
                torch.nn.LogSoftmax(dim=1),
            )

            cost = torch.nn.NLLLoss()
            optimizer = torch.optim.SGD(model.parameters(), lr=lr)

            for j in range(self.iterations):
                total_temper_softmax = torch.tensor(np.array([]), requires_grad=True, dtype=torch.float)

                for i, subset in enumerate(subsets_probas_train):
                    subsets_temper_softmax = torch.softmax(subset / tempers[0, i], dim=1)
                    total_temper_softmax = torch.cat((total_temper_softmax, subsets_temper_softmax), dim=1)
                total_temper_softmax = total_temper_softmax.to(torch.float32)
                predict = model(total_temper_softmax)
                loss = cost(predict, y_train)
                loss.backward()

                tempers.data.add_(-(lr * tempers.grad.data))
                optimizer.step()

                tempers.grad.data.zero_()
                optimizer.zero_grad()

                if j == (self.iterations - 1):
                    total_temper_softmax_test = torch.tensor(np.array([]), requires_grad=True, dtype=torch.float)
                    for i, subset in enumerate(subsets_probas_test):
                        subsets_temper_softmax = torch.softmax(subset / tempers[0, i], dim=1)
                        total_temper_softmax_test = torch.cat((total_temper_softmax_test, subsets_temper_softmax),
                                                              dim=1)
                    total_temper_softmax_test = total_temper_softmax_test.to(torch.float32)
                    outputs = model(total_temper_softmax_test)
                    a, predict_test = torch.max(outputs, 1)
                    array_predict_test = predict_test.data.detach().numpy()
                    conut_true = np.sum(array_predict_test == y_test.data.detach().numpy())
                    precision = conut_true / len(array_predict_test)

            if flag_save == True:
                return precision, total_temper_softmax.detach().numpy(), total_temper_softmax_test.detach().numpy()
            else:
                return precision

        elif parallel =="single_gpu":
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

            if lr is None:
                lr = self.learningRate
            y_train = torch.tensor(y_train, dtype=torch.long, device=device)
            y_test = torch.tensor(y_test, dtype=torch.long, device=device)

            subsets_probas_train = torch.tensor(subsets_probas_train, device=device)
            subsets_probas_test = torch.tensor(subsets_probas_test, device=device)

            tempers = torch.rand(size=(1, subsets_probas_train.shape[0]), requires_grad=True, device=device)

            model = torch.nn.Sequential(
                torch.nn.Linear(len(subsets_probas_train) * types_target.shape[0], types_target.shape[0]),
                torch.nn.LogSoftmax(dim=1),
            ).to(device)

            cost = torch.nn.NLLLoss()
            optimizer = torch.optim.SGD(model.parameters(), lr=lr)

            for j in range(self.iterations):
                total_temper_softmax = torch.tensor(np.array([]), requires_grad=True, dtype=torch.float, device=device)

                for i, subset in enumerate(subsets_probas_train):
                    subsets_temper_softmax = torch.softmax(subset / tempers[0, i], dim=1)
                    total_temper_softmax = torch.cat((total_temper_softmax, subsets_temper_softmax), dim=1)
                total_temper_softmax = total_temper_softmax.to(torch.float32)
                predict = model(total_temper_softmax)
                loss = cost(predict, y_train)
                loss.backward()

                tempers.data.add_(-(lr * tempers.grad.data))
                optimizer.step()

                tempers.grad.data.zero_()
                optimizer.zero_grad()

                if j == (self.iterations - 1):
                    total_temper_softmax_test = torch.tensor(np.array([]), requires_grad=True, dtype=torch.float,
                                                             device=device)
                    for i, subset in enumerate(subsets_probas_test):
                        subsets_temper_softmax = torch.softmax(subset / tempers[0, i], dim=1)
                        total_temper_softmax_test = torch.cat((total_temper_softmax_test, subsets_temper_softmax),
                                                              dim=1)
                    total_temper_softmax_test = total_temper_softmax_test.to(torch.float32)
                    outputs = model(total_temper_softmax_test)
                    a, predict_test = torch.max(outputs, 1)
                    array_predict_test = predict_test.data.cpu().detach().numpy()
                    conut_true = np.sum(array_predict_test == y_test.data.cpu().detach().numpy())
                    precision = conut_true / len(array_predict_test)

            if flag_save == True:
                return precision, total_temper_softmax.cpu().detach().numpy(), total_temper_softmax_test.cpu().detach().numpy()
            else:
                return precision



    def fit_and_transform(self,x_train,x_test,y_train,y_test,types_target):
        for i,target in enumerate(types_target) :
            y_train[y_train==target]=int(i)
            y_test[y_test == target] =int(i)

        #Convert the numeric data from string format to int format
        y_train = list(map(int, y_train))
        y_test = list(map(int, y_test))

        precsion, x_train_encoded, x_test_encoded = self.temper_control(copy.deepcopy(x_train),
                                                                        copy.deepcopy(x_test),
                                                                        copy.deepcopy(y_train),
                                                                        copy.deepcopy(y_test),
                                                                        types_target, self.learningRate,True)

        return precsion,x_train_encoded,x_test_encoded


def get_most_freq_value(list):
    if Counter(list).most_common(1)[0][1]==1:
        return random.choice(list)
    else:
        return Counter(list).most_common(1)[0][0]

def random_choice_params(iters,params_list):
    if len(params_list)>iters:
        return np.random.choice(params_list,iters,replace=False)
    else:
        result1=copy.deepcopy(params_list)
        random.shuffle(result1)
        result2=np.random.choice(params_list,(iters-len(params_list)),replace=True).tolist()
        return result1 + result2

def find_best_params(iter,train,valid,target_train,target_valid,num_target,types_target):
    num_inside_tree_list = [70, 100, 150, 200, 250, 300]
    max_depth_list = [4, 6, 8, 10, 12]

    lr_list = [1, 0.7, 0.4, 0.1, 0.07, 0.04, 0.01]
    random_search_num_inside_tree_list=random_choice_params(iter,num_inside_tree_list)
    random_search_max_depth_list=random_choice_params(iter,max_depth_list)

    best_num_inside_tree=None
    best_max_depth=None
    best_lr = None

    # Compare with default hyperparameters:
    embed = embed_trans_to_proba()
    temper_controler = embed_temper_control()

    probas_train, probas_test, y_train, y_test = embed.fit_and_transform(copy.deepcopy(train),
                                                                         copy.deepcopy(valid),
                                                                         copy.deepcopy(target_train),
                                                                         copy.deepcopy(target_valid), num_target)

    best_accuracy, x_train_encoded, x_test_encoded = temper_controler.fit_and_transform(copy.deepcopy(probas_train),
                                                                                   copy.deepcopy(probas_test),
                                                                                   copy.deepcopy(y_train.values),
                                                                                   copy.deepcopy(y_test.values),
                                                                                   types_target)
    print("default:{}".format(best_accuracy))

    # Compare with adjusted hyperparameters:
    for i in range(iter):
        params = {}
        params["num_inside_tree"] = random_search_num_inside_tree_list[i]
        params["max_depth"] = random_search_max_depth_list[i]

        embed = embed_trans_to_proba(params=params)
        probas_train, probas_test, y_train, y_test = embed.fit_and_transform(copy.deepcopy(train),
                                                                             copy.deepcopy(valid),
                                                                             copy.deepcopy(target_train),
                                                                             copy.deepcopy(target_valid), num_target)
        # Traverse one more layer for learning rate as it is highly sensitive and relatively time-efficient.
        for lr in lr_list:
            temper_controler = embed_temper_control(learningRate=lr)
            accuracy, x_train_encoded, x_test_encoded = temper_controler.fit_and_transform(copy.deepcopy(probas_train),
                                                                                                copy.deepcopy(probas_test),
                                                                                                copy.deepcopy(
                                                                                                    y_train.values),
                                                                                                copy.deepcopy(
                                                                                                    y_test.values),
                                                                                                types_target)
            print("{} {} and {} : {} ".format(random_search_num_inside_tree_list[i],random_search_max_depth_list[i],lr,accuracy))

            if best_accuracy < accuracy:
                best_accuracy = accuracy
                best_num_inside_tree = random_search_num_inside_tree_list[i]
                best_max_depth = random_search_max_depth_list[i]
                best_lr = lr

    return best_num_inside_tree,best_max_depth,best_lr



# Function for external invocation of the proposed encoding method
def proposed_weka_tree(data_train, data_test, target_train, target_test, dataname,if_adjust_hyperparameter=True):

    global mydataname
    mydataname=dataname

    types_target=np.unique(np.concatenate((target_train.values,target_test.values),axis=0))
    num_target=types_target.shape[0]

    if if_adjust_hyperparameter:
        best_iterlation_list = []
        best_population_list = []
        best_lr_list = []
        skf = StratifiedKFold(n_splits=3,shuffle=True)
        for train_index, valid_index in skf.split(data_train, target_train):
            best_iterlation, best_population,best_lr = find_best_params(10, copy.deepcopy(data_train.iloc[train_index, :]),
                                                                   copy.deepcopy(data_train.iloc[valid_index, :]),
                                                                   copy.deepcopy(target_train.iloc[train_index]),
                                                                   copy.deepcopy(target_train.iloc[valid_index]),num_target,types_target)

            best_iterlation_list.append(best_iterlation)
            best_population_list.append(best_population)
            best_lr_list.append(best_lr)

        final_best_iterlation = get_most_freq_value(best_iterlation_list)
        final_best_population = get_most_freq_value(best_population_list)
        final_best_lr = get_most_freq_value(best_lr_list)

        params = {}
        if final_best_iterlation is not None:
            params["iterlation"] = final_best_iterlation
        if final_best_population is not None:
            params["population"] = final_best_population

        print("final params this fold is:{}".format(params))
        embed = embed_trans_to_proba(params=params)

        if final_best_lr is not None:
            temper_controler = embed_temper_control(learningRate=final_best_lr)
        else:
            temper_controler = embed_temper_control()

    else:
        embed = embed_trans_to_proba()
        temper_controler = embed_temper_control()

    probas_train, probas_test, y_train, y_test = embed.fit_and_transform(copy.deepcopy(data_train), copy.deepcopy(data_test), copy.deepcopy(target_train), copy.deepcopy(target_test),num_target)
    precsion, x_train_encoded, x_test_encoded = temper_controler.fit_and_transform(copy.deepcopy(probas_train),
                                                                                   copy.deepcopy(probas_test),
                                                                                   copy.deepcopy(y_train.values),
                                                                                   copy.deepcopy(y_test.values),
                                                                                   types_target)

    print("final_accuracy_is:{}".format(precsion))


    df_x_train_encoded = pd.DataFrame(x_train_encoded, columns=None, index=None)
    df_x_test_encoded = pd.DataFrame(x_test_encoded, columns=None, index=None)

    return precsion,df_x_train_encoded,df_x_test_encoded


import sys
import numpy as np
import pandas as pd
import random
import torch
from catboost import CatBoostClassifier
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
import copy
from collections import Counter
from torch import nn


# This class is the first part of the proposed encoding method: Divide granularity and calculate probability distribution:
class embed_trans_to_proba:
    def __init__(self,params=None):
        self.params=params

    # Given the index of the current feature and a list of indices for the features to search, find the feature in the list that is most relevant to the current feature.
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


    # Divide the dataset into subsets at multiple granularities based on feature correlations.
    def divide(self,indexes):
        #Save the subset corresponding to the current granularity division.
        self.subsets.append(copy.deepcopy(indexes))

        # Recursive termination condition
        if len(indexes) == 1:
            return

        # Split features into two parts based on mutual information:
        s1=[]
        s2=[]
        now=random.choice(indexes)
        s1.append(now)
        flag_s1=False
        indexes.remove(now)
        while len(indexes) != 0 :
            now=self.find_most_relative_feature(now,indexes)
            if flag_s1 :
                s1.append(now)
                flag_s1 = False
            else:
                s2.append(now)
                flag_s1=True
            indexes.remove(now)

        # Recursively partition the two feature subsets
        self.divide(s1)
        self.divide(s2)


    # Compute the probability distribution of classes for a subset at a particular granularity.
    def fit_one_grain(self,subsets,categorical_index):
        #Find the column indices of the categorical features in the current subset for use as parameters in the subsequent CatboostClassifier fitting below
        cat_index_list = []
        for index_cat, index_subsets in enumerate(subsets):
            if index_subsets in categorical_index:
                cat_index_list.append(index_cat)

        if self.params is not None:
            cb = CatBoostClassifier(silent=True).set_params(**self.params)
        else:
            cb = CatBoostClassifier(silent=True)

        cb.fit(self.x_train.iloc[:, subsets], self.y_train, cat_features=cat_index_list)

        proba_train = cb.predict_proba(self.x_train.iloc[:, subsets])
        proba_test = cb.predict_proba(self.x_test.iloc[:, subsets])
        return proba_train,proba_test


    # Calculate the class probability distribution for all subsets.
    def trans_to_proba(self, categorical_index):
        self.subsets_probas_train = []
        self.subsets_probas_test = []

        for i, subsets in enumerate(self.subsets):
            proba_train, proba_test=self.fit_one_grain(i, subsets, categorical_index)
            self.subsets_probas_train.append(proba_train)
            self.subsets_probas_test.append(proba_test)

        self.subsets_probas_train = np.array(self.subsets_probas_train)
        self.subsets_probas_test = np.array(self.subsets_probas_test)


    # To obtain the accuracy of the predictions on the test set.
    def get_predict_test(self):
        return self.precision

    # The overall training and transformation functions for this class.
    def fit_and_transform(self,x_train,x_test,y_train,y_test,num_target):
        self.x_train=x_train
        self.x_test=x_test
        self.y_train=y_train
        self.y_test=y_test

        #Find the column indices of categorical features
        categorical_index=[]
        for i in range(x_train.shape[1]):
            dtypes=x_train.iloc[:,i].dtypes
            if dtypes != "int64":
                categorical_index.append(i)

        # num_target: The total number of categories for the labels.
        self.labels_nums=num_target
        self.subsets = []
        cols=x_train.shape[1]
        indexes=np.arange(cols).tolist()

        self.divide(copy.deepcopy(indexes))
        self.trans_to_proba(categorical_index)

        return self.subsets_probas_train, self.subsets_probas_test, self.y_train, self.y_test


#This class is the second part of the proposed encoding method:Adjusting temperature parameter:
class embed_temper_control:
    def __init__(self,iterations=1000,learningRate=1):
        self.iterations=iterations
        self.learningRate=learningRate


    def temper_control(self,subsets_probas_train,subsets_probas_test,y_train,y_test,types_target,lr=None,flag_save=False,parallel="single_gpu"):

        # There are three training modes for the model: cpu, single_gpu, and multi_gpu.
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

            # Train for "iterations" times.
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

                # After the final training iteration, calculate the prediction accuracy on the test set.
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

            # Train for "iterations" times.
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

                # After the final training iteration, calculate the prediction accuracy on the test set.
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

            # Train for "iterations" times.
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

                # After the final training iteration, calculate the prediction accuracy on the test set.
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

    # The overall training and transformation functions for this class.
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


# Find the value with the highest frequency in the list.
def get_most_freq_value(list):
    if Counter(list).most_common(1)[0][1]==1:
        return random.choice(list)
    else:
        return Counter(list).most_common(1)[0][0]


# Generate a hyperparameter search list with a length of "iters".
def random_choice_params(iters,params_list):
    if len(params_list)>iters:
        return np.random.choice(params_list,iters,replace=False)
    else:
        result1=copy.deepcopy(params_list)
        random.shuffle(result1)
        result2=np.random.choice(params_list,(iters-len(params_list)),replace=True).tolist()
        return result1 + result2

# This method is used for finding the optimal hyperparameters.
def find_best_params(iter,train,valid,target_train,target_valid,num_target,types_target):
    l2_leaf_reg_list = [1,4,9,16,25,36]
    depth_list = [4,6,8,10,12]
    n_estimators_list = [70,100,150,200,250,300]
    lr_list = [1, 0.7, 0.4, 0.1, 0.07, 0.04, 0.01]

    random_search_l2_leaf_reg_list=random_choice_params(iter,l2_leaf_reg_list)
    random_search_depth_list=random_choice_params(iter,depth_list)
    random_search_n_estimators_list = random_choice_params(iter, n_estimators_list)

    best_l2_leaf_reg=None
    best_depth=None
    best_n_estimators = None
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
        params["l2_leaf_reg"] = random_search_l2_leaf_reg_list[i]
        params["depth"] = random_search_depth_list[i]
        params["n_estimators"] = random_search_n_estimators_list[i]

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
            print("{} {} {} and {} : {} ".format(random_search_l2_leaf_reg_list[i],random_search_depth_list[i],random_search_n_estimators_list[i],lr,accuracy))

            if best_accuracy < accuracy:
                best_accuracy = accuracy
                best_l2_leaf_reg = random_search_l2_leaf_reg_list[i]
                best_depth = random_search_depth_list[i]
                best_n_estimators = random_search_n_estimators_list[i]
                best_lr = lr

    return best_l2_leaf_reg,best_depth,best_n_estimators,best_lr


# Function for external invocation of the proposed encoding method
def proposed(data_train,data_test,target_train,target_test,if_adjust_hyperparameter=True):

    types_target=np.unique(np.concatenate((target_train.values,target_test.values),axis=0))
    num_target=types_target.shape[0]

    if if_adjust_hyperparameter:
        best_l2_leaf_reg_list = []
        best_depth_list = []
        best_n_estimators_list = []
        best_lr_list = []

        # Using three-fold cross-validation, divide the training set into sub-training sets and sub-validation sets. Use the accuracy of the sub-validation sets as a metric to search for the optimal hyperparameters.
        skf = StratifiedKFold(n_splits=3)
        for train_index, valid_index in skf.split(data_train, target_train):
            best_l2_leaf_reg, best_depth,best_n_estimators,best_lr = find_best_params(10, copy.deepcopy(data_train.iloc[train_index, :]),
                                                                   copy.deepcopy(data_train.iloc[valid_index, :]),
                                                                   copy.deepcopy(target_train.iloc[train_index]),
                                                                   copy.deepcopy(target_train.iloc[valid_index]),num_target,types_target)
            best_l2_leaf_reg_list.append(best_l2_leaf_reg)
            best_depth_list.append(best_depth)
            best_n_estimators_list.append(best_n_estimators)
            best_lr_list.append(best_lr)

        final_best_l2_leaf_reg = get_most_freq_value(best_l2_leaf_reg_list)
        final_best_depth = get_most_freq_value(best_depth_list)
        final_best_n_estimators = get_most_freq_value(best_n_estimators_list)
        final_best_lr = get_most_freq_value(best_lr_list)

        params = {}
        if final_best_l2_leaf_reg is not None:
            params["l2_leaf_reg"] = final_best_l2_leaf_reg
        if final_best_depth is not None:
            params["depth"] = final_best_depth
        if final_best_n_estimators is not None:
            params["n_estimators"] = final_best_n_estimators

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


    df_x_train_encoded=pd.DataFrame(x_train_encoded,columns=None,index=None)
    df_x_test_encoded = pd.DataFrame(x_test_encoded, columns=None, index=None)
    return precsion,df_x_train_encoded,df_x_test_encoded

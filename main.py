
import numpy as np
import pandas as pd
import copy

from jpype._core import startJVM, shutdownJVM

import GCForest
import GCForest_weka

from proposed import proposed
from proposed_weka_tree import proposed_weka_tree






if __name__ == '__main__':

    # The project for generating MyRandomForest.jar can be found in the folder named MyRandomForest in the same directory,in which MyRandomForest class can be seen in the src directory .
    # Calling proposed_weka_tree and GCForest_weka both require starting the JVM:
    startJVM(jdk_path+"jdk1.8.0_391/jre/lib/amd64/server/libjvm.so", "-ea",
             "-Djava.class.path=%s" % (jar_path + 'MyRandomForest.jar'))

    #Dataset description: Read the dataset into two DataFrame variables, 'data' and 'target,' according to features and labels. Then, split them into 'train' and 'test' sets, respectively.
    # data_train
    # data_test
    # target_train
    # target_test


    #proposed（CB）
    precision, data_train_encoded, data_test_encoded = proposed(
        copy.deepcopy(data_train), copy.deepcopy(data_test),
        copy.deepcopy(target_train), copy.deepcopy(target_test),
        if_adjust_hyperparameter=False)

    #proposed（RF）
    precision, data_train_encoded, data_test_encoded = proposed_weka_tree(
        copy.deepcopy(data_train), copy.deepcopy(data_test),
        copy.deepcopy(target_train), copy.deepcopy(target_test),
        dataname=dataname + str(time),
        if_adjust_hyperparameter=True)

    # CF(tree_ensemble_method=random_forest_in_weka)
    accuracy = GCForest_weka.main(
        copy.deepcopy(data_train_encoded), copy.deepcopy(data_test_encoded),
        copy.deepcopy(target_train), copy.deepcopy(target_test), dataname=dataname,
        if_is_encoded=if_encoded)

    # CF(tree_ensemble_method=random_forest_in_sklearn)
    accuracy = GCForest.main(
        copy.deepcopy(data_train_encoded), copy.deepcopy(data_test_encoded),
        copy.deepcopy(target_train), copy.deepcopy(target_test))


    shutdownJVM()
















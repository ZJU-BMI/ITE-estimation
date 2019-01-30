import pandas as pd
from sklearn.preprocessing import minmax_scale, Imputer
import fancyimpute as fi
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
import numpy as np
from sklearn import metrics
from sklearn.linear_model.logistic import LogisticRegression
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.preprocessing import OneHotEncoder
import pickle
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
import tensorflow as tf
from cnn_8class import CNN8class
from data.read_data import DataSet

with open('split_hf.pickle', 'rb') as f:
    sssList = pickle.load(f)

med_data = pd.read_csv('follow_up_data_2_1y_2_drug_8.csv', encoding='gbk')
med_data = med_data.values

t = med_data[:,-1]
t = t.reshape(-1,1)

patient_data = pd.read_csv('follow_up_data_2_1y_2_feature_drug.csv', encoding='gbk')
patient_data = patient_data.values
patient_data = patient_data[:, 1:106]
patient_data = fi.IterativeImputer().fit_transform(patient_data)
#patient_data = minmax_scale(patient_data, feature_range=(0, 1))
x = patient_data

outcome=pd.read_csv('follow_up_data_2_1y_2_outcome.csv', encoding='gbk')
outcome=outcome.values
outcome=outcome[:,1]
y=outcome.reshape(-1,1)

Acc_tol = np.zeros(shape=(10, 20))
precision_tol = np.zeros(shape=(10, 20))
recall_tol = np.zeros(shape=(10, 20))
f1_tol = np.zeros(shape=(10, 20))
for i in range(0, 10): #hyperparameter tuning
    print("iteration number: %d" % i)
    sss = sssList[i]
    for train_index, test_index in sss.split(x, t):
        train_val_x = x[train_index]
        train_val_y = y[train_index]
        train_val_t = t[train_index]

        for train_index1, test_index1 in sss.split(train_val_x, train_val_t):
            train_x = train_val_x[train_index1]
            train_y = train_val_y[train_index1]
            train_t = train_val_t[train_index1]

            val_x = train_val_x[test_index1]
            val_y = train_val_y[test_index1]
            val_t = train_val_t[test_index1]

            sample_num = np.size(val_x, 0)

            train_data = DataSet(train_x, np.hstack((train_y, train_t)))

            index_c0 = np.argwhere(train_t == 0)
            index_c1 = np.argwhere(train_t == 1)
            index_c2 = np.argwhere(train_t == 2)
            index_c3 = np.argwhere(train_t == 3)
            index_c4 = np.argwhere(train_t == 4)
            index_c5 = np.argwhere(train_t == 5)
            index_c6 = np.argwhere(train_t == 6)
            index_c7 = np.argwhere(train_t == 7)

            train_y_0 = train_y[index_c0[:, 0]]
            train_y_1 = train_y[index_c1[:, 0]]
            train_y_2 = train_y[index_c2[:, 0]]
            train_y_3 = train_y[index_c3[:, 0]]
            train_y_4 = train_y[index_c4[:, 0]]
            train_y_5 = train_y[index_c5[:, 0]]
            train_y_6 = train_y[index_c6[:, 0]]
            train_y_7 = train_y[index_c7[:, 0]]

            train_x_0 = train_x[index_c0[:, 0]]
            train_x_1 = train_x[index_c1[:, 0]]
            train_x_2 = train_x[index_c2[:, 0]]
            train_x_3 = train_x[index_c3[:, 0]]
            train_x_4 = train_x[index_c4[:, 0]]
            train_x_5 = train_x[index_c5[:, 0]]
            train_x_6 = train_x[index_c6[:, 0]]
            train_x_7 = train_x[index_c7[:, 0]]

            index_c0_val = np.argwhere(val_t == 0)
            index_c1_val = np.argwhere(val_t == 1)
            index_c2_val = np.argwhere(val_t == 2)
            index_c3_val = np.argwhere(val_t == 3)
            index_c4_val = np.argwhere(val_t == 4)
            index_c5_val = np.argwhere(val_t == 5)
            index_c6_val = np.argwhere(val_t == 6)
            index_c7_val = np.argwhere(val_t == 7)

            for n in range(0, 20):
                print("neighbor number: %d" % n)
                for repeatitions in range(1):
                    print("repeatition number: %d" % repeatitions)
                    predict_0 = np.zeros(shape=(sample_num, 1))
                    predict_1 = np.zeros(shape=(sample_num, 1))
                    predict_2 = np.zeros(shape=(sample_num, 1))
                    predict_3 = np.zeros(shape=(sample_num, 1))
                    predict_4 = np.zeros(shape=(sample_num, 1))
                    predict_5 = np.zeros(shape=(sample_num, 1))
                    predict_6 = np.zeros(shape=(sample_num, 1))
                    predict_7 = np.zeros(shape=(sample_num, 1))

                    tf.reset_default_graph()
                    cnn = CNN8class(dense_units=46)
                    cnn.train_process(train_data)
                    train_x_0_encode = cnn.cnn_x(train_x_0)
                    train_x_1_encode = cnn.cnn_x(train_x_1)
                    train_x_2_encode = cnn.cnn_x(train_x_2)
                    train_x_3_encode = cnn.cnn_x(train_x_3)
                    train_x_4_encode = cnn.cnn_x(train_x_4)
                    train_x_5_encode = cnn.cnn_x(train_x_5)
                    train_x_6_encode = cnn.cnn_x(train_x_6)
                    train_x_7_encode = cnn.cnn_x(train_x_7)
                    val_x_encode = cnn.cnn_x(val_x)

                    n_n = n+1 # neighbour number

                    dis_0 = pairwise_distances(val_x_encode, train_x_0_encode, metric='euclidean')
                    dis_0 = minmax_scale(dis_0, axis=1, feature_range=(0, 1))
                    position_0 = np.argsort(dis_0, axis=1)[:, 0:50]
                    for i_0 in range(sample_num):
                        w_0 = (1 - dis_0[i_0, position_0[i_0, 0:n_n]]) / np.sum(1 - dis_0[i_0, position_0[i_0, 0:n_n]])
                        w_0 = w_0.reshape(-1, 1)
                        train_y_0_ = train_y_0[position_0[i_0, 0:n_n]]
                        train_y_0_ = train_y_0_.reshape(1, -1)
                        predict_0[i_0] = np.dot(train_y_0_, w_0)
                    predict_0 = np.int64(predict_0 >= 0.5)

                    dis_1 = pairwise_distances(val_x_encode, train_x_1_encode, metric='euclidean')
                    dis_1 = minmax_scale(dis_1, axis=1, feature_range=(0, 1))
                    position_1 = np.argsort(dis_1, axis=1)[:, 0:50]
                    for i_1 in range(sample_num):
                        w_1 = (1 - dis_1[i_1, position_1[i_1, 0:n_n]]) / np.sum(1 - dis_1[i_1, position_1[i_1, 0:n_n]])
                        w_1 = w_1.reshape(-1, 1)
                        train_y_1_ = train_y_1[position_1[i_1, 0:n_n]]
                        train_y_1_ = train_y_1_.reshape(1, -1)
                        predict_1[i_1] = np.dot(train_y_1_, w_1)
                    predict_1 = np.int64(predict_1 >= 0.5)

                    dis_2 = pairwise_distances(val_x_encode, train_x_2_encode, metric='euclidean')
                    dis_2 = minmax_scale(dis_2, axis=1, feature_range=(0, 1))
                    position_2 = np.argsort(dis_2, axis=1)[:, 0:50]
                    for i_2 in range(sample_num):
                        w_2 = (1 - dis_2[i_2, position_2[i_2, 0:n_n]]) / np.sum(1 - dis_2[i_2, position_2[i_2, 0:n_n]])
                        w_2 = w_2.reshape(-1, 1)
                        train_y_2_ = train_y_2[position_2[i_2, 0:n_n]]
                        train_y_2_ = train_y_2_.reshape(1, -1)
                        predict_2[i_2] = np.dot(train_y_2_, w_2)
                    predict_2 = np.int64(predict_2 >= 0.5)

                    dis_3 = pairwise_distances(val_x_encode, train_x_3_encode, metric='euclidean')
                    dis_3 = minmax_scale(dis_3, axis=1, feature_range=(0, 1))
                    position_3 = np.argsort(dis_3, axis=1)[:, 0:50]
                    for i_3 in range(sample_num):
                        w_3 = (1 - dis_3[i_3, position_3[i_3, 0:n_n]]) / np.sum(1 - dis_3[i_3, position_3[i_3, 0:n_n]])
                        w_3 = w_3.reshape(-1, 1)
                        train_y_3_ = train_y_3[position_3[i_3, 0:n_n]]
                        train_y_3_ = train_y_3_.reshape(1, -1)
                        predict_3[i_3] = np.dot(train_y_3_, w_3)
                    predict_3 = np.int64(predict_3 >= 0.5)

                    dis_4 = pairwise_distances(val_x_encode, train_x_4_encode, metric='euclidean')
                    dis_4 = minmax_scale(dis_4, axis=1, feature_range=(0, 1))
                    position_4 = np.argsort(dis_4, axis=1)[:, 0:50]
                    for i_4 in range(sample_num):
                        w_4 = (1 - dis_4[i_4, position_4[i_4, 0:n_n]]) / np.sum(1 - dis_4[i_4, position_4[i_4, 0:n_n]])
                        w_4 = w_4.reshape(-1, 1)
                        train_y_4_ = train_y_4[position_4[i_4, 0:n_n]]
                        train_y_4_ = train_y_4_.reshape(1, -1)
                        predict_4[i_4] = np.dot(train_y_4_, w_4)
                    predict_4 = np.int64(predict_4 >= 0.5)

                    dis_5 = pairwise_distances(val_x_encode, train_x_5_encode, metric='euclidean')
                    dis_5 = minmax_scale(dis_5, axis=1, feature_range=(0, 1))
                    position_5 = np.argsort(dis_5, axis=1)[:, 0:50]
                    for i_5 in range(sample_num):
                        w_5 = (1 - dis_5[i_5, position_5[i_5, 0:n_n]]) / np.sum(1 - dis_5[i_5, position_5[i_5, 0:n_n]])
                        w_5 = w_5.reshape(-1, 1)
                        train_y_5_ = train_y_5[position_5[i_5, 0:n_n]]
                        train_y_5_ = train_y_5_.reshape(1, -1)
                        predict_5[i_5] = np.dot(train_y_5_, w_5)
                    predict_5 = np.int64(predict_5 >= 0.5)

                    dis_6 = pairwise_distances(val_x_encode, train_x_6_encode, metric='euclidean')
                    dis_6 = minmax_scale(dis_6, axis=1, feature_range=(0, 1))
                    position_6 = np.argsort(dis_6, axis=1)[:, 0:50]
                    for i_6 in range(sample_num):
                        w_6 = (1 - dis_6[i_6, position_6[i_6, 0:n_n]]) / np.sum(1 - dis_6[i_6, position_6[i_6, 0:n_n]])
                        w_6 = w_6.reshape(-1, 1)
                        train_y_6_ = train_y_6[position_6[i_6, 0:n_n]]
                        train_y_6_ = train_y_6_.reshape(1, -1)
                        predict_6[i_6] = np.dot(train_y_6_, w_6)
                    predict_6 = np.int64(predict_6 >= 0.5)

                    dis_7 = pairwise_distances(val_x_encode, train_x_7_encode, metric='euclidean')
                    dis_7 = minmax_scale(dis_7, axis=1, feature_range=(0, 1))
                    position_7 = np.argsort(dis_7, axis=1)[:, 0:50]
                    for i_7 in range(sample_num):
                        w_7 = (1 - dis_7[i_7, position_7[i_7, 0:n_n]]) / np.sum(1 - dis_7[i_7, position_7[i_7, 0:n_n]])
                        w_7 = w_7.reshape(-1, 1)
                        train_y_7_ = train_y_7[position_7[i_7, 0:n_n]]
                        train_y_7_ = train_y_7_.reshape(1, -1)
                        predict_7[i_7] = np.dot(train_y_7_, w_7)
                    predict_7 = np.int64(predict_7 >= 0.5)

                    predict = np.vstack((predict_0[index_c0_val[:, 0]], predict_1[index_c1_val[:, 0]], predict_2[index_c2_val[:, 0]],
                                         predict_3[index_c3_val[:, 0]], predict_4[index_c4_val[:, 0]], predict_5[index_c5_val[:, 0]],
                                         predict_6[index_c6_val[:, 0]], predict_7[index_c7_val[:, 0]]))
                    val_y_tol = np.vstack(
                        (val_y[index_c0_val[:, 0]], val_y[index_c1_val[:, 0]], val_y[index_c2_val[:, 0]], val_y[index_c3_val[:, 0]], val_y[index_c4_val[:, 0]], val_y[index_c5_val[:, 0]], val_y[index_c6_val[:, 0]], val_y[index_c7_val[:, 0]]))
                    cm = metrics.confusion_matrix(val_y_tol, predict)
                    Acc = metrics.accuracy_score(val_y_tol, predict)
                    precision = metrics.precision_score(val_y_tol, predict)
                    recall = metrics.recall_score(val_y_tol, predict)
                    f1 = metrics.f1_score(val_y_tol, predict)

                    Acc_tol[i, n] = Acc
                    precision_tol[i, n] = precision
                    recall_tol[i, n] = recall
                    f1_tol[i, n] = f1
                    print("end")

'''
np.save(file="Acc_tol_hf_0_4_neighbor.npy", arr=Acc_tol)
np.save(file="precision_tol_hf_0_4_neighbor.npy", arr=precision_tol)
np.save(file="recall_tol_hf_0_4_neighbor.npy", arr=recall_tol)
np.save(file="f1_tol_hf_0_4_neighbor.npy", arr=f1_tol)
'''

mean_Acc_tol = np.mean(Acc_tol, axis=0)
mean_precision_tol = np.mean(precision_tol, axis=0)
mean_recall_tol = np.mean(recall_tol, axis=0)
mean_f1_tol = np.mean(f1_tol, axis=0)

np.savetxt("mean_cnncf_hf_30d_neighbor.csv", np.vstack((mean_Acc_tol, mean_precision_tol, mean_recall_tol, mean_f1_tol)), delimiter=',')
np.savetxt("result_cnncf_hf_30d_neighbor.csv", np.vstack((Acc_tol, precision_tol, recall_tol, f1_tol)), delimiter=',')
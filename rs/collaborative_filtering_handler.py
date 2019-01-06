from collections import defaultdict
from surprise import accuracy, BaselineOnly
from surprise import KNNBasic
import numpy as np
import pandas as pd
from surprise import SVD
from surprise import Dataset
import time

from surprise.prediction_algorithms.matrix_factorization import NMF


def get_top_n(predictions, n=10):

    # First map the predictions to each user.
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))

    # Then sort the predictions for each user and retrieve the k highest ones.
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    return top_n


def use_svd():
    start = time.time()
    performance = []

    data = Dataset.load_builtin('ml-100k')
    trainset = data.build_full_trainset()

    print('Using SVD')
    algo_SVD = SVD()
    algo_SVD.fit(trainset)

    testset = trainset.build_anti_testset()
    predictions_SVD = algo_SVD.test(testset)

    accuracy_rmse = accuracy.rmse(predictions_SVD)
    accuracy_mae = accuracy.mae(predictions_SVD)
    performance.append(accuracy_rmse)
    performance.append(accuracy_mae)

    end = time.time()
    performance.append(end-start)

    return performance

def use_nmf():
    start = time.time()
    performance = []

    data = Dataset.load_builtin('ml-100k')
    trainset = data.build_full_trainset()

    print('Using NMF')
    algo_NMF = NMF()
    algo_NMF.fit(trainset)

    testset = trainset.build_anti_testset()
    predictions_NMF = algo_NMF.test(testset)

    accuracy_rmse = accuracy.rmse(predictions_NMF)
    accuracy_mae = accuracy.mae(predictions_NMF)
    performance.append(accuracy_rmse)
    performance.append(accuracy_mae)

    end = time.time()
    performance.append(end-start)

    return performance

def use_knn():
    start = time.time()
    performance = []

    data = Dataset.load_builtin('ml-100k')
    trainset = data.build_full_trainset()

    print('Using KNN')
    algo_KNN = KNNBasic()
    algo_KNN.fit(trainset)

    testset = trainset.build_anti_testset()
    predictions_KNN = algo_KNN.test(testset)

    accuracy_rmse = accuracy.rmse(predictions_KNN)
    accuracy_mae = accuracy.mae(predictions_KNN)
    performance.append(accuracy_rmse)
    performance.append(accuracy_mae)

    end = time.time()
    performance.append(end - start)

    return performance

def use_cosine_similarity():
    start = time.time()
    performance = []

    data = Dataset.load_builtin('ml-100k')
    trainset = data.build_full_trainset()

    print('Using cosine similarity')
    sim_options = {'name': 'cosine',
                   'user_based': False  # compute  similarities between items
                   }
    algo_cosine = KNNBasic(sim_options=sim_options)
    algo_cosine.fit(trainset)

    testset = trainset.build_anti_testset()
    predictions_KNN = algo_cosine.test(testset)

    accuracy_rmse = accuracy.rmse(predictions_KNN)
    accuracy_mae = accuracy.mae(predictions_KNN)
    performance.append(accuracy_rmse)
    performance.append(accuracy_mae)

    end = time.time()
    performance.append(end - start)

    return performance

def use_pearson_baseline():
    start = time.time()
    performance = []

    data = Dataset.load_builtin('ml-100k')
    trainset = data.build_full_trainset()

    print('Using Pearson baseline')
    sim_options = {'name': 'pearson_baseline',
                   'shrinkage': 0  # no shrinkage
                   }
    algo_pearson = KNNBasic(sim_options=sim_options)
    algo_pearson.fit(trainset)

    testset = trainset.build_anti_testset()
    predictions_KNN = algo_pearson.test(testset)

    accuracy_rmse = accuracy.rmse(predictions_KNN)
    accuracy_mae = accuracy.mae(predictions_KNN)
    performance.append(accuracy_rmse)
    performance.append(accuracy_mae)

    end = time.time()
    performance.append(end - start)

    return performance

def use_als():
    start = time.time()
    performance = []

    data = Dataset.load_builtin('ml-100k')
    trainset = data.build_full_trainset()

    print('Using ALS')
    bsl_options = {'method': 'als',
                   'n_epochs': 20,
                   'reg_u': 12,
                   'reg_i': 5
                   }
    algo_ALS = BaselineOnly(bsl_options=bsl_options)
    algo_ALS.fit(trainset)

    testset = trainset.build_anti_testset()
    predictions_ALS = algo_ALS.test(testset)

    accuracy_rmse \
        = accuracy.rmse(predictions_ALS)
    accuracy_mae = accuracy.mae(predictions_ALS)
    performance.append(accuracy_rmse)
    performance.append(accuracy_mae)

    end = time.time()
    performance.append(end - start)

    return performance

def use_sgd():
    start = time.time()
    performance = []

    data = Dataset.load_builtin('ml-100k')
    trainset = data.build_full_trainset()

    print('Using SGD')
    bsl_options = {'method': 'sgd',
                   'learning_rate': .005,
                   }

    algo_SGD = BaselineOnly(bsl_options=bsl_options)
    algo_SGD.fit(trainset)

    testset = trainset.build_anti_testset()
    predictions_SGD = algo_SGD.test(testset)

    accuracy_rmse = accuracy.rmse(predictions_SGD)
    accuracy_mae = accuracy.mae(predictions_SGD)
    performance.append(accuracy_rmse)
    performance.append(accuracy_mae)

    end = time.time()
    performance.append(end - start)

    return performance

def make_predictions(user_id):
    performance = []
    algorithms = ['SVD', 'KNN', 'ALS']

    # First train an SVD algorithm on the movielens dataset.
    data = Dataset.load_builtin('ml-100k')
    trainset = data.build_full_trainset()

    algo_SVD = SVD()
    algo_SVD.fit(trainset)

    # Then predict ratings for all pairs (u, i) that are NOT in the training set.
    # SVD algorithm
    testset = trainset.build_anti_testset()
    predictions_SVD = algo_SVD.test(testset)

    accurancy_SVD = accuracy.rmse(predictions_SVD)
    performance.append(accurancy_SVD)

    algo_KNN = KNNBasic()
    algo_KNN.fit(trainset)

    predictions_KNN = algo_SVD.test(testset)

    accurancy_KNN = accuracy.rmse(predictions_KNN)
    performance.append(accurancy_KNN)

    bsl_options = {'method': 'als',
                   'n_epochs': 5,
                   'reg_u': 12,
                   'reg_i': 5
                   }
    algo_ALS = BaselineOnly(bsl_options=bsl_options)
    algo_ALS.fit(trainset)

    predictions_ALS = algo_ALS.test(testset)

    accurancy_ALS = accuracy.rmse(predictions_ALS)
    performance.append(accurancy_ALS)

    # comparing algorithms by performance
    best_performance_index = performance.index(min(performance))
    best_algorithm = algorithms[best_performance_index]

    if best_algorithm == 'SVD':
        top_n = get_top_n(predictions_SVD, n=10)
    elif best_algorithm == 'KNN':
        top_n = get_top_n(predictions_KNN, n=10)
    elif best_algorithm == 'ALS':
        top_n = get_top_n(predictions_ALS, n=10)

    i_cols = ['movie_id', 'movie_title', 'release_date', 'video_release_date', 'IMDb_URL', 'unknown', 'Action',
              'Adventure',
              'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
              'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']

    items = pd.read_csv('../../ml-100k/u.item', sep='|', names=i_cols,
                        encoding='latin-1')

    predictions = []
    # Print the recommended items for the user
    for uid, user_ratings in top_n.items():
        if int(uid) + 1 == int(user_id) + 1:
            # print(uid, [iid for (iid, _) in user_ratings])
            for (iid, _) in user_ratings:
                title = items[items['movie_id'] == int(iid) + 1]['movie_title']
                title_t = str(title)
                title_split = title_t.split()
                print(title_split)
                # print(title_split(1))
                # print(title_split(2))
                # print(title_t)
                predictions.append(title_t)

    return predictions

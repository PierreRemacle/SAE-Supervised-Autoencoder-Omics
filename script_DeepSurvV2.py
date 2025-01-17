# -*- coding: utf-8 -*-
"""


Copyright   I3S CNRS UCA 

This code implement the cross validation Train Validate Test 

https://scikit-learn.org/stable/modules/cross_validation.html

When using this code , please cite

 Michel Barlaud, Guillaume Perez, and Jean-Paul Marmorat.
Linear time bi-level l1,infini projection ; application to feature selection and
sparsification of auto-encoders neural networks.
http://arxiv.org/abs/2407.16293, 2024.

This code minimize the following constrained approach
in our Fully Connected neural Network (FCNN):
Loss(W) = ϕ( Ye , Y ) s.t. BP1,∞(W) ≤ η. (1)

Where Y is the true  value and Ye is the estimate age by the neural network, ϕ is the error loss, W are the weights of the FCNN
and BP1,∞ is the bi-level ℓ1,∞ projection.
    
"""
# %%
import functions.functions_network_pytorch as fnp
import functions.functions_torch_regression_V3 as ft
import os

import time
import matplotlib.pyplot as plt
from numpy import linalg as LA
import pandas as pd
import numpy as np
import seaborn as sns
import torch
import ot
from torch import nn
from sklearn import metrics
# from numpy import linalg as LA

# choice of wasserstein distance function
# from scipy.stats import wasserstein_distance


def wasserstein_distance(u, v): return ot.wasserstein_1d(u, v, p=2)


# %%

if __name__ == "__main__":

    ######## Parameters ########
    start_time = time.time()
    # Set seed
    SEEDS = [5]

    # Set device (GPU or CPU)
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    nfolds = 4  # Number of folds for the cross-validation process
    N_EPOCHS = 30  # Number of epochs for the first descent
    N_EPOCHS_MASKGRAD = 30  # Number of epochs for training masked gradient
    LR = 0.00005  # Learning rate
    BATCH_SIZE = 50  # Optimize the trade off between accuracy and computational time

    # log transform of the input data
    doLog = False
    # unit scaling of the input data
    doScale = True
    # row normalization of the input data
    doRowNorm = False

    # distribution normalisation for Wasserstein
    WDNorm = False

    # parameters alpha and eta :

    # alpha is used in the criterion, define how much RMSE is consider over wasserstein in the loss function
    ALPHAS = np.array([0.0005, 0.0002, 0.0001, 0.00005]
                      )  # [0.1,0.01,0.002,0.0005]
    ETAS = np.array([0.025, 0.1, 0.2, 0.5, 1])  # [0.05,0.2,0.5,1,2]

    mse = nn.MSELoss(reduction="sum")

    def criterion_regression(u, v): return wasserstein_distance(
        u, v) + ALPHAS[0] * mse(u, v)
    # criterion_regression = wasserstein_distance
    # criterion_regression = nn.SmoothL1Loss(reduction="sum")  # SmoothL1Loss

    # Dataset choice
    # file_name= 'Synth_Reg_500f_64inf_2000s.csv'
    # file_name = 'dataset_anonymized.csv'
    # file_name="TimeInterval_no_outliers_no_null.csv"
    # file_name = "TimeInterval_no_outliers_no_null_without_exitus.csv"
    # file_name="TimeInterval_no_outliers_no_null_only_cause_1.csv"
    file_name = "TimeInterval.csv"

    test_size = 0.1  # Test size during the train test split

    # Choose Architecture
    # net_name = 'LeNet'
    # net_name = "netBio"
    net_name = 'FAIR'
    # net_name = 'dnn'

    # Choose if normalisation layer in the network
    norm = False

    # Choose nb of hidden neurons for NetBio and Fair network
    n_hidden = 300  # amount of neurons on netbio's hidden layer

    run_model = "No_proj"  # default model run
    # Do projection at the middle layer or not
    DO_PROJ_MIDDLE = False
    GRADIENT_MASK = True  # Whether to do a second descent
    if GRADIENT_MASK:
        run_model = "ProjectionLastEpoch"

    # Choose projection function
    if not GRADIENT_MASK:
        TYPE_PROJ = "No_proj"
        TYPE_PROJ_NAME = "No_proj"
    else:

        # TYPE_PROJ = ft.proj_l1ball  # projection l1
        # TYPE_PROJ = ft.proj_l11ball  # original projection l11 (col-wise zeros)
        # TYPE_PROJ = ft.proj_l21ball   # projection l21
        # TYPE_PROJ = ft.proj_l1infball  # projection l1,inf
        TYPE_PROJ = ft.bilevel_proj_l1Inftyball  # projection bilevel l1,inf
        # TYPE_PROJ = 'bilevel_proj_l11ball' #Projection bilevel l11

        TYPE_PROJ_NAME = TYPE_PROJ.__name__

    # TYPE_ACTIVATION = "tanh"
    # TYPE_ACTIVATION = "gelu"
    TYPE_ACTIVATION = "relu"
    # TYPE_ACTIVATION = "silu"

    AXIS = 1  # 1 for columns (features), 0 for rows (neurons)
    TOL = 1e-3  # error margin for the L1inf algorithm and gradient masking

    bW = 0.5  # Kernel size for distribution plots

    DoTopGenes = True  # Compute feature rankings

    DoTopFeatures = True

    DoSparsity = True  # Show the sparsity of the SAE

    # Save Results or not
    # note that will save the metric on the last test of eta and alpha only.
    # only the metric defined at the line 165 to 171 will be saved over all the couple (eta,alpha)
    SAVE_FILE = True

    ######## Main routine ########

    # Output Path
    outputPath = (
        "results_stat"
        + "/"
        + file_name.split(".")[0]
        + "/"
    )
    if not os.path.exists(outputPath):  # make the directory if it does not exist
        os.makedirs(outputPath)

    # initialisation of the data array that will compute the value over the different value of ETA and ALPHA
    datas_finaltest = []
    datas_retest = []
    datas_retrain = []
    datas_test = []
    datas_train = []
    selectedFeatures = [np.zeros((len(ETAS), 3)) for i in range(len(ALPHAS))]
    sparcities = []

    for ei, ETA in enumerate(ETAS):
        print("\n\n testing eta value : ", ETA, "\n\n")
        for ai, alpha in enumerate(ALPHAS):

            print("\n\n testing eta,alpha value : ", ETA, alpha, "\n\n")

            # some of the following may not need to be in the loops (like loading the data).
            def criterion_regression(u, v): return ot.wasserstein_1d(
                u, v, p=2) + alpha * mse(u, v)

            # Load data
            X, X_test, Y, y_test, feature_names, label_name_train, label_name_test, patient_name, gaussianKDE, gaussianKDETest, divided = ft.ReadDataCV_surv(
                file_name, test_size=test_size, doScale=doScale, doLog=doLog,  doRowNorm=doRowNorm
            )

            feature_len = len(feature_names)
            print(f"Number of features: {feature_len}")
            seed_idx = 0
            data_train = np.zeros((nfolds * len(SEEDS), 6))
            data_test = np.zeros((nfolds * len(SEEDS), 6))
            data_retrain = np.zeros((nfolds * len(SEEDS), 6))
            data_retest = np.zeros((nfolds * len(SEEDS), 6))

            data_finalTest = np.zeros((nfolds * len(SEEDS), 6))

            sparsity_matrix = np.zeros((nfolds * len(SEEDS), 1))
            sparsity_matrix_retraing = np.zeros((nfolds * len(SEEDS), 1))

            # routine for the current couple eta,alpha
            for seed in SEEDS:
                np.random.seed(seed)
                torch.manual_seed(seed)
                torch.cuda.manual_seed(seed)
                for fold_idx in range(nfolds):
                    GRADIENT_MASK = True
                    start_time = time.time()
                    train_dl, test_dl, train_len, test_len, Ytest = ft.CrossValSurv(
                        X, Y, patient_name, BATCH_SIZE, fold_idx, seed
                    )
                    print(
                        "Len of train set: {}, Len of test set: {}".format(
                            train_len, test_len)
                    )
                    print("----------- Start fold ",
                          fold_idx, "----------------")

                    print("----------- Start Training ----------------")
                    # Define the SEED to fix the initial parameters

                    data_encoder, net, best_model = ft.training(seed, feature_len, TYPE_ACTIVATION, DEVICE, n_hidden, norm, feature_names,
                                                                GRADIENT_MASK, net_name, LR, criterion_regression, train_dl, train_len,
                                                                gaussianKDE, test_dl, test_len, outputPath, TYPE_PROJ, SEEDS, fold_idx,
                                                                nfolds, N_EPOCHS, N_EPOCHS_MASKGRAD, DO_PROJ_MIDDLE, ETA, AXIS, TOL)
                    end_time = time.time()

                    # Calculate and print the execution time
                    execution_time = end_time - start_time
                    print(f"Execution time for training :")
                    print(f"{execution_time} seconds for fold {fold_idx}")

                    data_encoder = data_encoder.cpu().detach().numpy()

                    (
                        data_encoder_test,
                        integrated_brier_score,
                        ev,
                    ) = ft.runBestNet_survie(
                        test_dl,
                        outputPath,
                        fold_idx,
                        net,
                        feature_names,
                        test_len,
                        best_model,
                    )

                    if seed == SEEDS[-1]:
                        if fold_idx == 0:
                            integrated_brier_scores = [integrated_brier_score]
                            LP_test = data_encoder_test.detach().cpu().numpy()
                        else:
                            integrated_brier_scores.append(
                                integrated_brier_score)
                            LP_test = np.concatenate(
                                (LP_test, data_encoder_test.detach().cpu().numpy())
                            )
                        # Compute average Integrated Brier Score over all folds
                        mean_brier_score = np.mean(integrated_brier_scores)
                        print(f"Mean Integrated Brier Score: ")
                        print(f"{mean_brier_score:.4f}")

                        # Extracting hazards from data_encoder_test for visualization
                        # Assuming hazards are in the first column
                        hazards = LP_test[:, 0]
                        durations = LP_test[:, 1]  # True durations
                        events = LP_test[:, 2]  # Event indicators

                        # Evaluating survival metrics and plotting survival curves (optional)
                        survival_times = np.linspace(
                            np.min(durations), np.max(durations), 100)
                        brier_scores = ev.brier_score(survival_times)

                        plt.figure()
                        plt.plot(survival_times, brier_scores,
                                 label="Brier Score")
                        plt.xlabel("Time")
                        plt.ylabel("Brier Score")
                        plt.title("Brier Score Over Time")
                        plt.legend()
                        plt.show()
                        plt.savefig(f'brier_score_{seed}.png')

                    label_predicted = data_encoder[:, 0]
                    labels_encoder = data_encoder[:, -1]
                    data_encoder_test = data_encoder_test.cpu().detach().numpy()
                    # mse score
                    data_train[seed_idx * 4 + fold_idx, 0] = metrics.mean_squared_error(
                        label_predicted, labels_encoder
                    )

                    label_predicted_test = data_encoder_test[:, 0]
                    labels_encoder_test = data_encoder_test[:, -1]
                    data_test[seed_idx * 4 + fold_idx, 0] = metrics.mean_squared_error(
                        label_predicted_test, labels_encoder_test
                    )

                    data_train[seed_idx * 4 + fold_idx, 1] = metrics.mean_squared_error(
                        label_predicted, labels_encoder
                    )**0.5 * divided

                    data_test[seed_idx * 4 + fold_idx, 1] = metrics.mean_squared_error(
                        label_predicted_test, labels_encoder_test
                    )**0.5 * divided

                    # MAE score
                    data_train[seed_idx * 4 + fold_idx, 2] = metrics.mean_absolute_error(
                        label_predicted, labels_encoder
                    ) * divided

                    data_test[seed_idx * 4 + fold_idx, 2] = metrics.mean_absolute_error(
                        label_predicted_test, labels_encoder_test
                    ) * divided

                    data_train[seed_idx * 4 + fold_idx, 3], data_train[seed_idx * 4 + fold_idx, 4] = ft.valueGap(
                        label_predicted, labels_encoder, divided
                    )

                    data_test[seed_idx * 4 + fold_idx, 3], data_test[seed_idx * 4 + fold_idx, 4] = ft.valueGap(
                        label_predicted_test, labels_encoder_test, divided
                    )
                    if WDNorm:
                        # WasserteinDistance
                        data_train[seed_idx * 4 + fold_idx, 5] = wasserstein_distance(labels_encoder / np.sum(
                            labels_encoder), label_predicted / np.sum(label_predicted)) * divided

                        data_test[seed_idx * 4 + fold_idx, 5] = wasserstein_distance(labels_encoder_test / np.sum(
                            labels_encoder_test), label_predicted_test / np.sum(label_predicted_test)) * divided
                    else:
                        data_train[seed_idx * 4 + fold_idx, 5] = wasserstein_distance(
                            labels_encoder, label_predicted) * divided

                        data_test[seed_idx * 4 + fold_idx, 5] = wasserstein_distance(
                            labels_encoder_test, label_predicted_test) * divided

                    # Get Top Genes of each class

                    # method = 'Shap'   # (SHapley Additive exPlanation) needs a nb_samples
                    nb_samples = 300  # Randomly choose nb_samples to calculate their Shap Value, time vs nb_samples seems exponential
                    # method = 'Captum_ig'   # Integrated Gradients
                    method = "Captum_dl"  # Deeplift
                    # method = 'Captum_gs'  # GradientShap

                    tps1 = time.perf_counter()
                    print("Running topGenes...")
                    df_topGenes = ft.topGenes(
                        X,
                        Y,
                        feature_names,
                        feature_len,
                        method,
                        nb_samples,
                        DEVICE,
                        net,
                        TOL
                    )
                    df_topGenes.index = df_topGenes.iloc[:, 0]
                    print("topGenes finished")
                    tps2 = time.perf_counter()

                    if fold_idx != 0:  # not first fold need to get previous topGenes
                        df = pd.read_csv(
                            "{}{}_topGenes_{}_{}.csv".format(
                                outputPath, str(
                                    TYPE_PROJ_NAME), method, str(nb_samples)
                            ),
                            sep=";",
                            header=0,
                            index_col=0,
                        )
                        df_topGenes.index = df_topGenes.iloc[:, 0]
                        df_topGenes = df.join(
                            df_topGenes.iloc[:, 1], lsuffix="_",)

                    df_topGenes.to_csv(
                        "{}{}_topGenes_{}_{}.csv".format(
                            outputPath, str(
                                TYPE_PROJ_NAME), method, str(nb_samples)
                        ),
                        sep=";",
                    )
                    tps2 = time.perf_counter()
                    print("Execution time topGenes  : ", tps2 - tps1)

                    if DoSparsity:
                        mat_in = net.state_dict()["encoder.0.weight"]
                        mat_col_sparsity = ft.sparsity_col(
                            mat_in, device=DEVICE)
                        sparsity_matrix[seed_idx * 4 +
                                        fold_idx, 0] = mat_col_sparsity

                    if DoTopFeatures:

                        weightsF, spasity_w = fnp.weights_and_sparsity(
                            net.encoder, TOL)
                        layer_list = [x for x in weightsF.values()]
                        # Chercher les colonnes qui non pas que des 0
                        non_zero_columns = ~np.all(layer_list[0] == 0, axis=0)
                        indices_non_zero_columns = np.where(
                            non_zero_columns)[0]
                        featuresSelected = feature_names[indices_non_zero_columns]
                        normL = LA.norm(layer_list[0], 2, axis=0)
                        normL = normL / max(normL)
                        normFeaturesSelected = normL[indices_non_zero_columns]
                        # dfFeatureSelected = pd.DataFrame(featuresSelected, columns=['Fold'+str(fold_idx)])
                        dfFeatureSelected = pd.DataFrame({
                            'Fold': featuresSelected,
                            'NormL2 ': normFeaturesSelected
                        })

                        if fold_idx != 0:
                            df = pd.read_csv(
                                "{}{}_topFeatures_NormL2.csv".format(
                                    outputPath, str(TYPE_PROJ_NAME)
                                ),
                                sep=";",
                                header=0,
                                index_col=0,
                            )
                            dfFeatureSelected = df.merge(dfFeatureSelected.iloc[:, 0:], how='left',
                                                         on='Fold', suffixes=(f"{fold_idx-1}", f"{fold_idx}"))
                        dfFeatureSelected.fillna(0, inplace=True)
                        dfFeatureSelected.to_csv('{}{}_topFeatures_NormL2.csv'.format(outputPath,
                                                                                      str(TYPE_PROJ_NAME)), sep=";")
                        #     dfFeatureSelected = df.join(dfFeatureSelected.iloc[:, 0:], lsuffix=f"_{fold_idx-1}")

                        # dfFeatureSelected.to_csv('{}{}_FeaturesSelectedFold.csv'.format(outputPath, str(TYPE_PROJ_NAME)),sep=";")

                    weights, spasity_w = fnp.weights_and_sparsity(net, TOL)
                    spasity_percentage_entry = {}
                    for keys in weights.keys():
                        spasity_percentage_entry[keys] = spasity_w[keys] * 100
                    print("spasity % of all layers entry \n",
                          spasity_percentage_entry)
                    layer_list = [x for x in weights.values()]
                    ft.show_img(layer_list, file_name)

                    print("---------------- Start Testing on the 20% ----------------")

                    # load les dl
                    dtest = ft.LoadDataset(
                        X_test, y_test, list(range(len(X_test))))
                    # _, test_set = torch.utils.data.random_split(dtest, [0])
                    test_dl = torch.utils.data.DataLoader(dtest, batch_size=1)

                    (
                        data_encoder_test20,
                        integrated_brier_score,
                        ev,
                    ) = ft.runBestNet_survie(
                        test_dl,
                        outputPath,
                        fold_idx,
                        net,
                        feature_names,
                        test_len,
                        best_model,
                    )

                    if seed == SEEDS[-1]:
                        if fold_idx == 0:
                            # Initialize variables for final aggregation
                            integrated_brier_scores = [integrated_brier_score]
                            LP_test = data_encoder_test20.detach().cpu().numpy()
                        else:
                            # Aggregate results across folds
                            integrated_brier_scores.append(
                                integrated_brier_score)
                            LP_test = np.concatenate(
                                (LP_test, data_encoder_test20.detach().cpu().numpy())
                            )
                        # Compute average Integrated Brier Score over all folds
                        mean_brier_score = np.mean(integrated_brier_scores)
                        print(f"Mean Integrated Brier Score: ")
                        print(f"{mean_brier_score:.4f}")

                        # Extracting hazards from data_encoder_test for visualization
                        # Assuming hazards are in the first column
                        hazards = LP_test[:, 0]
                        durations = LP_test[:, 1]  # True durations
                        events = LP_test[:, 2]  # Event indicators

                        # Evaluating survival metrics and plotting survival curves (optional)
                        survival_times = np.linspace(
                            np.min(durations), np.max(durations), 100)
                        brier_scores = ev.brier_score(survival_times)

                        plt.figure()
                        plt.plot(survival_times, brier_scores,
                                 label="Brier Score")
                        plt.xlabel("Time")
                        plt.ylabel("Brier Score")
                        plt.title("Brier Score Over Time")
                        plt.legend()
                        plt.show()
                        plt.savefig(f'brier_score_{seed}.png')

                    data_encoder_test20 = data_encoder_test20.cpu().detach().numpy()
                    # mse score

                    label_predicted_test20 = data_encoder_test20[:, 0]
                    labels_encodertest20 = data_encoder_test20[:, -1]
                    data_finalTest[seed_idx * 4 + fold_idx, 0] = metrics.mean_squared_error(
                        label_predicted_test20, labels_encodertest20
                    )

                    data_finalTest[seed_idx * 4 + fold_idx, 1] = metrics.mean_squared_error(
                        label_predicted_test20, labels_encodertest20
                    )**0.5 * divided

                    # MAE score
                    data_finalTest[seed_idx * 4 + fold_idx, 2] = metrics.mean_absolute_error(
                        label_predicted_test20, labels_encodertest20
                    ) * divided

                    data_finalTest[seed_idx * 4 + fold_idx, 3], data_finalTest[seed_idx * 4 + fold_idx, 4] = ft.valueGap(
                        label_predicted_test20, labels_encodertest20, divided
                    )

                    if WDNorm:
                        data_finalTest[seed_idx * 4 + fold_idx, 5] = wasserstein_distance(
                            labels_encodertest20 /
                            np.sum(labels_encodertest20), label_predicted_test20 /
                            np.sum(label_predicted_test20),
                        ) * divided
                    else:
                        data_finalTest[seed_idx * 4 + fold_idx, 5] = wasserstein_distance(
                            labels_encodertest20, label_predicted_test20,
                        ) * divided

                # Moyenne sur les SEED
                if DoTopGenes:
                    df = pd.read_csv(
                        "{}{}_topGenes_{}_{}.csv".format(
                            outputPath, str(
                                TYPE_PROJ_NAME), method, str(nb_samples)
                        ),
                        sep=";",
                        header=0,
                        index_col=0,
                    )
                    df_val = df.values[1:, 1:].astype(float)
                    df_mean = df_val.mean(axis=1).reshape(-1, 1)
                    df_std = df_val.std(axis=1).reshape(-1, 1)
                    foldCol = ["Fold "+str(i) for i in range(nfolds)]
                    df = pd.DataFrame(
                        np.concatenate(
                            (df.values[1:, :], df_mean, df_std), axis=1),
                        columns=[
                            "Features"]
                        + foldCol +
                        [
                            "Mean",
                            "Std",
                        ],
                    )
                    df_topGenes = df
                    df_topGenes = df_topGenes.sort_values(
                        by="Mean", ascending=False)
                    df_topGenes = df_topGenes.reindex(
                        columns=[
                            "Features",
                            "Mean"]
                        + foldCol +
                        [
                            "Std",
                        ]
                    )
                    df_topGenes.to_csv(
                        "{}{}_topGenes_{}_{}.csv".format(
                            outputPath, str(
                                TYPE_PROJ_NAME), method, str(nb_samples)
                        ),
                        sep=";",
                        index=0,
                    )

                    if seed == SEEDS[0]:
                        df_topGenes_mean = df_topGenes.iloc[:, 0:2]
                        df_topGenes_mean.index = df_topGenes.iloc[:, 0]
                    else:
                        df = pd.read_csv(
                            "{}{}_topGenes_Mean_{}_{}.csv".format(
                                outputPath, str(
                                    TYPE_PROJ_NAME), method, str(nb_samples)
                            ),
                            sep=";",
                            header=0,
                            index_col=0,
                        )
                        df_topGenes.index = df_topGenes.iloc[:, 0]
                        df_topGenes_mean = df.join(
                            df_topGenes.iloc[:, 1], lsuffix=seed)

                    df_topGenes_mean.to_csv(
                        "{}{}_topGenes_Mean_{}_{}.csv".format(
                            outputPath, str(
                                TYPE_PROJ_NAME), method, str(nb_samples)
                        ),
                        sep=";",
                    )

                    if DoSparsity:
                        mat_in = net.state_dict()["encoder.0.weight"]
                        mat_col_sparsity = ft.sparsity_col(
                            mat_in, device=DEVICE)
                        sparsity_matrix_retraing[seed_idx *
                                                 4 + fold_idx, 0] = mat_col_sparsity

                seed_idx += 1

            if DoTopFeatures:
                df = pd.read_csv(
                    "{}{}_topFeatures_NormL2.csv".format(
                        outputPath, str(TYPE_PROJ_NAME)
                    ),
                    sep=";",
                    header=0,
                    index_col=0,
                )
                df_val = df.values[0:, 1:].astype(float)
                df_mean = df_val.mean(axis=1).reshape(-1, 1)
                df_std = df_val.std(axis=1).reshape(-1, 1)
                foldCol = ["Fold "+str(i) for i in range(nfolds)]
                df = pd.DataFrame(
                    np.concatenate(
                        (df.values[0:, :], df_mean, df_std), axis=1),
                    columns=[
                        "Features"]
                    + foldCol +
                    [
                        "Mean",
                        "Std",
                    ],
                )
                df_topFeatures = df
                df_topFeatures = df_topFeatures.sort_values(
                    by="Mean", ascending=False)
                df_topFeatures = df_topFeatures.reindex(
                    columns=[
                        "Features",
                        "Mean"]
                    + foldCol +
                    ["Std",
                     ]
                )
                df_topFeatures.to_csv(
                    "{}{}_topFeatures_NormL2.csv".format(
                        outputPath, str(TYPE_PROJ_NAME)
                    ),
                    sep=";",
                    index=0,
                )

                if seed == SEEDS[0]:
                    df_topFeatures_mean = df_topFeatures.iloc[:, 0:2]
                    df_topFeatures_mean.index = df_topFeatures.iloc[:, 0]
                else:
                    df = pd.read_csv(
                        "{}{}_topFeatures_Mean_NormL2.csv".format(
                            outputPath, str(TYPE_PROJ_NAME)
                        ),
                        sep=";",
                        header=0,
                        index_col=0,
                    )
                    df_topFeatures.index = df_topFeatures.iloc[:, 0]

                    df_topFeatures_mean = df.merge(df_topFeatures.iloc[:, 1], how='left',
                                                   on='Features', suffixes=(f"{seed-1}", f"{seed}"))

                    # df.join(df_topFeatures.iloc[:, 1], lsuffix=seed)

                df_topFeatures_mean.to_csv(
                    "{}{}_topFeatures_Mean_NormL2.csv".format(
                        outputPath, str(TYPE_PROJ_NAME)
                    ),
                    sep=";",
                )

            seed_idx += 1

            # metrics
            df_metricsTrain, df_metricsTest = ft.packMetricsResult(
                data_train, data_test, nfolds * len(SEEDS)
            )

            reg_metrics = ["MSE", "RMSE", "MAE",
                           "Negative gap", "Positive gap", "WD"]
            df_metricsTrain_classif = df_metricsTrain[reg_metrics]
            df_metricsTest_classif = df_metricsTest[reg_metrics]
            print("\nMetrics Train")
            # print(df_metricsTrain_clustering)
            print(df_metricsTrain_classif)
            print("\nMetrics Test")
            # print(df_metricsTest_clustering)
            print(df_metricsTest_classif)

            # # metrics
            # df_metricsreTrain, df_metricsRetest = ft.packMetricsResult(
            #     data_retrain, data_retest, nfolds * len(SEEDS)
            # )

            # df_metricsreTrain_classif = df_metricsreTrain[reg_metrics]
            # df_metricsRetest_classif = df_metricsRetest[reg_metrics]
            # print("\nMetrics Retrain")
            # #print(df_metricsTrain_clustering)
            # print(df_metricsreTrain_classif)
            # print("\nMetrics Retest")
            # # print(df_metricsTest_clustering)
            # print(df_metricsRetest_classif)

            df_metricsFinalTest = ft.packMetric(
                data_finalTest, nfolds*len(SEEDS))
            df_metrics_FinalTest = df_metricsFinalTest[reg_metrics]
            print("\nMetrics Final Test")
            # print(df_metricsTrain_clustering)
            print(df_metrics_FinalTest)

            if DoSparsity:
                # make df for the sparsity:
                columns = (
                    ["Sparsity"]
                )
                ind_df = ["Fold " + str(x + 1)
                          for x in range(nfolds * len(SEEDS))]

                df_sparcity = pd.DataFrame(
                    sparsity_matrix, index=ind_df, columns=columns)
                df_sparcity.loc["Mean"] = df_sparcity.apply(lambda x: x.mean())
                df_sparcity.loc["Std"] = df_sparcity.apply(lambda x: x.std())
                sparc = df_sparcity.loc["Mean", "Sparsity"]
                print('\n Sparsity on the encoder for the training')
                print(df_sparcity)
                print(f'\n On average we have ')
                print(f'{round(100-sparc)}')
                print('% features selected, thus ')
                print(
                    f"{round(((100-sparc)/100)*feature_len)}")
                print(f"features")

            if DoTopGenes:
                df = pd.read_csv(
                    "{}{}_topGenes_Mean_{}_{}.csv".format(
                        outputPath, str(
                            TYPE_PROJ_NAME), method, str(nb_samples)
                    ),
                    sep=";",
                    header=0,
                    index_col=0,
                )
                df_val = df.values[:, 1:].astype(float)
                df_mean = df_val.mean(axis=1).reshape(-1, 1)
                df_std = df_val.std(axis=1).reshape(-1, 1)
                df_meanstd = df_std / df_mean
                col_seed = ["Seed " + str(i) for i in SEEDS]
                df = pd.DataFrame(
                    np.concatenate(
                        (df.values[:, :], df_mean, df_std, df_meanstd), axis=1),
                    columns=["Features"] + col_seed + ["Mean", "Std", "Mstd"],
                )
                df_topGenes = df
                df_topGenes = df_topGenes.sort_values(
                    by="Mean", ascending=False)
                df_topGenes = df_topGenes.reindex(
                    columns=["Features", "Mean"] + col_seed + ["Std", "Mstd"]
                )
                df_topGenes.to_csv(
                    "{}{}_topGenes_Mean_{}_{}.csv".format(
                        outputPath, str(
                            TYPE_PROJ_NAME), method, str(nb_samples)
                    ),
                    sep=";",
                    index=0,
                )

            weights_entry, spasity_w_entry = fnp.weights_and_sparsity(net, TOL)
            # spasity_percentage_entry = {}
            # for keys in spasity_w_entry.keys():
            #     spasity_percentage_entry[keys] = spasity_w_entry[keys] * 100
            # print("spasity % of all layers entry \n", spasity_percentage_entry)
            # print("-----------------------")
            weights, spasity_w = fnp.weights_and_sparsity(net, TOL)

            layer_list = [x for x in weights.values()]
            titile_list = [x for x in spasity_w.keys()]
            # print(f"After Projection, Sum is: {np.sum(np.abs(weights_interim_enc['encoder.0.weight']))}")

            ft.show_img(layer_list, file_name)

            # some processing of the data needed for ploting the next graph
            # reorganize the captum selected feature to be in the same order as the selected feature of L2
            # for features selected by captum and not selected by L2, they are concatened at the end.
            match = df_topGenes[df_topGenes['Features'].isin(df_topFeatures['Features'])].set_index(
                'Features').reindex(df_topFeatures['Features']).reset_index()
            remain = df_topGenes[~df_topGenes['Features'].isin(
                df_topFeatures['Features'])]
            # Combine the two parts
            result = pd.concat([match, remain], ignore_index=True)
            # drop null value
            result.drop(result[result['Mean'] == 0].index, inplace=True)

            # plot selected feature of L2 and captum
            plt.figure()
            plt.plot(df_topFeatures["Mean"].to_numpy(), color="tab:blue")
            plt.plot(result["Mean"].to_numpy(), color="tab:orange")
            plt.xlabel("features")
            plt.ylabel("mean")
            plt.title("captum vs L2")
            plt.legend(["L2", "CAPTUM"])
            plt.grid()
            plt.show()

            # same as above but with barplot
            # bigger figure to fit all Features name
            plt.figure(figsize=(16, 9))
            plt.bar(df_topFeatures["Features"], df_topFeatures['Mean'],
                    width=0.4, label='L2', align='center')
            plt.bar(result["Features"], result['Mean'],
                    width=0.4, label='captum', align='edge')
            plt.legend()
            plt.xticks(ticks=range(
                len(result["Features"])), labels=result['Features'], rotation='vertical')
            plt.xlabel("features")
            plt.ylabel("mean")
            plt.title("captum vs L2")
            plt.show()

            selectedFeatures[ai][ei] = [ETA, len(df_topFeatures), len(result)]

            # Loss figure
            if os.path.exists(file_name.split(".")[0] + "_Loss_No_proj.npy") and os.path.exists(
                file_name.split(".")[0] + "_Loss_MaskGrad.npy"
            ):
                loss_no_proj = np.load(file_name.split(".")[
                                       0] + "_Loss_No_proj.npy")
                loss_with_proj = np.load(file_name.split(".")[
                                         0] + "_Loss_MaskGrad.npy")
                plt.figure()
                plt.title(file_name.split(".")[0] + " Loss")
                plt.xlabel("Epoch")
                plt.ylabel("TotalLoss")
                plt.plot(loss_no_proj, label="No projection")
                plt.plot(loss_with_proj, label="With projection ")
                plt.legend()
                plt.show()

            if SAVE_FILE:
                df_metricsFinalTest.to_csv(
                    "{}{}_acctest.csv".format(outputPath, str(TYPE_PROJ_NAME)), sep=";"
                )
                if DoSparsity:
                    df_sparcity.to_csv(
                        "{}{}_sparsity.csv".format(outputPath, str(TYPE_PROJ_NAME)), sep=";"
                    )

            end_time = time.time()
            execution_time = end_time-start_time
            print(f"Execution time: {execution_time} seconds")
            # retrieving data :
            datas_finaltest.append(data_finalTest)
            datas_retest.append(data_retest)
            datas_retrain.append(data_retrain)
            datas_test.append(data_test)
            datas_train.append(data_train)
            sparcities.append(sparsity_matrix)

    print("\nComputing finished !")

    # save data of couple eta and alpha:
    if (SAVE_FILE):
        i = 0
        for eta in ETAS:
            for alph in ALPHAS:
                np.savetxt("{}{}_data_train_eta_{}_alpha_{}.csv".format(
                    outputPath, str(TYPE_PROJ_NAME), eta, alph), datas_train[i], delimiter=";")
                np.savetxt("{}{}_data_test_eta_{}_alpha_{}.csv".format(
                    outputPath, str(TYPE_PROJ_NAME), eta, alph), datas_test[i], delimiter=";")
                np.savetxt("{}{}_data_finaltest_eta_{}_alpha_{}.csv".format(
                    outputPath, str(TYPE_PROJ_NAME), eta, alph), datas_finaltest[i], delimiter=";")
                np.savetxt("{}{}_data_sparities_eta_{}_alpha_{}.csv".format(
                    outputPath, str(TYPE_PROJ_NAME), eta, alph), sparcities[i], delimiter=";")
                i += 1

    for ai, alph in enumerate(ALPHAS):
        np.savetxt("{}{}_data_features_alpha_{}.csv".format(outputPath, str(
            TYPE_PROJ_NAME), alph), selectedFeatures[ai], delimiter=";")

# %%

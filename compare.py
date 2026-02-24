from lifelines import KaplanMeierFitter
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torchtuples as tt
from sklearn.preprocessing import StandardScaler, scale
from lifelines import CoxPHFitter
from sksurv.metrics import concordance_index_censored, brier_score
from sksurv.util import Surv
import lifelines
import os

import functions_.functions_torch_regression_V3 as ft
import functions_.functions_network_pytorch as fnp
import functions_.functions_DeepSurv as fds

# Set random seed for reproducibility
np.random.seed(4234)
torch.manual_seed(423)

# Constants
REPOSITORY = "results_stat/compare/"
FILE_NAME = './TimeInterval.csv'
DO_SCALE, DO_LOG, DO_ROW_NORM, TEST_SIZE = True, False, True, None
N_HIDDEN = 300
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_data():
    """Loads and preprocesses the dataset."""
    return ft.ReadDataCV_surv(FILE_NAME, test_size=TEST_SIZE, doScale=DO_SCALE, doLog=DO_LOG, doRowNorm=DO_ROW_NORM)


def preprocess_features(X, X_test):
    """Scales and normalizes the features."""
    scaler = StandardScaler()
    X_train, X_test = scaler.fit_transform(X), scaler.transform(X_test)
    X_train, X_test = scale(X_train, axis=0), scale(X_test, axis=0)
    X_train -= np.mean(X_train, axis=1, keepdims=True)
    X_test -= np.mean(X_test, axis=1, keepdims=True)
    return torch.tensor(X_train, dtype=torch.float32), torch.tensor(X_test, dtype=torch.float32)


def load_model(feature_len):
    """Loads a pre-trained DeepSurv model."""
    state_dict = torch.load(f"{REPOSITORY}/best_net")
    net = fds.MLP(feature_len, N_HIDDEN, 1, batch_norm=True,
                  dropout=0.1, output_bias=False).to(DEVICE).float()
    net.load_state_dict(state_dict)
    net.eval()
    return fds.Custom_CoxPH(net, tt.optim.Adam)


def train_cox_model(df_train):
    """Fits a Cox Proportional Hazards model."""
    cph = CoxPHFitter()
    cph.fit(df_train, duration_col="Duration", event_col="Event")
    return cph


def evaluate_model(model, X_train, Y, X_test, durations):
    """Computes the baseline hazards and survival function predictions."""
    model.compute_baseline_hazards(X_train, Y, max_duration=durations.max())
    print(model.net(torch.tensor(X_test, dtype=torch.float32).float().to(DEVICE)))

    return model.predict_surv_df(X_test, max_duration=durations.max())


def deepSurv_basic(X, X_fold, X_test, Y, Y_fold, y_test, times):
    """Trains a basic DeepSurv model and returns survival predictions."""
    feature_len = X.shape[1]
    net = ft.buildNet(feature_len, 'relu', 'DeepSurv', DEVICE, N_HIDDEN, True)
    model = fds.Custom_CoxPH(net, tt.optim.Adam)

    X, X_test = map(lambda x: torch.tensor(
        x, dtype=torch.float32), [X, X_test])
    y_test = (torch.tensor(y_test[1], dtype=torch.float32), torch.tensor(
        y_test[0], dtype=torch.float32))

    lr_finder = model.lr_finder(X, Y, batch_size=50, tolerance=10)
    model.optimizer.set_lr(lr_finder.get_best_lr())

    model.fit(X, Y, batch_size=50, epochs=50, callbacks=[
              tt.callbacks.EarlyStopping(patience=32)], val_data=(X_fold, Y_fold))
    model.compute_baseline_hazards()
    return model.predict_surv_df(X_test), model.predict(X_test)


def deepSurv_with_projection(X, X_fold, X_test, Y, Y_fold, y_test, times, feature_names, foldid):
    """Trains a DeepSurv model with projection"""
    # if REPOSITORY does not exist, create it
    if not os.path.exists(REPOSITORY):
        os.makedirs(REPOSITORY)

    SEEDS = [5]
    def criterion_regression(u, v): return wasserstein_distance(
        u, v) + alpha * mse(u, v)

    TYPE_PROJ = ft.bilevel_proj_l1Inftyball  # projection bilevel l1,inf
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # generate generic patient name "numbers" for all Y
    names = [str(i) for i in range(len(Y[0]))]

    Y = (Y[1].detach().cpu().numpy(), Y[0].detach().cpu().numpy())
    dtrain = ft.LoadDataset(X, (Y[0], Y[1]), np.array(names))
    train_dl = torch.utils.data.DataLoader(
        dtrain, batch_size=50, shuffle=True)
    Y_fold = (Y_fold[1].detach().cpu().numpy(),
              Y_fold[0].detach().cpu().numpy())
    names = [str(i) for i in range(len(Y_fold[0]))]
    dfold = ft.LoadDataset(X_fold, Y_fold, np.array(names))
    fold_dl = torch.utils.data.DataLoader(dfold, batch_size=1, shuffle=True)
    data_encoder, net, best_model = ft.training(
        seed=42,
        feature_len=len(feature_names),
        TYPE_ACTIVATION='silu',
        DEVICE=DEVICE,
        n_hidden=300,
        norm=False,
        feature_names=feature_names,
        GRADIENT_MASK=True,
        net_name='DeepSurv',
        LR=0.0001,
        criterion_regression=criterion_regression,
        train_dl=train_dl,
        train_len=len(Y[0]),
        gaussianKDE=None,
        test_dl=fold_dl,
        test_len=len(Y_fold[0]),
        outputPath=REPOSITORY,
        TYPE_PROJ=TYPE_PROJ,
        SEEDS=SEEDS,
        fold_idx=foldid,
        nfolds=1,
        N_EPOCHS=100,
        N_EPOCHS_MASKGRAD=50,
        DO_PROJ_MIDDLE=False,
        ETA=0.5,
        AXIS=1,
        TOL=1e-3,
        lambda_wass=0.0,
    )
    data_encoder = data_encoder.cpu().detach().numpy()
    print("done training")
    (
        data_encoder_test20,
        integrated_brier_score,
        ev
    ) = ft.runBestNet_survie(
        test_dl=fold_dl,
        outputPath=REPOSITORY,
        nfold=foldid,
        net=net,
        feature_name=feature_names,
        test_len=len(Y_fold[0]),
        model=best_model,
    )


def compute_c_index(y_test, hazards, cph, df_test):
    """Computes the concordance index for DeepSurv, DeepSurv Basic, and Cox models."""
    c_index_deepsurv = concordance_index_censored(
        np.array(y_test[0], dtype=bool), y_test[1], hazards)[0]
    c_index_cox = lifelines.utils.concordance_index(
        np.array(y_test[1], dtype=float), cph.predict_expectation(
            df_test).to_numpy().flatten(), np.array(y_test[0], dtype=bool)
    )
    return c_index_deepsurv, c_index_cox

    # Function to plot Brier scores with confidence intervals


def plot_brier_scores(times, mean_brier_ds, mean_brier_cox, mean_brier_basic,
                      ci_ds_lower, ci_ds_upper, ci_cox_lower, ci_cox_upper,
                      ci_basic_lower, ci_basic_upper):
    """Plots the Brier scores for DeepSurv, DeepSurv Basic, and Cox models, with confidence intervals."""
    plt.figure(figsize=(10, 5))

# Plot DeepSurv with Projection Brier Score and its confidence interval
    plt.plot(times, mean_brier_ds,
             label='DeepSurv with proj Brier Score', color='blue')
    plt.fill_between(times, ci_ds_lower, ci_ds_upper, color='blue', alpha=0.2)

# Plot DeepSurv Basic Brier Score and its confidence interval
    plt.plot(times, mean_brier_basic,
             label='DeepSurv Basic Brier Score', color='orange')
    plt.fill_between(times, ci_basic_lower, ci_basic_upper,
                     color='orange', alpha=0.2)

# Plot Cox Brier Score and its confidence interval
    plt.plot(times, mean_brier_cox, label='Cox Brier Score', color='green')
    plt.fill_between(times, ci_cox_lower, ci_cox_upper,
                     color='green', alpha=0.2)

    plt.ylim(-0.1, 1.1)
    plt.xlabel('Time')
    plt.ylabel('Brier Score')
    plt.title('Brier Score Comparison with Confidence Intervals')
    plt.legend()
    plt.show()


def compute_brier_scores(surv, Y, y_test, cph_surv, basic, times):
    """Computes the Brier scores for DeepSurv, DeepSurv Basic, and Cox models."""

    # Print types and shapes for debugging

    print(cph_surv)
    print(surv)
    print(basic)
    # Convert to structured survival data
    Y_structured = Surv.from_arrays(Y[1], Y[0])
    y_test_structured = Surv.from_arrays(y_test[0], y_test[1])

    # Create a complete list of durations (including the ones in `times`)
    all_durations = sorted(set(surv.index).union(set(times)))

    # Reindex the dataframes to include all durations, adding rows with NaN for missing durations
    surv_reindexed = surv.reindex(all_durations)
    basic_reindexed = basic.reindex(all_durations)

    # Fill missing values with forward fill (ffill)
    surv_filled = surv_reindexed.ffill(axis=0)
    basic_filled = basic_reindexed.ffill(axis=0)

    # Fill remaining NaN values with backward fill (bfill)
    surv_filled = surv_filled.bfill(axis=0)
    basic_filled = basic_filled.bfill(axis=0)

    # Select only rows that correspond to the times in the provided list
    surv_filtered = surv_filled.loc[surv_filled.index.isin(times)]
    basic_filtered = basic_filled.loc[basic_filled.index.isin(times)]

    print(surv_filtered)
    print(cph_surv)
    print(basic_filtered)

    # Compute the Brier scores for each model
    brier_cox = brier_score(
        y_test_structured, y_test_structured, cph_surv.T, times)
    brier_ds = brier_score(
        Y_structured, y_test_structured, surv_filtered.T, times)
    brier_basic = brier_score(
        Y_structured, y_test_structured, basic_filtered.T, times)

    return brier_ds, brier_cox, brier_basic

    # Calculate mean and confidence intervals for Brier scores


def calculate_confidence_intervals(brier_scores):
    brier_mean = np.mean(brier_scores, axis=0)
    brier_sem = np.std(brier_scores, axis=0) / np.sqrt(len(brier_scores))
    ci_lower = brier_mean - 1.96 * brier_sem
    ci_upper = brier_mean + 1.96 * brier_sem
    return brier_mean, ci_lower, ci_upper


# Main Execution
X, X_test, Y, y_test, feature_names, *_ = load_data()
X_train, X_test = preprocess_features(X, X_test)
durations, events = Y[1].astype(np.float32), Y[0].astype(np.float32)
Y = (torch.tensor(durations, dtype=torch.float32),
     torch.tensor(events, dtype=torch.float32))
times = np.linspace(int(min(durations)), int(max(durations)),
                    num=int(max(durations)) - int(min(durations)) + 1)
print(times)

# Initialize KFold
# Adjust the number of splits as needed
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Lists to hold results
c_index_deepsurv_list = []
c_index_cox_list = []
integrated_brier_ds_list = []
integrated_brier_cox_list = []
integrated_brier_basic_list = []

# Lists for Brier scores for confidence intervals
brier_ds_list = []
brier_cox_list = []
brier_basic_list = []

# Cross-validation
for train_index, val_index in kf.split(X):

    predict_time = np.sort(np.unique(y_test[1]))
    X_train_fold, X_val_fold = X[train_index], X[val_index]
    Y_train_fold = (Y[0][train_index], Y[1][train_index])
    Y_val_fold = (Y[0][val_index], Y[1][val_index])

    X_train_fold = torch.tensor(X_train_fold, dtype=torch.float32).float()
    X_val_fold = torch.tensor(X_val_fold, dtype=torch.float32).float()
    print(times)
    # Load and evaluate DeepSurv model
    deepSurv_with_projection(X_train_fold, X_val_fold, X_test,
                             Y_train_fold, Y_val_fold, y_test, times, feature_names, 0)

    model = load_model(len(feature_names))
    model.compute_baseline_hazards(
        X_train_fold, Y_train_fold)
    print(X_test.shape)
    hazards = [h[0] for h in model.predict(X_test)]
    surv = evaluate_model(model, X_train_fold,
                          Y_train_fold, X_test, times)
    print(surv.shape)
    # Prepare training DataFrame for Cox model
    Y_np = (Y_train_fold[0].cpu().numpy(), Y_train_fold[1].cpu().numpy())
    df_train = pd.DataFrame(
        np.concatenate([X_train_fold, np.column_stack(Y_np)], axis=1),
        columns=list(feature_names) + ["Duration", "Event"]
    )
    Y_test_np = (y_test[1], y_test[0])
    df_test = pd.DataFrame(np.concatenate([X_test, np.column_stack(
        Y_test_np)], axis=1), columns=list(feature_names) + ["Duration", "Event"])

    # Remove features with zero variance
    events = df_train['Event'].astype(bool)
    cols_to_remove = [col for col in df_train.columns if col not in ['Duration', 'Event'] and (
        df_train.loc[events, col].var() < 0.5 or df_train.loc[~events, col].var() < 0.5)]

    df_train.drop(columns=cols_to_remove, inplace=True)
    df_test.drop(columns=cols_to_remove, inplace=True)

    # Train Cox model
    cph = train_cox_model(df_train)
    print(df_test)
    cph_surv = cph.predict_survival_function(df_test, times=times)

    # Train basic DeepSurv model
    basic_surv, basic_hazards = deepSurv_basic(
        X_train_fold, X_val_fold,  X_test, Y_train_fold, Y_val_fold, y_test, times)

    # Compute concordance index
    c_index_deepsurv, c_index_cox = compute_c_index(
        y_test, hazards, cph, df_test)
    c_index_basic = concordance_index_censored(
        np.array(y_test[0], dtype=bool), y_test[1], [h[0] for h in basic_hazards])[0]

    # Compute Brier scores
    brier_ds, brier_cox, brier_basic = compute_brier_scores(
        surv, Y, y_test, cph_surv, basic_surv, times)

    # Store Brier scores for confidence intervals
    brier_ds_list.append(brier_ds[1])
    brier_cox_list.append(brier_cox[1])
    brier_basic_list.append(brier_basic[1])

    # Calculate integrated Brier scores
    integrated_brier_ds = np.trapz(brier_ds[1], brier_ds[0]) / brier_ds[0][-1]
    integrated_brier_cox = np.trapz(
        brier_cox[1], brier_cox[0]) / brier_cox[0][-1]
    integrated_brier_basic = np.trapz(
        brier_basic[1], brier_basic[0]) / brier_basic[0][-1]

    # Append results
    c_index_deepsurv_list.append(c_index_deepsurv)
    c_index_cox_list.append(c_index_cox)
    integrated_brier_ds_list.append(integrated_brier_ds)
    integrated_brier_cox_list.append(integrated_brier_cox)
    integrated_brier_basic_list.append(integrated_brier_basic)

# Convert lists to numpy arrays for calculations
print(brier_ds_list)
print(brier_cox_list)
print(brier_basic_list)


brier_ds_array = np.array(brier_ds_list)
brier_cox_array = np.array(brier_cox_list)
brier_basic_array = np.array(brier_basic_list)

mean_brier_ds, ci_ds_lower, ci_ds_upper = calculate_confidence_intervals(
    brier_ds_array)
mean_brier_cox, ci_cox_lower, ci_cox_upper = calculate_confidence_intervals(
    brier_cox_array)
mean_brier_basic, ci_basic_lower, ci_basic_upper = calculate_confidence_intervals(
    brier_basic_array)

# Create a DataFrame to hold the results
results = pd.DataFrame({
    'Model': ['Cox Model', 'DeepSurv', 'DeepSurv with Projection'],
    'C-index': [np.mean(c_index_cox_list), np.mean(c_index_basic), np.mean(c_index_deepsurv_list)],
    'Integrated Brier Score': [np.mean(integrated_brier_cox_list), np.mean(integrated_brier_basic_list), np.mean(integrated_brier_ds_list)]
})

# Print the results in a clean format
print(results.to_string(index=False))

# Call the plot function with the calculated means and confidence intervals
plot_brier_scores(times,
                  mean_brier_ds, mean_brier_cox, mean_brier_basic,
                  ci_ds_lower, ci_ds_upper,
                  ci_cox_lower, ci_cox_upper,
                  ci_basic_lower, ci_basic_upper)

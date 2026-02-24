
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

import functions_.functions_torch_regression_V3 as ft
import functions_.functions_network_pytorch as fnp
import functions_.functions_DeepSurv as fds

# Set random seed for reproducibility
np.random.seed(1234)
torch.manual_seed(123)

# Constants
REPOSITORY = "results_stat"
FILE_NAME = './TimeInterval.csv'
DO_SCALE, DO_LOG, DO_ROW_NORM, TEST_SIZE = False, False, False, None
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
    return model.predict_surv_df(X_test, max_duration=durations.max())


def deepSurv_basic(X, X_test, Y, y_test, times):
    print("lets go")
    """Trains a basic DeepSurv model and returns survival predictions."""
    feature_len = X.shape[1]
    net = ft.buildNet(feature_len, 'relu', 'DeepSurv', DEVICE, N_HIDDEN, True)
    model = fds.Custom_CoxPH(net, tt.optim.Adam)

    X, X_test = map(lambda x: torch.tensor(
        x, dtype=torch.float32), [X, X_test])
    # split X and Y into training and validation sets
    # 90% of the data is used for training, 10% for validation
    prop = 0.1
    X, X_val = X[:-int(prop * len(X))], X[-int(prop * len(X)):]
    print(X.shape, X_val.shape)

    Y_temp = (Y[0][:-int(prop * len(Y[0]))], Y[1][:-int(prop * len(Y[1]))])
    y_val = (torch.tensor(Y[0][-int(prop * len(Y[0])):], dtype=torch.float32), torch.tensor(
        Y[1][-int(prop * len(Y[1])):], dtype=torch.float32))
    Y = Y_temp
    print(Y[0].shape, Y[1].shape)
    print(y_val[0].shape, y_val[1].shape)
    y_test = (torch.tensor(y_test[1], dtype=torch.float32), torch.tensor(
        y_test[0], dtype=torch.float32))

    lr_finder = model.lr_finder(X, Y, batch_size=256, tolerance=10)
    model.optimizer.set_lr(lr_finder.get_best_lr())

    model.fit(X, Y, batch_size=256, epochs=200, callbacks=[
              tt.callbacks.EarlyStopping(patience=64)], val_data=(X_val, y_val))
    model.compute_baseline_hazards()
    return model.predict_surv_df(X_test), model.predict(X_test)


def compute_c_index(y_test, hazards, cph, df_test):
    """Computes the concordance index for DeepSurv, DeepSurv Basic, and Cox models."""
    c_index_deepsurv = concordance_index_censored(
        np.array(y_test[0], dtype=bool), y_test[1], hazards)[0]
    c_index_cox = lifelines.utils.concordance_index(
        np.array(y_test[1], dtype=float), cph.predict_expectation(
            df_test).to_numpy().flatten(), np.array(y_test[0], dtype=bool)
    )
    return c_index_deepsurv, c_index_cox


def plot_brier_scores(times, brier_ds, brier_cox, brier_basic):
    """Plots the Brier scores for DeepSurv, DeepSurv Basic, and Cox models."""
    plt.figure(figsize=(10, 5))
    plt.plot(times, brier_ds, label='DeepSurv with proj Brier Score')
    plt.plot(times, brier_basic, label='DeepSurv Basic Brier Score')
    plt.plot(brier_cox[0], brier_cox[1], label='Cox Brier Score')
    plt.ylim(-0.1, 1.1)
    plt.xlabel('Time')
    plt.ylabel('Brier Score')
    plt.title('Brier Score Comparison')
    plt.legend()
    plt.show()


def compute_brier_scores(surv, Y, y_test, cph_surv, basic, times):
    """Computes the Brier scores for DeepSurv, DeepSurv Basic, and Cox models."""
    ev = fds.EvalSurv(surv, Y[0].detach().cpu().numpy(
    ), Y[1].detach().cpu().numpy(), censor_surv='km')
    brier_ds = np.array(ev.brier_score(times))

    ev_basic = fds.EvalSurv(basic, Y[0].detach().cpu(
    ).numpy(), Y[1].detach().cpu().numpy(), censor_surv='km')
    brier_basic = np.array(ev_basic.brier_score(times))

    Y_structured = Surv.from_arrays(Y[1], Y[0])
    y_test_structured = Surv.from_arrays(y_test[0], y_test[1])
    brier_cox = brier_score(Y_structured, y_test_structured, cph_surv.T, times)

    return brier_ds, brier_cox, brier_basic


# Main Execution
X, X_test, Y, y_test, feature_names, *_ = load_data()
durations, events = Y[1].astype(np.float32), Y[0].astype(np.float32)
times = np.linspace(durations.min(), durations.max(), 6000)
Y = (torch.tensor(durations, dtype=torch.float32),
     torch.tensor(events, dtype=torch.float32))
X_train, X_test = preprocess_features(X, X_test)

# Load and evaluate DeepSurv model
model = load_model(len(feature_names))
model.compute_baseline_hazards(X_train, Y)
hazards = [h[0] for h in model.predict(X_test)]
surv = evaluate_model(model, X_train, Y, X_test, durations)


# Train Cox model
print(type(Y), len(Y))
print([y.shape for y in Y])
Y_np = (Y[0].cpu().numpy(), Y[1].cpu().numpy())
df_train = pd.DataFrame(
    np.concatenate([X, np.column_stack(Y_np)], axis=1),
    columns=list(feature_names) + ["Duration", "Event"]
)
Y_test_np = (y_test[1], y_test[0])
df_test = pd.DataFrame(np.concatenate([X_test, np.column_stack(
    Y_test_np)], axis=1), columns=list(feature_names) + ["Duration", "Event"])

# Remove features with zero variance for events/non-events
# this is necessary for the Cox model to work
events = df_train['Event'].astype(bool)
cols_to_remove = [col for col in df_train.columns if col not in ['Duration', 'Event'] and (
    df_train.loc[events, col].var() < 0.1 or df_train.loc[~events, col].var() < 0.1)]
# print(f"Removing {len(cols_to_remove)} features with zero variance")
# for col in cols_to_remove:
#     print(f"Removing {col}")
df_train.drop(columns=cols_to_remove, inplace=True)
df_test.drop(columns=cols_to_remove, inplace=True)

cph = train_cox_model(df_train)
cph_surv = cph.predict_survival_function(df_test, times=times)

# Train basic DeepSurv model
basic_surv, basic_hazards = deepSurv_basic(X, X_test, Y, y_test, times)

# Compute concordance index
c_index_deepsurv, c_index_cox = compute_c_index(y_test, hazards, cph, df_test)
c_index_basic = concordance_index_censored(
    np.array(y_test[0], dtype=bool), y_test[1], [h[0] for h in basic_hazards])[0]

# Compute Brier scores
brier_ds, brier_cox, brier_basic = compute_brier_scores(
    surv, Y, y_test, cph_surv, basic_surv, times)
integrated_brier_ds = np.trapz(brier_ds, times)/times[-1]
integrated_brier_cox = np.trapz(brier_cox[1], brier_cox[0])/brier_cox[0][-1]
integrated_brier_basic = np.trapz(brier_basic, times)/times[-1]

# Create a DataFrame to hold the results
results = pd.DataFrame({
    'Model': ['Cox Model', 'DeepSurv', 'DeepSurv with Projection'],
    'C-index': [c_index_cox,  c_index_basic, c_index_deepsurv],
    'Integrated Brier Score': [integrated_brier_cox, integrated_brier_basic, integrated_brier_ds]
})

# Print the results in a clean format
print(results.to_string(index=False))

plot_brier_scores(times, brier_ds, brier_cox, brier_basic)

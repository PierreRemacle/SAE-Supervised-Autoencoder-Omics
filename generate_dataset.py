import dash_bootstrap_components as dbc
from dash import html
from dash import dcc
import numpy as np
import pandas as pd
import plotly.express as px
import dash
from dash import dcc, html
from dash.dependencies import Input, Output

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
from plotly import graph_objs as go

import functions_.functions_torch_regression_V3 as ft
import functions_.functions_network_pytorch as fnp
import functions_.functions_DeepSurv as fds

from sklearn.model_selection import train_test_split


def deepSurv_basic(X, X_fold, X_test, Y, Y_fold, y_test, times):
    """Trains a basic DeepSurv model and returns survival predictions."""
    feature_len = X.shape[1]
    net = ft.buildNet(feature_len, 'relu', 'DeepSurv', DEVICE, N_HIDDEN, True)
    model = fds.Custom_CoxPH(net, tt.optim.Adam)
    X, X_test = map(lambda x: torch.tensor(
        x, dtype=torch.float32), [X, X_test])
    Y_fold = (torch.tensor(Y_fold[:, 1], dtype=torch.float32), torch.tensor(
        Y_fold[:, 0], dtype=torch.float32))
    y_test = (torch.tensor(y_test[:, 1], dtype=torch.float32), torch.tensor(
        y_test[:, 0], dtype=torch.float32))

    durations, events = Y[:, 1].astype(np.float32), Y[:, 0].astype(np.float32)
    Y = (torch.tensor(durations, dtype=torch.float32),
         torch.tensor(events, dtype=torch.float32))
    lr_finder = model.lr_finder(X, Y, batch_size=30, tolerance=100)
    model.optimizer.set_lr(lr_finder.get_best_lr())

    model.fit(X, Y, batch_size=10, epochs=500, callbacks=[
              tt.callbacks.EarlyStopping(patience=10)], val_data=(X_fold, Y_fold))
    # model.compute_baseline_hazards()

    return None, model.predict(X_test)


# Set seed for reproducibility
np.random.seed(42)

# Parameters
n_samples_train = 62000  # Number of samples
n_samples_val = 3100  # Number of samples
n_samples_test = 31000  # Number of samples
n_samples = n_samples_train + n_samples_val + n_samples_test
prop_train_test_val = n_samples_train / \
    (n_samples_val + n_samples_test + n_samples_train)
prop_test_val = n_samples_val / (n_samples_test + n_samples_val)
n_features = 120   # Total number of features
n_relevant = 12   # Number of useful features
lambda_max = 5.0   # Maximum lambda for Gaussian components
lambda_0 = -0.3    # Baseline hazard

# Generate feature matrix: each value between -1 and 1
X = np.random.uniform(-1, 1, size=(n_samples, n_features))

# Select 12 relevant features (first 12 for simplicity)
X_relevant = X[:, :n_relevant]
print(X_relevant)

# Define risk components
h_x_1 = np.log(lambda_max) * \
    np.exp(-(X_relevant[:, 0] ** 2 + X_relevant[:, 1] ** 2) / 2)
h_x_2 = 0.4 * X_relevant[:, 2] * X_relevant[:, 3]
h_x_3 = 0.5 * X_relevant[:, 4] - 0.3 * X_relevant[:, 5]**2
h_x_4 = 0.5 * np.sin((X_relevant[:, 6] + X_relevant[:, 7]) * np.pi)
h_x_5 = np.tanh(X_relevant[:, 8] ** 2 + X_relevant[:, 9] ** 2)
h_x_6 = np.where(np.log(lambda_max) *
                 np.exp(-(X_relevant[:, 10] ** 2 + X_relevant[:, 11] ** 2) / 2) > 1.2, np.sin(X_relevant[:, 10]), np.cos(X_relevant[:, 11]))
# h_x_2 = np.zeros(n_samples)
# h_x_3 = np.zeros(n_samples)
# h_x_4 = np.zeros(n_samples)
# h_x_5 = np.zeros(n_samples)
# h_x_6 = np.zeros(n_samples)

# Combine all components
h_xs = [h_x_1, h_x_2, h_x_3, h_x_4, h_x_5, h_x_6]
h_x = h_x_1 + h_x_2 + h_x_3 + h_x_4 + h_x_5 + h_x_6

# Create DataFrame
feature_names = [f"X{i+1}" for i in range(n_relevant)]
df = pd.DataFrame(X, columns=feature_names +
                  [f"X_nul{i+1}" for i in range(n_relevant, n_features)])

# Get global color scale range

T = 10 * np.exp(5) / np.exp(h_x)

Time_where_80_percent_of_data = np.percentile(T, 90)
E = np.where(T > Time_where_80_percent_of_data, 0, 1)

T_censored = T.copy()
# t = t_80 where 80 percent of the data is censored if t > t_80
T_censored[T > Time_where_80_percent_of_data] = Time_where_80_percent_of_data
df["T"] = T_censored
df["E"] = E


def save_df(df):

    # save the dataframe
    labels = ["T"]
    patients = np.array(range(1, n_samples + 1))
    big_column = "Patient_number;"

    for patient in patients:
        big_column += str(patient) + ";"
    print(big_column)
    big_column = big_column[:-1]

    for col in df.columns:
        datas = df[col].to_numpy()
        string1 = col + ";"
        for data in datas:
            string1 += str(data) + ";"
        string1 = string1[:-1]

    i = 1
    for label in labels:

        filename = label.replace(" ", "") + ".csv"
        label_string = "Label-Time-surv" + ";"
        print(label)
        print(df)
        for val in df[label].values:
            label_string += str(val) + ";"
        label_string = label_string[:-1]

        with open(filename, "w") as f:
            f.write(big_column + "\n")
            f.write(label_string + "\n")
            for col in df.columns:
                if col not in labels:
                    datas = df[col].to_numpy()
                    string1 = col + ";"
                    for data in datas:
                        string1 += str(data) + ";"
                    string1 = string1[:-1]
                    f.write(string1 + "\n")
            f.close()

            print("Done")

        i = i+1

    for i in range(len(labels)):
        # put name into camel case
        label = labels[i].replace(" ", "")
        filename = label + ".csv"
        df = pd.read_csv(filename, index_col=False, header=0,
                         delimiter=";", decimal=",")
        print(df)
        # save the dataframe as a csv file
        df.to_csv(filename, index=False, header=True, sep=';')


# save_df(df)


df["h_x"] = h_x
df["h_x_1"], df["h_x_2"], df["h_x_3"], df["h_x_4"], df["h_x_5"], df["h_x_6"] = h_x_1, h_x_2, h_x_3, h_x_4, h_x_5, h_x_6

# describe the data
print(df["h_x"].describe())
print(df["h_x_1"].describe())
print(df["h_x_2"].describe())
print(df["h_x_3"].describe())
print(df["h_x_4"].describe())
print(df["h_x_5"].describe())
print(df["h_x_6"].describe())

color_min, color_max = df["h_x"].min(), df["h_x"].max()
# deep surv
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
N_HIDDEN = [100, 100, 100, 100, 100]

# Split data into training and testing sets (80% training, 20% testing)
T_here = T.reshape(-1, 1)
E = E.reshape(-1, 1)
ET = np.concatenate((E, T_here), axis=1)

# X_train, X_test, Y_train, Y_test = train_test_split(
#     X_relevant, ET, test_size=prop_train_test_val, random_state=42)
X_train, X_test, Y_train, Y_test = train_test_split(
    X, ET, test_size=prop_train_test_val, random_state=42)


# Split training data into training and validation sets (80% training, 20% validation)
X_train, X_val, Y_train, Y_val = train_test_split(
    X_train, Y_train, test_size=prop_test_val, random_state=42)

print(X_train.shape, X_val.shape, X_test.shape)

# cast everything to float32 for pytorch
X_train = X_train.astype(np.float32)
X_val = X_val.astype(np.float32)
X_test = X_test.astype(np.float32)
Y_train = Y_train.astype(np.float32)
Y_val = Y_val.astype(np.float32)
Y_test = Y_test.astype(np.float32)
T_here = T_here.astype(np.float32)

# Initialize DeepSurv model
pred, h_hat = deepSurv_basic(torch.tensor(X_train), torch.tensor(
    X_val), torch.tensor(X_test), Y_train, Y_val, Y_test, T)


c_index_basic = concordance_index_censored(
    np.array(Y_test[0], dtype=bool), Y_test[1], [h[0] for h in pred])[0]
print(c_index_basic)

X_relevant = X_test[:, :n_relevant]

# Define risk components
h_x_1 = np.log(lambda_max) * \
    np.exp(-(X_relevant[:, 0] ** 2 + X_relevant[:, 1] ** 2) / 2)
h_x_2 = 0.4 * X_relevant[:, 2] * X_relevant[:, 3]
h_x_3 = 0.5 * X_relevant[:, 4] - 0.3 * X_relevant[:, 5]**2
h_x_4 = 0.5 * np.sin((X_relevant[:, 6] + X_relevant[:, 7]) * np.pi)
h_x_5 = np.tanh(X_relevant[:, 8] ** 2 + X_relevant[:, 9] ** 2)
h_x_6 = np.where(np.log(lambda_max) *
                 np.exp(-(X_relevant[:, 10] ** 2 + X_relevant[:, 11] ** 2) / 2) > 1.2, np.sin(X_relevant[:, 10]), np.cos(X_relevant[:, 11]))
# h_x_2 = np.zeros(len(X_relevant))
# h_x_3 = np.zeros(len(X_relevant))
# h_x_4 = np.zeros(len(X_relevant))
# h_x_5 = np.zeros(len(X_relevant))
# h_x_6 = np.zeros(len(X_relevant))

real_h_x = h_x_1 + h_x_2 + h_x_3 + h_x_4 + h_x_5 + h_x_6
mean_real = np.mean(real_h_x)
mean_pred = np.mean(h_hat.cpu().detach().numpy())
std_real = np.std(real_h_x)
std_pred = np.std(h_hat.cpu().detach().numpy())

# Adjust predictions to match scale of real h(x)
h_hat_calibrated = (h_hat - mean_pred) * (std_real / std_pred) + mean_real
# h_hat_calibrated = h_hat - mean_pred + mean_real

feature_names = [f"X{i+1}" for i in range(n_relevant)]
df_deep_surv = pd.DataFrame(X_test, columns=feature_names +
                            [f"X_nul{i+1}" for i in range(n_relevant, n_features)])
# df_deep_surv = pd.DataFrame(X_test, columns=feature_names)
df_deep_surv["h_x"] = h_hat_calibrated.cpu().detach().numpy()


T_surv = 10 * np.exp(5) / np.exp(real_h_x)

df_deep_surv["h_x_1"] = df_deep_surv["h_x"] - \
    h_x_2 - h_x_3 - h_x_4 - h_x_5 - h_x_6
df_deep_surv["h_x_2"] = df_deep_surv["h_x"] - \
    h_x_1 - h_x_3 - h_x_4 - h_x_5 - h_x_6
df_deep_surv["h_x_3"] = df_deep_surv["h_x"] - \
    h_x_1 - h_x_2 - h_x_4 - h_x_5 - h_x_6
df_deep_surv["h_x_4"] = df_deep_surv["h_x"] - \
    h_x_1 - h_x_2 - h_x_3 - h_x_5 - h_x_6
df_deep_surv["h_x_5"] = df_deep_surv["h_x"] - \
    h_x_1 - h_x_2 - h_x_3 - h_x_4 - h_x_6
df_deep_surv["h_x_6"] = df_deep_surv["h_x"] - \
    h_x_1 - h_x_2 - h_x_3 - h_x_4 - h_x_5
real_h_x = h_x_1 + h_x_2 + h_x_3 + h_x_4 + h_x_5 + h_x_6
df_deep_surv["real_h_x"] = real_h_x
df_deep_surv["diff"] = real_h_x - df_deep_surv["h_x"]

# describe the data
print(df_deep_surv["h_x"].describe())
print(df_deep_surv["h_x_1"].describe())
print(df_deep_surv["h_x_2"].describe())
print(df_deep_surv["h_x_3"].describe())
print(df_deep_surv["h_x_4"].describe())
print(df_deep_surv["h_x_5"].describe())
print(df_deep_surv["h_x_6"].describe())

print(df_deep_surv["diff"].describe())

# compute brier score of the test set


def compute_brier_scores(surv, Y, y_test, times):
    """Computes the Brier scores for DeepSurv, DeepSurv Basic, and Cox models."""

    # Print types and shapes for debugging

    print(surv)
    # Convert to structured survival data
    Y_structured = Surv.from_arrays(Y[1], Y[0])
    y_test_structured = Surv.from_arrays(y_test[0], y_test[1])

    # Create a complete list of durations (including the ones in `times`)
    all_durations = sorted(set(surv.index).union(set(times)))

    # Reindex the dataframes to include all durations, adding rows with NaN for missing durations
    surv_reindexed = surv.reindex(all_durations)

    # Fill missing values with forward fill (ffill)
    surv_filled = surv_reindexed.ffill(axis=0)

    # Fill remaining NaN values with backward fill (bfill)
    surv_filled = surv_filled.bfill(axis=0)

    # Select only rows that correspond to the times in the provided list
    surv_filtered = surv_filled.loc[surv_filled.index.isin(times)]

    # Compute the Brier scores for each model
    brier_ds = brier_score(
        Y_structured, y_test_structured, surv_filtered.T, times)

    return brier_ds


# surv
# times = np.linspace(int(min(durations)), int(max(durations)),
#                     num=int(max(durations)) - int(min(durations)) + 1)
# brirer_score = compute_brier_scores(mo, Y_train, y_test, times)


# Define feature combinations
comb_dict = {
    "Gaussian (x_1 , x_2)": ("h_x_1", "X1", "X2"),
    "Interaction (x_3, x_4)": ("h_x_2", "X3", "X4"),
    "Linear (x_5 , x_6)": ("h_x_3", "X5", "X6"),
    "Sinusoidal (x_7 , x_8)": ("h_x_4", "X7", "X8"),
    "Tanh (x_9 , x_10)": ("h_x_5", "X9", "X10"),
    "Conditional (x_10 , x_11)": ("h_x_6", "X11", "X12")
}

# Initialize Dash app
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Interactive Risk Score Visualization"),

    html.Label("Select Combination:"),
    dcc.Dropdown(
        id='comb',
        options=[{'label': key, 'value': key} for key in comb_dict.keys()],
        value="Gaussian (x_1 , x_2)"
    ),
    html.Button('h(x)', id='submit-val', n_clicks=0),
    # First row: Three square graphs
    # Row for the three square graphs
    html.Div(children=[
        dcc.Graph(id='scatter-plot-1',
                  style={'width': '30vw', 'height': '30vw', 'display': 'inline-block'}),
        dcc.Graph(id='scatter-plot-2',
                  style={'width': '30vw', 'height': '30vw', 'display': 'inline-block'}),
        dcc.Graph(id='scatter-plot-3',
                  style={'width': '30vw', 'height': '30vw', 'display': 'inline-block'}),
    ], style={'text-align': 'center'}),

    # Second row: One full-width graph
    html.Div([
        dcc.Graph(id='time-plot')
    ], className="row"),

    html.Div([
        dcc.Graph(id='h(x) vs approximations')
    ], className="row")
])


@ app.callback(
    Output('scatter-plot-1', 'figure'),
    Output('time-plot', 'figure'),
    Output('scatter-plot-2', 'figure'),
    Output('scatter-plot-3', 'figure'),
    Output('h(x) vs approximations', 'figure'),
    [Input('comb', 'value'), Input('submit-val', 'n_clicks')]
)
def update_plot(selected_comb, n_clicks):
    h_x_col, x_col, y_col = comb_dict[selected_comb]

    # Determine the global min and max of the 'Risk' variable (for consistent color range)
    min_risk = df[h_x_col].min()
    max_risk = df[h_x_col].max()

    if n_clicks % 2 == 0:
        fig = px.scatter(df, x=x_col, y=y_col, color=h_x_col,
                         color_continuous_scale="viridis", range_color=[min_risk, max_risk],
                         title=f"Risk Score: {selected_comb}",
                         labels={x_col: x_col, y_col: y_col, h_x_col: "Risk"})
        # fmt:  off
        fig3 = px.scatter(df_deep_surv, x=x_col, y=y_col, color=h_x_col,
                          color_continuous_scale="viridis", range_color=[min_risk, max_risk],
                          title=f"Estimated Risk Score: {selected_comb}",
                          labels={x_col: x_col, y_col: y_col, h_x_col: "Risk"})
        # fmt:  on

    else:
        fig = px.scatter(df, x=x_col, y=y_col, color=df["h_x"],
                         color_continuous_scale="viridis", range_color=[min_risk, max_risk],
                         title=f"Risk Score: {selected_comb}",
                         labels={x_col: x_col, y_col: y_col, h_x_col: "Risk"})
        # no refactor
        # fmt: off 
        fig3 = px.scatter(df_deep_surv, x=x_col, y=y_col, color=df_deep_surv["h_x"],
                          color_continuous_scale="viridis", range_color=[min_risk, max_risk],
                          title=f"Estimated Risk Score: {selected_comb}",
                          labels={x_col: x_col, y_col: y_col, h_x_col: "Risk"})
        # fmt: on

    fig4 = px.scatter(df_deep_surv, x=x_col, y=y_col, color=df_deep_surv["diff"],
                      color_continuous_scale="viridis",
                      title=f"Difference: {selected_comb}",
                      labels={x_col: x_col, y_col: y_col, h_x_col: "Risk"})
 # Fix color range to ensure the color bar stays the same size
    fig.update_layout(
        coloraxis=dict(colorbar=dict(len=0.6, thickness=15, x=1.05)),
        # Adjust right margin to make space
        autosize=False,
        margin=dict(l=50, r=100, t=50, b=50),
        xaxis=dict(scaleanchor='y'),  # Keep axes square
    )
    fig3.update_layout(
        coloraxis=dict(colorbar=dict(len=0.6, thickness=15, x=1.05)),
        # Adjust right margin to make space
        #
        autosize=False,
        margin=dict(l=50, r=100, t=50, b=50),
        xaxis=dict(scaleanchor='y'),  # Keep axes square
    )

    fig4.update_layout(
        coloraxis=dict(colorbar=dict(len=0.6, thickness=15, x=1.05)),
        # Adjust right margin to make space
        #
        autosize=False,
        margin=dict(l=50, r=100, t=50, b=50),
        xaxis=dict(scaleanchor='y'),  # Keep axes square
    )
    # time sorted plot for T
# Create an empty figure
    fig2 = go.Figure()

# Add the first scatter plot
    fig2.add_scatter(x=T, y=df["h_x"], mode='markers', name='h(x)')

# Add the second scatter plot (DeepSurv)
    fig2.add_scatter(x=T_surv, y=df_deep_surv["h_x"], mode='markers',
                     name='DeepSurv', marker=dict(color='red'))

# Add the third scatter plot (Difference: h(x) - DeepSurv)
#     fig2.add_scatter(x=T_surv, y=df["h_x"] - df_deep_surv["h_x"], mode='markers',
#                      name='DeepSurv - h(x)', marker=dict(color='blue'))
#
# # Add the fourth scatter plot (Ratio: h(x) / DeepSurv)
#     fig2.add_scatter(x=T_surv, y=df["h_x"] / df_deep_surv["h_x"], mode='markers',
#                      name='DeepSurv / h(x)', marker=dict(color='green'))

# Update layout to add titles and labels
    fig2.update_layout(
        title="Risk Score vs Time",
        xaxis_title="Time",
        yaxis_title="Risk Score"
    )
    # find where 90 percent of the data is and plot a line
    fig2.add_shape(type="line", x0=Time_where_80_percent_of_data, y0=-1, x1=Time_where_80_percent_of_data, y1=4,
                   line=dict(color="Red", width=2))
    # add a text to the line
    fig2.add_annotation(
        x=Time_where_80_percent_of_data, y=3, text=str(Time_where_80_percent_of_data), showarrow=True, arrowhead=1)
    fig5 = go.Figure()
    fig5.add_scatter(y=df_deep_surv["h_x"],
                     x=df_deep_surv["real_h_x"], mode='markers', name='h(x)', marker=dict(color='blue', size=1))
# Add identity line
    min_val = min(real_h_x.min(), df_deep_surv["h_x"].min())
    max_val = max(real_h_x.max(), df_deep_surv["h_x"].max())
    fig5.add_scatter(x=[min_val, max_val], y=[min_val, max_val],
                     mode='lines', name='Perfect prediction', line=dict(dash='dash'))
    return fig, fig2, fig3, fig4, fig5


# if __name__ == '__main__':
    # app.run_server(debug=True, use_reloader=False)

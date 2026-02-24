import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

import functions_.functions_torch_regression_V3 as ft
import functions_.functions_network_pytorch as fnp
import functions_.functions_DeepSurv as fds

import functions_.functions_network_pytorch as fnp
import functions_.functions_DeepSurv as fds  # DeepSurv

import torchtuples as tt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import StandardScaler, scale as scale
np.random.seed(1234)
_ = torch.manual_seed(123)

repository = "results_stat"

# Load data

# unit scaling of the input data
doScale = False
# log transform of the input data
doLog = False
# row normalization of the input data
doRowNorm = False
# test size to 0
test_size = None

n_hidden = 300  # amount of neurons on netbio's hidden layer

file_name = './TimeInterval.csv'

X, X_test, Y, y_test, feature_names, label_name_train, label_name_test, patient_name, gaussianKDE, gaussianKDETest, divided = ft.ReadDataCV_surv(
    file_name, test_size=test_size, doScale=doScale, doLog=doLog,  doRowNorm=doRowNorm
)

feature_len = len(feature_names)

# Définir le device
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Charger le dictionnaire de poids
state_dict = torch.load(repository + '/best_net')
print("State Dict Keys:")
print(state_dict.keys())  # Vérifier les clés du state_dict

# Initialiser le modèle
net = fds.MLP(feature_len, n_hidden, 1, batch_norm=True,
              dropout=0.1, output_bias=False).to(DEVICE)

# Affichage des couches du modèle
print("Model Layers:")
for name, param in net.named_parameters():
    print(name)

# Charger les poids
net.load_state_dict(state_dict)
net.eval()

# Créer le modèle Cox
model = fds.Custom_CoxPH(net, tt.optim.Adam)

# Vérification des données Y
durations = Y[1]  # La première colonne doit contenir les temps de survie
events = Y[0]  # La deuxième colonne doit contenir les événements (0 ou 1)
Y = (durations, events)

# Affichage de la structure de Y
print(f"Shape of Y (durations, events): {Y[0].shape}, {Y[1].shape}")

# Vérification de la structure de X_test
x_test = np.array(X_test[16], dtype=np.float32).reshape(
    1, -1)  # Assurer une forme correcte

print(f"Shape of X_test: {X_test[1].shape}")
print(f"Shape of x_test after reshape: {x_test.shape}")

df_test = pd.DataFrame(x_test, columns=feature_names)
# fuze X and Y
original = np.concatenate([np.array(X), np.array(Y).T], axis=1)
features_with_event_names = list(feature_names) + ["Duration", "Event"]
df_train = pd.DataFrame(original, columns=features_with_event_names)
# get description for durations and events
print("All data")
print(df_train[["Smoking", "Duration", "Event", "Age"]
               ].groupby("Smoking").describe())
df_filtered = df_train[(df_train["Age"] >= 30) & (df_train["Age"] <= 40)]
print("30-40")
print(df_filtered[["Smoking", "Duration", "Event"]
                  ].groupby("Smoking").describe())
df_filtered = df_train[(df_train["Age"] >= 40) & (df_train["Age"] <= 50)]

print("40-50")
print(df_filtered[["Smoking", "Duration", "Event"]
                  ].groupby("Smoking").describe())
df_filtered = df_train[(df_train["Age"] >= 50) & (df_train["Age"] <= 60)]

print("50-60")
print(df_filtered[["Smoking", "Duration", "Event"]
                  ].groupby("Smoking").describe())

df_filtered = df_train[(df_train["Age"] >= 60)]
print("60-")
print(df_filtered[["Smoking", "Duration", "Event"]
                  ].groupby("Smoking").describe())
# copy 4 time the first line of the test data
df_test = pd.concat([df_test]*3, ignore_index=True)


df_test.loc[0, 'Smoking'] = 0
df_test.loc[0, 'Alcoholism'] = 0
df_test.loc[0, 'Age'] = 50
df_test.loc[0, 'Gender_D'] = 0


df_test.loc[1, 'Smoking'] = 0
df_test.loc[1, 'Alcoholism'] = 0
df_test.loc[1, 'Age'] = 50
df_test.loc[1, 'Gender_D'] = 1


df_test.loc[2, 'Smoking'] = 0
df_test.loc[2, 'Alcoholism'] = 1
df_test.loc[2, 'Age'] = 50
df_test.loc[2, 'Gender_D'] = 1


# df_test.loc[3, 'Age'] = 40
# df_test.loc[3, 'Smoking'] = 0
# df_test.loc[3, 'Alcoholism'] = 0
# df_test.loc[3, 'Gender_D'] = 0

# Affichage des premières colonnes
print(df_test[['Smoking', 'Gender_D', 'Alcoholism', 'Age']])


print(df_test)
x_test = df_test.to_numpy()


# Vérification des types de données
print(f"Data type of X_test: {x_test.dtype}")
print(f"Data type of Y: {Y[0].dtype}, {Y[1].dtype}")

# Assurer que X et Y sont au bon type
X = X.astype(np.float32)
Y = (Y[0].astype(np.float32), Y[1].astype(np.float32))

# Calcul des baseline hazards

# Prédiction de la fonction de survie

X_train = X
Y_train = Y
print(type(Y_train))
print(type(X_train))
print(x_test)
# apply StandardScaler to X
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
x_test = scaler.transform(x_test)
X_train = X_train - np.mean(X_train, axis=0)
X_test = X_test - np.mean(X_test, axis=0)
x_test = x_test - np.mean(x_test, axis=0)
X_train = scale(X_train, axis=0)  # Standardization along rows
X_test = scale(X_test, axis=0)  # Standardization along rows
x_test = scale(x_test, axis=0)  # Standardization along rows


X_train = X_train - np.mean(X_train, axis=1, keepdims=True)
X_test = X_test - np.mean(X_test, axis=1, keepdims=True)
x_test = x_test - np.mean(x_test, axis=1, keepdims=True)
_ = model.compute_baseline_hazards(X_train, Y_train)
surv = model.predict_surv_df(x_test)

surv.plot()
plt.ylabel('S(t | x)')

_ = plt.xlabel('Time')
plt.title('Survival function')
# put the legend for each curve
X_test = df_test.to_dict(orient='records')
# plt.legend([str(X_test[0]["Age"])+" " + str(X_test[0]["Gender_D"]) + " "+str(X_test[0]["Smoking"])+" "+str(X_test[0]["Alcoholism"]),
#             str(X_test[1]["Age"])+" " + str(X_test[1]["Gender_D"]) + " " +
#             str(X_test[1]["Smoking"])+" "+str(X_test[1]["Alcoholism"]),
#             str(X_test[2]["Age"])+" " + str(X_test[2]["Gender_D"]) + " " +
#             str(X_test[2]["Smoking"])+" "+str(X_test[2]["Alcoholism"]),
#             str(X_test[3]["Age"])+" " + str(X_test[3]["Gender_D"]) + " "+str(X_test[3]["Smoking"])+" "+str(X_test[3]["Alcoholism"])])
# plt.legend([
#     str(X_test[i]["Age"]) + " " + str(X_test[i]["Gender_D"]) + " " +
#     str(X_test[i]["Smoking"]) + " " + str(X_test[i]["Alcoholism"])
#     for i in range(len(X_test))
# ])
# plt.legend([
#     str(X_test[i]["Age"])
#     for i in range(len(X_test))
# ])
plt.legend([
    str(X_test[i]["Age"]) + " " + str(X_test[i]["Gender_D"]) +
    " " + str(X_test[i]["Alcoholism"])
    for i in range(len(X_test))
])
plt.show()

import torch
import torch.nn as nn
import shap
import sklearn.datasets
import sklearn.preprocessing
import sklearn.preprocessing
import os
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.inspection import permutation_importance
import torch.nn.functional as F
from skorch import NeuralNetRegressor
from skorch.callbacks import EpochScoring


# 定义神经网络模型
class NeuralNetworkSkorch(NeuralNetRegressor):
    def get_loss(self, y_pred, y_true, X=None, training=False):
        return F.mse_loss(y_pred, y_true)

# 定义 R2 callback
def r2_score_callback(net, dataset, y):
    y_pred = net.predict(dataset)
    return r2_score(y, y_pred)

# Define a simple neural network using PyTorch
class NeuralNetwork(nn.Module):
    def __init__(self, input_size):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Load California Housing data
california_data = sklearn.datasets.fetch_california_housing()
X_california = california_data["data"]
y_california = california_data["target"]

# Split data into train and test sets
X_train_california, X_test_california, y_train_california, y_test_california = sklearn.model_selection.train_test_split(
    X_california, y_california, test_size=0.2, random_state=111
)

# Standardize data
scaler_california = sklearn.preprocessing.StandardScaler()
X_train_scaled_california = scaler_california.fit_transform(X_train_california)
X_test_scaled_california = scaler_california.transform(X_test_california)

# Convert data to PyTorch tensors
X_train_tensor_california = torch.FloatTensor(X_train_scaled_california)
X_test_tensor_california = torch.FloatTensor(X_test_scaled_california)
y_train_tensor_california = torch.FloatTensor(y_train_california).view(-1, 1)

# Initialize and train the neural network
input_size_california = X_train_tensor_california.shape[1]

epochs = [500]
# epochs = [1, 10, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
R2 = []
for epoch in epochs:
    model_california = NeuralNetwork(input_size_california)
    criterion_california = nn.MSELoss()
    optimizer_california = torch.optim.Adam(model_california.parameters(), lr=0.001)
    for i in range(epoch):
        optimizer_california.zero_grad()
        outputs_california = model_california(X_train_tensor_california)
        loss_california = criterion_california(outputs_california, y_train_tensor_california)
        loss_california.backward()
        optimizer_california.step()

    pred_test_california = model_california(X_test_tensor_california).detach().numpy()

    # Calculate R2 score
    score_r2 = r2_score(y_test_california, pred_test_california)
    R2.append(round(score_r2,4))
    background_california = shap.utils.sample(X_train_tensor_california, 10)

    explainer_california = shap.DeepExplainer(model_california, background_california)
    shap_values_california = explainer_california.shap_values(X_test_tensor_california)

    #print("SHAP values:", shap_values_california)
    X_test_numpy = X_test_tensor_california.numpy()
    plt.figure(figsize=(6, 3.5))
    summary_plot = shap.summary_plot(shap_values_california, X_test_numpy,
                                 feature_names=california_data.feature_names,
                                 plot_type="violin", show=False)
    
    plt.savefig(f'NN_{epoch}_california_housing_shap_importance.svg')
    plt.show()


    pred_test_california = model_california(X_test_tensor_california).detach().numpy()
    plt.figure(figsize=(6, 3.5))
    plt.scatter(pred_test_california, y_test_california, alpha=0.2)
    plt.plot([0, 5], [0, 5], "--", color="#666666")
    plt.title(f"NN(epoch = {epoch},R2 = %.4f)" % score_r2)  # Ensure you have calculated score_r2
    plt.xlabel("Predicted MedianHouseVal")
    plt.ylabel("True MedianHouseVal")
    plt.grid()
    plt.tight_layout()
    plt.savefig(f'NN_{epoch}_figure_california_housing_regression.svg')
    plt.show()

    
    
    model_california_skorch = NeuralNetworkSkorch(
        NeuralNetwork,
        module__input_size=input_size_california,
        max_epochs=epoch,
        lr=0.001,
        optimizer=torch.optim.Adam,
        criterion=torch.nn.MSELoss,
        iterator_train__shuffle=True,
        callbacks=[
            ('r2', EpochScoring(r2_score_callback, name='r2', lower_is_better=False)),
        ],
        verbose=0
    )
    X_train_numpy = X_train_tensor_california.numpy()
    y_train_numpy = y_train_tensor_california.numpy()
    model_california_skorch.fit(X_train_numpy, y_train_numpy)
    result = permutation_importance(model_california_skorch, X_test_tensor_california.numpy(), y_test_california, n_repeats=30, random_state=42)
    perm_importance = result.importances_mean
    perm_std = result.importances_std
    sorted_idx = np.argsort(perm_importance)
    # 绘制图表
    fig, ax = plt.subplots()
    ax.boxplot(result.importances[sorted_idx].T, vert=False)
    ax.set_yticklabels(np.array(california_data.feature_names)[sorted_idx])
    plt.grid()
    plt.xlabel('Permutation feature importances (impact on model output)')
    plt.savefig(f'NN_{epoch}_figure_california_housing_permutation_importance.svg')
    plt.show()
    

    plt.show()
print(R2)
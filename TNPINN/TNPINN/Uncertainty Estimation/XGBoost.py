import os
from math import sqrt
import numpy as np
import pandas as pd
import pywt
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from matplotlib import pyplot as plt

# 参数设置
window_size = 30
forecast_steps = 10
input_size = 6 * window_size
hidden_size = 64
output_size = forecast_steps
physics_weight = 0.1
epochs = 1000
learningrate = 0.005
dropout_rate = 0.2
mc_samples = 100

# XGBoost 参数
xgb_params = {
    'n_estimators': 500,
    'max_depth': 6,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42
}

# 设置随机种子
def setup_seed(seed):
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

# 小波去噪
def wavelet_denoising(data, wavelet='db4', level=4, threshold=0.1):
    coeffs = pywt.wavedec(data, wavelet, level=level)
    coeffs = [pywt.threshold(c, threshold, mode='soft') for c in coeffs]
    return pywt.waverec(coeffs, wavelet)

# 归一化函数
def min_max_normalization_per_feature(data):
    if isinstance(data, np.ndarray):
        data = torch.tensor(data, dtype=torch.float32)
    min_vals = torch.min(data, dim=0)[0]
    max_vals = torch.max(data, dim=0)[0]
    return (data - min_vals) / (max_vals - min_vals + 1e-6)

# 生成时序序列
def build_sequences(text, window_size, forecast_steps):
    x, y = [], []
    for i in range(len(text) - window_size - forecast_steps):
        sequence = text[i:i + window_size, :]
        target = text[i + window_size:i + window_size + forecast_steps, -1]
        x.append(sequence)
        y.append(target)
    return np.array(x), np.array(y, dtype=np.float32)

# 评估函数
def evaluation(y_test, y_predict):
    mae = mean_absolute_error(y_test, y_predict)
    mse = mean_squared_error(y_test, y_predict)
    rmse = sqrt(mse)
    return mae, mse, rmse

# 新增：计算 NLL
def compute_nll(y_true, mean_pred, std_pred):
    y_true = y_true.flatten()
    mean_pred = mean_pred.flatten()
    std_pred = std_pred.flatten() + 1e-6  # 避免除零
    nll = 0.5 * np.log(2 * np.pi) + np.log(std_pred) + 0.5 * ((y_true - mean_pred) ** 2) / (std_pred ** 2)
    return np.mean(nll)

# 新增：计算 Calibration Error
def compute_calibration_error(y_true, mean_pred, std_pred, confidence_levels=[0.5, 0.68, 0.95]):
    y_true = y_true.flatten()
    mean_pred = mean_pred.flatten()
    std_pred = std_pred.flatten() + 1e-6
    errors = []
    for cl in confidence_levels:
        z = np.abs(np.percentile(np.random.normal(0, 1), (1 + cl) * 50))  # z-score for confidence level
        lower = mean_pred - z * std_pred
        upper = mean_pred + z * std_pred
        coverage = np.mean((y_true >= lower) & (y_true <= upper))
        errors.append((coverage - cl) ** 2)
    return np.sqrt(np.mean(errors))

# 新增：计算 MPIW
def compute_mpiw(ci_lower, ci_upper):
    ci_lower = ci_lower.flatten()
    ci_upper = ci_upper.flatten()
    return np.mean(ci_upper - ci_lower)

# 物理信息神经网络 (PINN)
class PINN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_rate=dropout_rate):
        super(PINN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, output_size)
        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, mc_dropout=False):
        x = self.activation(self.fc1(x))
        if mc_dropout:
            x = self.dropout(x)
        x = self.activation(self.fc2(x))
        if mc_dropout:
            x = self.dropout(x)
        x = self.activation(self.fc3(x))
        if mc_dropout:
            x = self.dropout(x)
        x = self.fc4(x)
        return x

    def physics_loss(self, x, y_pred):
        physics_loss = 0.0
        if y_pred.size(1) > 1:
            diff = y_pred[:, 1:] - y_pred[:, :-1]
            monotonic_loss = torch.relu(diff).mean()
            physics_loss += monotonic_loss
        if x.requires_grad:
            grad_outputs = torch.ones_like(y_pred)
            gradients = torch.autograd.grad(
                outputs=y_pred,
                inputs=x,
                grad_outputs=grad_outputs,
                create_graph=True,
                retain_graph=True,
                only_inputs=True
            )[0]
            capacity_grad = gradients[:, -1:] if gradients.size(1) > 0 else gradients
            current_capacity = x[:, -1:] if x.size(1) > 0 else x
            expected_decay = -0.01 * current_capacity
            decay_loss = torch.mean((capacity_grad - expected_decay) ** 2)
            physics_loss += decay_loss
        return physics_loss

# PINN+XGBoost 混合模型
class PINNXGBoostModel:
    def __init__(self, input_size, hidden_size, output_size, device):
        self.device = device
        self.pinn = PINN(input_size, hidden_size, output_size).to(device)
        self.xgb_model = xgb.XGBRegressor(**xgb_params)
        self.scaler = StandardScaler()
        self.is_fitted = False

    def train_pinn(self, train_data, train_target, epochs, lr):
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.pinn.parameters(), lr=lr)
        train_losses = []
        physics_losses = []
        for epoch in range(epochs):
            self.pinn.train()
            optimizer.zero_grad()
            pinn_output = self.pinn(train_data, mc_dropout=True)
            data_loss = criterion(pinn_output, train_target)
            physics_loss = self.pinn.physics_loss(train_data, pinn_output)
            total_loss = data_loss + physics_weight * physics_loss
            total_loss.backward()
            optimizer.step()
            train_losses.append(data_loss.item())
            physics_losses.append(physics_loss.item())
            if epoch % 100 == 0:
                print(f"Epoch {epoch}: Data Loss = {data_loss.item():.6f}, Physics Loss = {physics_loss.item():.6f}")
        return train_losses, physics_losses

    def extract_features(self, data, mc_dropout=False):
        self.pinn.eval()
        with torch.no_grad():
            x = self.pinn.activation(self.pinn.fc1(data))
            x = self.pinn.activation(self.pinn.fc2(x))
            pinn_features = self.pinn.activation(self.pinn.fc3(x))
            pinn_output = self.pinn.fc4(pinn_features)
            original_features = data.cpu().numpy()
            pinn_features_np = pinn_features.cpu().numpy()
            pinn_output_np = pinn_output.cpu().numpy()
            if len(pinn_output_np.shape) > 2:
                pinn_output_np = pinn_output_np.reshape(pinn_output_np.shape[0], -1)
            combined_features = np.concatenate([
                original_features,
                pinn_features_np,
                pinn_output_np
            ], axis=1)
        return combined_features

    def fit(self, train_data, train_target, epochs=1000, lr=0.005):
        print("Training PINN...")
        train_losses, physics_losses = self.train_pinn(train_data, train_target, epochs, lr)
        print("Extracting features for XGBoost...")
        train_features = self.extract_features(train_data)
        train_features_scaled = self.scaler.fit_transform(train_features)
        if isinstance(train_target, torch.Tensor):
            train_target_np = train_target.cpu().numpy()
        else:
            train_target_np = train_target
        if len(train_target_np.shape) > 1 and train_target_np.shape[1] > 1:
            self.xgb_models = []
            for step in range(train_target_np.shape[1]):
                print(f"Training XGBoost for step {step + 1}...")
                xgb_model = xgb.XGBRegressor(**xgb_params)
                xgb_model.fit(train_features_scaled, train_target_np[:, step])
                self.xgb_models.append(xgb_model)
        else:
            print("Training XGBoost...")
            self.xgb_model.fit(train_features_scaled, train_target_np.ravel())
        self.is_fitted = True
        return train_losses, physics_losses

    def predict(self, test_data):
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        test_features = self.extract_features(test_data)
        test_features_scaled = self.scaler.transform(test_features)
        if hasattr(self, 'xgb_models'):
            predictions = []
            for model in self.xgb_models:
                step_pred = model.predict(test_features_scaled)
                predictions.append(step_pred)
            predictions = np.column_stack(predictions)
        else:
            predictions = self.xgb_model.predict(test_features_scaled)
        return predictions

    def predict_with_uncertainty(self, test_data, mc_samples=100):
        self.pinn.train()
        predictions = []
        for _ in range(mc_samples):
            with torch.no_grad():
                pred = self.pinn(test_data, mc_dropout=True)
            predictions.append(pred.detach().cpu().numpy())
        predictions = np.stack(predictions, axis=0)
        mean_pred = np.mean(predictions, axis=0)
        std_pred = np.std(predictions, axis=0)
        ci_lower = mean_pred - 1.96 * std_pred
        ci_upper = mean_pred + 1.96 * std_pred
        return mean_pred, std_pred, ci_lower, ci_upper

# 早停机制
class EarlyStopping:
    def __init__(self, patience=100, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0

    def __call__(self, val_loss):
        if val_loss < self.best_val_loss - self.delta:
            self.best_val_loss = val_loss
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1
        if self.epochs_without_improvement >= self.patience:
            return True
        return False

# 数据加载和处理
folder_path = '../NASA Cleaned'
battery_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
battery_data = {}

for file_name in battery_files:
    file_path = os.path.join(folder_path, file_name)
    df = pd.read_csv(file_path)
    features = df.iloc[:, 1:].values
    last_column = df.iloc[:, -1].values
    last_column = np.apply_along_axis(wavelet_denoising, 0, last_column)
    features = np.column_stack((features[:, 0:5], last_column / 2))
    features = min_max_normalization_per_feature(features)
    data, target = build_sequences(features, window_size, forecast_steps)
    data_reshaped = data.reshape(data.shape[0], -1)
    battery_data[file_name] = (data_reshaped, target)

# 交叉验证
maes, rmses, nlls, calibration_errors, mpiws = [], [], [], [], []
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

for test_battery, (test_data, test_target) in battery_data.items():
    print(f"\nTesting on battery: {test_battery}")
    train_data, train_target = [], []
    for battery, (data, target) in battery_data.items():
        if battery != test_battery:
            train_data.append(data)
            train_target.append(target)
    train_data = np.concatenate(train_data)
    train_target = np.concatenate(train_target)

    train_data_tensor = torch.tensor(train_data, requires_grad=True, dtype=torch.float32).to(device)
    train_target_tensor = torch.tensor(train_target, dtype=torch.float32).to(device)
    test_data_tensor = torch.tensor(test_data, requires_grad=True, dtype=torch.float32).to(device)
    test_target_tensor = torch.tensor(test_target, dtype=torch.float32).to(device)

    setup_seed(0)
    model = PINNXGBoostModel(
        input_size=train_data.shape[1],
        hidden_size=hidden_size,
        output_size=forecast_steps,
        device=device
    )

    # 训练模型
    train_losses, physics_losses = model.fit(
        train_data_tensor,
        train_target_tensor,
        epochs=epochs,
        lr=learningrate
    )

    # 预测（XGBoost 预测）
    predictions = model.predict(test_data_tensor)

    # MC 不确定性估计（PINN 部分）
    mean_pred, std_pred, ci_lower, ci_upper = model.predict_with_uncertainty(test_data_tensor, mc_samples)

    # 评估（使用 XGBoost 预测结果）
    test_np = test_target_tensor.detach().cpu().numpy()
    if len(predictions.shape) == 1:
        predictions = predictions.reshape(-1, 1)
    if len(test_np.shape) == 1:
        test_np = test_np.reshape(-1, 1)

    mae, mse, rmse = evaluation(test_np, predictions)
    print(f"RMSE: {rmse * 100:.3f}, MAE: {mae * 100:.3f}")
    maes.append(mae * 100)
    rmses.append(rmse * 100)

    # 计算新指标（使用 PINN 的 MC 预测）
    nll = compute_nll(test_np, mean_pred, std_pred)
    calibration_error = compute_calibration_error(test_np, mean_pred, std_pred)
    mpiw = compute_mpiw(ci_lower, ci_upper)
    print(f"NLL: {nll:.3f}, Calibration Error: {calibration_error:.3f}, MPIW: {mpiw:.3f}")
    nlls.append(nll)
    calibration_errors.append(calibration_error)
    mpiws.append(mpiw)

    # 计算覆盖概率（基于 PINN 的 MC 预测）
    coverage = np.mean((test_np >= ci_lower) & (test_np <= ci_upper))
    print(f"95% Coverage Probability (PINN): {coverage * 100:.2f}%")

    # 绘制训练损失图
    plt.figure(figsize=(12, 6))
    plt.plot(train_losses, label="Data Loss")
    plt.plot(physics_losses, label="Physics Loss")
    plt.title(f"Training Losses for {test_battery}")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

    # 绘制预测和置信区间（基于 PINN 的 MC 预测，示例：第一个样本）
    plt.figure(figsize=(12, 6))
    time_steps = np.arange(forecast_steps)
    sample_idx = 1
    plt.plot(time_steps, test_np[sample_idx], label="True", marker='o')
    plt.plot(time_steps, mean_pred[sample_idx], label="Predicted (PINN)", marker='x')
    plt.fill_between(time_steps, ci_lower[sample_idx], ci_upper[sample_idx],
                     color="red", alpha=0.2, label="95% CI")
    plt.title(f"Prediction with 95% CI for {test_battery} (PINN)", fontsize=18)
    plt.xlabel("Time Step", fontsize=18)
    plt.ylabel("SOH(%)", fontsize=18)
    plt.tick_params(axis='both', labelsize=20)
    plt.legend(fontsize=18)
    plt.grid(True)
    plt.show()

    # 绘制预测与真实值的散点图（XGBoost 预测）
    test_np_flat = test_np.flatten()
    pred_np_flat = predictions.flatten()
    plt.figure(figsize=(8, 8))
    plt.scatter(test_np_flat, pred_np_flat, s=100, color='dodgerblue', alpha=0.8)
    plt.plot([min(test_np_flat), max(test_np_flat)], [min(test_np_flat), max(test_np_flat)], 'r--', label='Ideal')
    plt.xlabel("True Values")
    plt.ylabel("Predicted Values (XGBoost)")
    plt.title(f"Prediction vs True for {test_battery}")
    plt.legend()
    plt.grid(True)
    plt.show()

# 汇总交叉验证结果
print("\nCross-validation results:")
print(f"Average RMSE: {np.mean(rmses):.3f}")
print(f"Average MAE: {np.mean(maes):.3f}")
print(f"Average NLL: {np.mean(nlls):.3f}")
print(f"Average Calibration Error: {np.mean(calibration_errors):.3f}")
print(f"Average MPIW: {np.mean(mpiws):.3f}")
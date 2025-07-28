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

window_size = 10  # 设置窗口大小
forecast_steps = 10
# 模型参数
input_size = 6  # 输入特征数
hidden_size = 64  # PINN隐藏层大小
output_size = forecast_steps  # 输出层大小
physics_weight = 0.1  # 物理约束权重
epochs = 1000  # 训练轮数
learningrate = 0.005  # 学习率

# XGBoost参数
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
    np.random.seed(seed)  # Numpy module.
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


# 小波变换去噪函数（软阈值法）
def wavelet_denoising(data, wavelet='db4', level=4, threshold=0.1):
    coeffs = pywt.wavedec(data, wavelet, level=level)
    coeffs = [pywt.threshold(c, threshold, mode='soft') for c in coeffs]
    return pywt.waverec(coeffs, wavelet)


# 物理信息神经网络 (PINN)
class PINN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(PINN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, output_size)
        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.dropout(x)
        x = self.activation(self.fc2(x))
        x = self.dropout(x)
        x = self.activation(self.fc3(x))
        x = self.fc4(x)
        return x

    def physics_loss(self, x, y_pred):
        """
        实现电池容量衰减的物理约束
        电池容量应该满足：
        1. 单调递减性：容量随时间递减
        2. 指数衰减规律：dC/dt = -k*C^n (简化的容量衰减模型)
        """
        batch_size = x.size(0)
        physics_loss = 0.0

        # 约束1：单调递减性
        if y_pred.size(1) > 1:
            diff = y_pred[:, 1:] - y_pred[:, :-1]
            monotonic_loss = torch.relu(diff).mean()  # 惩罚递增部分
            physics_loss += monotonic_loss

        # 约束2：容量衰减率约束
        if x.requires_grad:
            # 计算容量对时间的梯度
            grad_outputs = torch.ones_like(y_pred)
            gradients = torch.autograd.grad(
                outputs=y_pred,
                inputs=x,
                grad_outputs=grad_outputs,
                create_graph=True,
                retain_graph=True,
                only_inputs=True
            )[0]

            # 容量衰减率应该为负且与当前容量相关
            capacity_grad = gradients[:, -1:] if gradients.size(1) > 0 else gradients
            current_capacity = x[:, -1:] if x.size(1) > 0 else x

            # 物理约束：dC/dt ≈ -k*C (k>0)
            expected_decay = -0.01 * current_capacity  # 假设衰减系数k=0.01
            decay_loss = torch.mean((capacity_grad - expected_decay) ** 2)
            physics_loss += decay_loss

        return physics_loss


# PINN+XGBoost混合模型
class PINNXGBoostModel:
    def __init__(self, input_size, hidden_size, output_size, device):
        self.device = device
        self.pinn = PINN(input_size, hidden_size, output_size).to(device)
        self.xgb_model = xgb.XGBRegressor(**xgb_params)
        self.scaler = StandardScaler()
        self.is_fitted = False

    def train_pinn(self, train_data, train_target, epochs, lr):
        """训练PINN模型"""
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.pinn.parameters(), lr=lr)

        train_losses = []
        physics_losses = []

        for epoch in range(epochs):
            self.pinn.train()
            optimizer.zero_grad()

            # 前向传播
            pinn_output = self.pinn(train_data)

            # 数据拟合损失
            data_loss = criterion(pinn_output, train_target)

            # 物理约束损失
            physics_loss = self.pinn.physics_loss(train_data, pinn_output)

            # 总损失
            total_loss = data_loss + physics_weight * physics_loss

            total_loss.backward()
            optimizer.step()

            train_losses.append(data_loss.item())
            physics_losses.append(physics_loss.item())

            if epoch % 100 == 0:
                print(f"Epoch {epoch}: Data Loss = {data_loss.item():.6f}, Physics Loss = {physics_loss.item():.6f}")

        return train_losses, physics_losses

    def extract_features(self, data):
        """从PINN中提取特征用于XGBoost"""
        self.pinn.eval()
        with torch.no_grad():
            # 获取PINN的中间层特征
            x = self.pinn.activation(self.pinn.fc1(data))
            x = self.pinn.activation(self.pinn.fc2(x))
            pinn_features = self.pinn.activation(self.pinn.fc3(x))

            # 获取PINN的输出
            pinn_output = self.pinn.fc4(pinn_features)

            # 合并原始特征、PINN特征和PINN输出
            original_features = data.cpu().numpy()
            pinn_features_np = pinn_features.cpu().numpy()
            pinn_output_np = pinn_output.cpu().numpy()

            # 重塑pinn_output以匹配样本数
            if len(pinn_output_np.shape) > 2:
                pinn_output_np = pinn_output_np.reshape(pinn_output_np.shape[0], -1)

            combined_features = np.concatenate([
                original_features,
                pinn_features_np,
                pinn_output_np
            ], axis=1)

        return combined_features

    def fit(self, train_data, train_target, epochs=1000, lr=0.005):
        """训练整个模型"""
        print("Training PINN...")
        train_losses, physics_losses = self.train_pinn(train_data, train_target, epochs, lr)

        print("Extracting features for XGBoost...")
        # 提取训练特征
        train_features = self.extract_features(train_data)

        # 标准化特征
        train_features_scaled = self.scaler.fit_transform(train_features)

        # 将目标数据转换为numpy格式
        if isinstance(train_target, torch.Tensor):
            train_target_np = train_target.cpu().numpy()
        else:
            train_target_np = train_target

        # 如果是多步预测，需要重塑目标数据
        if len(train_target_np.shape) > 1 and train_target_np.shape[1] > 1:
            # 为每个时间步训练单独的模型
            self.xgb_models = []
            for step in range(train_target_np.shape[1]):
                print(f"Training XGBoost for step {step + 1}...")
                xgb_model = xgb.XGBRegressor(**xgb_params)
                xgb_model.fit(train_features_scaled, train_target_np[:, step])
                self.xgb_models.append(xgb_model)
        else:
            # 单步预测
            print("Training XGBoost...")
            self.xgb_model.fit(train_features_scaled, train_target_np.ravel())

        self.is_fitted = True
        return train_losses, physics_losses

    def predict(self, test_data):
        """预测"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        # 提取测试特征
        test_features = self.extract_features(test_data)
        test_features_scaled = self.scaler.transform(test_features)

        # 使用XGBoost进行预测
        if hasattr(self, 'xgb_models'):
            # 多步预测
            predictions = []
            for model in self.xgb_models:
                step_pred = model.predict(test_features_scaled)
                predictions.append(step_pred)
            predictions = np.column_stack(predictions)
        else:
            # 单步预测
            predictions = self.xgb_model.predict(test_features_scaled)

        return predictions


def custom_normalization(data):
    capacity_col = data[:, -1]  # 获取Capacity列
    min_capacity = np.min(capacity_col)
    max_capacity = np.max(capacity_col)
    capacity_col_normalized = (capacity_col - min_capacity) / (max_capacity - min_capacity)

    other_cols = data[:, :-1]
    min_vals = np.min(other_cols, axis=0)
    max_vals = np.max(other_cols, axis=0)
    other_cols_normalized = 2 * (other_cols - min_vals) / (max_vals - min_vals) - 1

    return np.column_stack((other_cols_normalized, capacity_col_normalized))


def build_sequences(text, window_size, forecast_steps):
    x, y = [], []
    for i in range(len(text) - window_size - forecast_steps):
        sequence = text[i:i + window_size, :]  # 选择每一时刻的特征
        target = text[i + window_size:i + window_size + forecast_steps, -1]  # 预测未来forecast_steps个时间步的capacity
        x.append(sequence)
        y.append(target)
    return np.array(x), np.array(y, dtype=np.float32)


def min_max_normalization_per_feature(data):
    # 如果输入数据是 NumPy 数组，将其转换为 PyTorch 张量
    if isinstance(data, np.ndarray):
        data = torch.tensor(data, dtype=torch.float32)

    # 对每个特征（列）进行最小-最大归一化
    min_vals = torch.min(data, dim=0)[0]
    max_vals = torch.max(data, dim=0)[0]
    return (data - min_vals) / (max_vals - min_vals + 1e-6)


# 评估函数
def evaluation(y_test, y_predict):
    mae = mean_absolute_error(y_test, y_predict)
    mse = mean_squared_error(y_test, y_predict)
    rmse = sqrt(mean_squared_error(y_test, y_predict))
    return mae, mse, rmse


# 数据加载和处理
folder_path = '../NASA Cleaned'  # 替换为电池文件所在的文件夹路径
battery_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]  # 获取所有CSV文件
all_data = []
all_target = []
battery_data = {}

# 对每个电池的数据进行处理
for file_name in battery_files:
    file_path = os.path.join(folder_path, file_name)
    df = pd.read_csv(file_path)

    # 提取后8列特征
    features = df.iloc[:, 1:].values  # 获取后8列特征
    last_column = df.iloc[:, -1].values  # 获取最后一列特征

    features = np.column_stack((features[:, 0:5], last_column / 2))
    features = min_max_normalization_per_feature(features)

    # 创建序列数据
    data, target = build_sequences(features, window_size, forecast_steps)

    # 重塑数据以适应模型输入
    data_reshaped = data.reshape(data.shape[0], -1)  # 展平时间窗口

    # 将每个电池的数据保存到字典中
    battery_data[file_name] = (data_reshaped, target)


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


# 交叉验证
maes, rmses = [], []

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

for test_battery, (test_data, test_target) in battery_data.items():
    print(f"Testing on battery: {test_battery}")

    train_data = []
    train_target = []
    for battery, (data, target) in battery_data.items():
        if battery != test_battery:
            train_data.append(data)
            train_target.append(target)

    train_data = np.concatenate(train_data)
    train_target = np.concatenate(train_target)

    # 转换为张量
    train_data_tensor = torch.tensor(train_data, requires_grad=True, dtype=torch.float32).to(device)
    train_target_tensor = torch.tensor(train_target, dtype=torch.float32).to(device)
    test_data_tensor = torch.tensor(test_data, requires_grad=True, dtype=torch.float32).to(device)
    test_target_tensor = torch.tensor(test_target, dtype=torch.float32).to(device)

    setup_seed(0)

    # 创建PINN+XGBoost模型
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

    # 预测
    predictions = model.predict(test_data_tensor)

    # 评估
    test_np = test_target_tensor.detach().cpu().numpy()
    if len(predictions.shape) == 1:
        predictions = predictions.reshape(-1, 1)
    if len(test_np.shape) == 1:
        test_np = test_np.reshape(-1, 1)

    mae, mse, rmse = evaluation(test_np.ravel(), predictions.ravel())
    print(f"RMSE: {rmse * 100:.3f}, MAE: {mae * 100:.3f}")

    maes.append(mae * 100)
    rmses.append(rmse * 100)

# 汇总交叉验证结果
print("\nCross-validation results:")
print(f"Average RMSE: {np.mean(rmses):.3f}")
print(f"Average MAE: {np.mean(maes):.3f}")
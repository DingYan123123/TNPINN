import os
from math import sqrt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error
from matplotlib import pyplot as plt
import pywt  # 导入小波变换库
from torch.utils.data import DataLoader, TensorDataset, random_split

window_size = 20  # 设置窗口大小
forecast_steps = 10  # 预测步数
batch_size = 32  # 设置批次大小
input_size = 6  # 输入特征数
hidden_size = 64 # 隐藏层大小
output_size = forecast_steps  # 输出层大小
epochs = 1000  # 训练轮数
learningrate = 0.05  # 学习率
weight_decay = 0 # L2正则化强度，可调

# 设置随机种子以确保可重复性
def setup_seed(seed):
    np.random.seed(seed)  # Numpy module.
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


# LSTM模型
class WindowedLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, out_size):
        super(WindowedLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, out_size)  # 输出层，用于预测单个值

    def forward(self, input):
        output, _ = self.lstm(input)
        output = self.linear(output[:, -1, :])  # 取序列最后一个时间步的输出进行预测
        return output

def min_max_normalization(data):
    """
    Min-Max归一化方法，将数据缩放到 [0, 1] 范围内
    """
    return (data - np.min(data, axis=0)) / (np.max(data, axis=0) - np.min(data, axis=0))

def z_score_standardization(data):
    """
    Z-score标准化方法，将数据转换为均值为0，标准差为1的分布
    """
    return (data - np.mean(data, axis=0)) / np.std(data, axis=0)
# 归一化处理：将数据缩放到 [0, 1] 之间
def custom_normalization(data):
    capacity_col = data[:, -1]
    max_capacity = np.max(capacity_col)
    SOH = capacity_col / max_capacity
    max_SOH = np.max(SOH)
    min_SOH = np.min(SOH)
    capacity_col_normalized = 2 * (SOH - min_SOH) / (max_SOH - min_SOH) - 1
    other_cols = data[:, :-1]
    min_vals = np.min(other_cols, axis=0)
    max_vals = np.max(other_cols, axis=0)
    other_cols_normalized = 2 * (other_cols - min_vals) / (max_vals - min_vals) - 1
    return np.column_stack((other_cols_normalized, capacity_col_normalized))


# 小波变换去噪函数（软阈值法）
def wavelet_denoising(data, wavelet='db4', level=4, threshold=0.1):
    coeffs = pywt.wavedec(data, wavelet, level=level)
    coeffs = [pywt.threshold(c, threshold, mode='soft') for c in coeffs]
    return pywt.waverec(coeffs, wavelet)


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
    rmse = sqrt(mean_squared_error(y_test, y_predict))
    return mae, mse, rmse


# 早停类
class EarlyStopping:
    def __init__(self, patience=10, verbose=False, delta=0):
        """
        :param patience: 多少个epoch验证集性能没有提升就停止训练
        :param verbose: 是否打印停止信息
        :param delta: 最小性能提升，低于这个值时认为没有提升
        """
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.counter = 0
        self.best_model_wts = None

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.best_model_wts = model.state_dict()
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_model_wts = model.state_dict()
            self.counter = 0


# 数据加载和处理
folder_path = '../CALCE'  # 替换为电池文件所在的文件夹路径
battery_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]  # 获取所有CSV文件
battery_data = {}

# 对每个电池的数据进行处理
for file_name in battery_files:
    file_path = os.path.join(folder_path, file_name)
    df = pd.read_csv(file_path)

    # 提取后8列特征
    features = df.iloc[:, 1:].values  # 获取后8列特征
    last_column = df.iloc[:, -1].values # 获取最后一列特征
    # 应用小波变换去噪（软阈值处理）
    features = min_max_normalization(features)

    # 在这里加入小波分解代码并绘制图形
    # 假设需要对最后一列特征进行小波变换去噪处理
    # coeffs = pywt.wavedec(last_column, 'db4', level=4)
    # fig, axes = plt.subplots(5, 1, figsize=(12, 8))

    # 绘制原始数据
    # axes[0].plot(last_column, label="Original Signal", color='b')
    # axes[0].set_title("Original Signal")
    # axes[0].legend()

    # 绘制每一层的小波分解结果
    # for i in range(4):
    #     axes[i + 1].plot(coeffs[i], color='g',linewidth=5)
    #     # axes[i + 1].set_title(f"Approximation (Level {4-i})")
    #     axes[i + 1].legend()
    #
    # plt.tight_layout()
    # plt.show()

    # 创建数据和目标
    data, target = build_sequences(features, window_size, forecast_steps)

    # 将每个电池的数据保存到字典中
    battery_data[file_name] = (data, target)

# 交叉验证
maes, rmses = [], []

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# 训练过程
for test_battery, (test_data, test_target) in battery_data.items():
    print(f"Testing on battery: {test_battery}")

    # 训练集：所有电池数据，除了当前测试电池
    train_data = []
    train_target = []
    for battery, (data, target) in battery_data.items():
        if battery != test_battery:
            train_data.append(data)
            train_target.append(target)

    train_data = np.concatenate(train_data)
    train_target = np.concatenate(train_target)

    # 转换为PyTorch张量，并将其移动到GPU上（如果可用）
    train_data_tensor = torch.tensor(train_data, dtype=torch.float32).to(device)
    train_target_tensor = torch.tensor(train_target, dtype=torch.float32).to(device)
    test_data_tensor = torch.tensor(test_data, dtype=torch.float32).to(device)
    test_target_tensor = torch.tensor(test_target, dtype=torch.float32).to(device)

    # 创建数据加载器（DataLoader）
    train_dataset = TensorDataset(train_data_tensor, train_target_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 模型参数
    setup_seed(0)
    model = WindowedLSTM(input_size, hidden_size, output_size).to(device)

    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), weight_decay=weight_decay, lr=learningrate)

    # 早停实例化
    early_stopping = EarlyStopping(patience=5, verbose=True)

    # 训练损失记录
    train_losses = []
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            inputs, targets = inputs.to(device), targets.to(device)

            # 前向传播
            output = model(inputs)
            loss = criterion(output, targets)  # 计算损失

            # 反向传播
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        train_losses.append(epoch_loss / len(train_loader))

        # 在验证集上计算损失
        model.eval()
        with torch.no_grad():
            val_pred = model(test_data_tensor)
            val_loss = criterion(val_pred, test_target_tensor)

        # 使用早停
        early_stopping(val_loss.item(), model)
        if early_stopping.early_stop:
            print("Early stopping")
            model.load_state_dict(early_stopping.best_model_wts)
            break

    # 使用模型进行预测
    model.eval()
    with torch.no_grad():
        pred = model(test_data_tensor)

    # 评估模型
    pred_np = pred.detach().squeeze().cpu().numpy()
    # print(pred_np)
    test_np = test_target_tensor.detach().squeeze().cpu().numpy()
    mae, mse, rmse = evaluation(test_np, pred_np)
    print(f"RMSE: {rmse * 100 :.3f}, MAE: {mae * 100 :.3f}")

    # 保存评估指标
    maes.append(mae)
    rmses.append(rmse)

    # # 绘制训练损失图
    # plt.figure(figsize=(12, 6))
    # plt.plot(train_losses, label="Training Loss")
    # plt.title(f"Training Loss over Epochs for {test_battery}")
    # plt.xlabel("Epochs")
    # plt.ylabel("Loss")
    # plt.legend()
    # plt.grid(True)
    # plt.show()
    #
    # 绘制预测与真实值的对比图
    test_np = test_np.flatten()
    pred_np = pred_np.flatten()

    plt.figure(figsize=(8, 8))
    plt.scatter(test_np, pred_np, s=100, color='dodgerblue', alpha=0.8)
    plt.plot([min(test_np), max(test_np)], [min(test_np), max(test_np)], 'r--', label='Ideal')
    plt.xlabel("True Values")
    plt.ylabel("Predicted Values")
    plt.legend()
    plt.grid(True)
    plt.show()

# 汇总交叉验证结果
print("\nCross-validation results:")
print(f"Average RMSE: {np.mean(rmses) * 100:.3f}")
print(f"Average MAE: {np.mean(maes) * 100:.3f}")

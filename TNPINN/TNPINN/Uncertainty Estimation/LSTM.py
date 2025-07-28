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

# 参数设置
window_size = 30  # 设置窗口大小
forecast_steps = 10  # 预测步数
batch_size = 32  # 设置批次大小
input_size = 6  # 输入特征数
hidden_size = 64  # 隐藏层大小
output_size = forecast_steps  # 输出层大小
epochs = 1000  # 训练轮数
learningrate = 0.05  # 学习率
weight_decay = 0  # L2正则化强度
dropout_rate = 0.05  # Dropout率，用于蒙特卡洛不确定性估计
mc_samples = 100  # 蒙特卡洛采样次数


# 设置随机种子以确保可重复性
def setup_seed(seed):
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


# LSTM模型（加入Dropout）
class WindowedLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, out_size, dropout_rate=dropout_rate):
        super(WindowedLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)  # 增加Dropout层
        self.linear = nn.Linear(hidden_size, out_size)

    def forward(self, input, mc_dropout=False):
        output, _ = self.lstm(input)
        if mc_dropout:
            output = self.dropout(output)  # 在推理阶段保留Dropout
        output = self.linear(output[:, -1, :])  # 取序列最后一个时间步
        return output


# 归一化处理
def min_max_normalization(data):
    return (data - np.min(data, axis=0)) / (np.max(data, axis=0) - np.min(data, axis=0))


def z_score_standardization(data):
    return (data - np.mean(data, axis=0)) / np.std(data, axis=0)


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


# 小波变换去噪
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


# 不确定性评估指标
def calculate_uncertainty_metrics(y_true, ci_lower, ci_upper, confidence_level=0.95, eta=None):
    """
    计算不确定性评估指标：PICP, PINAW, CWC

    Args:
        y_true: 真实值 [n_samples, forecast_steps]
        ci_lower: 置信区间下界 [n_samples, forecast_steps]
        ci_upper: 置信区间上界 [n_samples, forecast_steps]
        confidence_level: 置信水平，默认0.95
        eta: 惩罚因子，如果为None则自适应设置

    Returns:
        picp: Prediction Interval Coverage Probability
        pinaw: Prediction Interval Normalized Average Width
        cwc: Coverage Width-based Criterion
        eta_used: 实际使用的eta值
    """
    # PICP (Prediction Interval Coverage Probability)
    coverage = (y_true >= ci_lower) & (y_true <= ci_upper)
    picp = np.mean(coverage)

    # PINAW (Prediction Interval Normalized Average Width)
    interval_width = ci_upper - ci_lower
    y_range = np.max(y_true) - np.min(y_true)
    pinaw = np.mean(interval_width) / y_range

    # 自适应eta设置
    if eta is None:
        coverage_deficit = confidence_level - picp
        if coverage_deficit <= 0:
            eta = 0.1  # 覆盖率满足时使用较小的eta
        elif coverage_deficit <= 0.05:  # 覆盖率接近目标
            eta = 0.3
        elif coverage_deficit <= 0.10:  # 覆盖率中等偏差
            eta = 0.5
        else:  # 覆盖率严重不足
            eta = 1.0

    # CWC (Coverage Width-based Criterion)
    if picp >= confidence_level:
        # 如果覆盖率满足要求，CWC = PINAW
        cwc = pinaw
    else:
        # 如果覆盖率不足，增加惩罚项
        coverage_penalty = eta * (confidence_level - picp) / confidence_level
        cwc = pinaw * (1 + coverage_penalty)

    return picp, pinaw, cwc, eta


# 早停类
class EarlyStopping:
    def __init__(self, patience=10, verbose=False, delta=0):
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


# 蒙特卡洛不确定性估计
def mc_uncertainty_estimation(model, input_data, mc_samples=100):
    model.train()  # 保持Dropout活跃
    predictions = []
    for _ in range(mc_samples):
        with torch.no_grad():
            pred = model(input_data, mc_dropout=True)
        predictions.append(pred.cpu().numpy())
    predictions = np.stack(predictions, axis=0)  # [mc_samples, batch_size, forecast_steps]
    mean_pred = np.mean(predictions, axis=0)  # 均值
    std_pred = np.std(predictions, axis=0)  # 标准差
    ci_lower = mean_pred - 1.96 * std_pred  # 95%置信区间下界
    ci_upper = mean_pred + 1.96 * std_pred  # 95%置信区间上界
    return mean_pred, std_pred, ci_lower, ci_upper
def mc_uncertainty_estimation(model, input_data, mc_samples=100, lower_q=2.5, upper_q=97.5):
    """
    使用蒙特卡洛 Dropout 进行不确定性估计，基于分位数构造预测区间。

    参数:
        model: 已训练模型
        input_data: 输入数据（tensor）
        mc_samples: 采样次数
        lower_q: 下分位数 (默认2.5，对应95% CI)
        upper_q: 上分位数 (默认97.5，对应95% CI)

    返回:
        mean_pred: 平均预测值
        std_pred: 标准差（可选用于分析）
        ci_lower: 分位数下界
        ci_upper: 分位数上界
    """
    model.train()  # 激活 Dropout
    predictions = []

    for _ in range(mc_samples):
        pred, _, _ = model(input_data, mc_dropout=True)
        predictions.append(pred.detach().cpu().numpy())

    predictions = np.stack(predictions, axis=0)  # [mc_samples, N, forecast_steps]
    mean_pred = np.mean(predictions, axis=0)
    std_pred = np.std(predictions, axis=0)
    ci_lower = np.percentile(predictions, 2.5, axis=0)
    ci_upper = np.percentile(predictions, 97.5, axis=0)

# 数据加载和处理
folder_path = '../NASA Cleaned'  # 替换为实际路径
battery_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
battery_data = {}

for file_name in battery_files:
    file_path = os.path.join(folder_path, file_name)
    df = pd.read_csv(file_path)
    features = df.iloc[:, 1:].values
    features = custom_normalization(features)
    data, target = build_sequences(features, window_size, forecast_steps)
    battery_data[file_name] = (data, target)

# 交叉验证
maes, rmses = [], []
picps, pinaws, cwcs, etas = [], [], [], []  # 添加eta存储
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# 训练和测试
for test_battery, (test_data, test_target) in battery_data.items():
    print(f"Testing on battery: {test_battery}")

    # 训练集：排除当前测试电池
    train_data = []
    train_target = []
    for battery, (data, target) in battery_data.items():
        if battery != test_battery:
            train_data.append(data)
            train_target.append(target)
    train_data = np.concatenate(train_data)
    train_target = np.concatenate(train_target)

    # 转换为PyTorch张量
    train_data_tensor = torch.tensor(train_data, dtype=torch.float32).to(device)
    train_target_tensor = torch.tensor(train_target, dtype=torch.float32).to(device)
    test_data_tensor = torch.tensor(test_data, dtype=torch.float32).to(device)
    test_target_tensor = torch.tensor(test_target, dtype=torch.float32).to(device)

    # 创建数据加载器
    train_dataset = TensorDataset(train_data_tensor, train_target_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 初始化模型
    setup_seed(0)
    model = WindowedLSTM(input_size, hidden_size, output_size, dropout_rate).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), weight_decay=weight_decay, lr=learningrate)
    early_stopping = EarlyStopping(patience=5, verbose=True)

    # 训练
    train_losses = []
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            inputs, targets = inputs.to(device), targets.to(device)
            output = model(inputs)
            loss = criterion(output, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        train_losses.append(epoch_loss / len(train_loader))

        # 验证
        model.eval()
        with torch.no_grad():
            val_pred = model(test_data_tensor)
            val_loss = criterion(val_pred, test_target_tensor)
        early_stopping(val_loss.item(), model)
        if early_stopping.early_stop:
            print("Early stopping")
            model.load_state_dict(early_stopping.best_model_wts)
            break

    # 蒙特卡洛不确定性估计
    mean_pred, std_pred, ci_lower, ci_upper = mc_uncertainty_estimation(model, test_data_tensor, mc_samples)

    # 评估
    test_np = test_target_tensor.cpu().numpy()
    mae, mse, rmse = evaluation(test_np, mean_pred)
    print(f"RMSE: {rmse * 100 :.3f}, MAE: {mae * 100 :.3f}")

    # 计算不确定性指标
    picp, pinaw, cwc, eta_used = calculate_uncertainty_metrics(test_np, ci_lower, ci_upper)
    print(f"PICP (95% Coverage): {picp * 100:.2f}%")
    print(f"PINAW (Normalized Width): {pinaw:.4f}")
    print(f"CWC (Coverage-Width Criterion): {cwc:.4f} (eta={eta_used:.1f})")

    maes.append(mae)
    rmses.append(rmse)
    picps.append(picp)
    pinaws.append(pinaw)
    cwcs.append(cwc)
    etas.append(eta_used)

    # 可视化预测和置信区间（示例：第一个样本的预测序列）
    plt.figure(figsize=(12, 6))
    time_steps = np.arange(forecast_steps)
    sample_idx = 1  # 第一个样本
    plt.plot(time_steps, test_np[sample_idx], label="True", marker='o')
    plt.plot(time_steps, mean_pred[sample_idx], label="Predicted", marker='x')
    plt.fill_between(time_steps, ci_lower[sample_idx], ci_upper[sample_idx],
                     color="red", alpha=0.2, label="95% CI")
    # plt.title(f"Prediction with 95% CI for {test_battery}", fontsize=18)
    plt.xlabel("Time Step", fontsize=20)  # 放大横坐标标签字体
    plt.ylabel("SOH(%)", fontsize=20)  # 放大纵坐标标签字体
    plt.tick_params(axis='both', labelsize=20)  # 放大横纵坐标刻度字体
    plt.legend(fontsize=20)  # 放大图例字体
    plt.legend()
    plt.grid(True)
    plt.show()

    print("-" * 60)

# 汇总结果
print("\n" + "=" * 60)
print("CROSS-VALIDATION RESULTS SUMMARY")
print("=" * 60)
print(f"Average RMSE: {np.mean(rmses) * 100:.3f} ± {np.std(rmses) * 100:.3f}")
print(f"Average MAE: {np.mean(maes) * 100:.3f} ± {np.std(maes) * 100:.3f}")
print("\nUNCERTAINTY QUANTIFICATION METRICS:")
print(f"Average PICP: {np.mean(picps) * 100:.2f}% ± {np.std(picps) * 100:.2f}%")
print(f"Average PINAW: {np.mean(pinaws):.4f} ± {np.std(pinaws):.4f}")
print(f"Average CWC: {np.mean(cwcs):.4f} ± {np.std(cwcs):.4f} (avg eta={np.mean(etas):.1f})")
print("=" * 60)

# 详细结果表格
print("\nDETAILED RESULTS BY BATTERY:")
print("-" * 110)
print(f"{'Battery':<20} {'RMSE(%)':<10} {'MAE(%)':<10} {'PICP(%)':<10} {'PINAW':<8} {'CWC':<8} {'ETA':<5}")
print("-" * 110)
for i, (battery_name, _) in enumerate(battery_data.items()):
    print(f"{battery_name:<20} {rmses[i] * 100:<10.3f} {maes[i] * 100:<10.3f} "
          f"{picps[i] * 100:<10.2f} {pinaws[i]:<8.4f} {cwcs[i]:<8.4f} {etas[i]:<5.1f}")
print("-" * 110)
import os
from math import sqrt
import numpy as np
import pandas as pd
import pywt
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error
from matplotlib import pyplot as plt


window_size = 20  # 设置窗口大小
forecast_steps = 10
# 模型参数
input_size = 6  # 输入特征数
hidden_size = 64 # 隐藏层大小
output_size = forecast_steps  # 输出层大小
HPM_hidden_size = 2
epochs = 1000  # 训练轮数
learningrate = 0.005  # 学习率


# 设置随机种子
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
# 小波变换去噪函数（软阈值法）
def wavelet_denoising(data, wavelet='db4', level=4, threshold=0.1):
    coeffs = pywt.wavedec(data, wavelet, level=level)
    coeffs = [pywt.threshold(c, threshold, mode='soft') for c in coeffs]
    return pywt.waverec(coeffs, wavelet)


# LSTM模型
class WindowedLSTM(nn.Module):
    def __init__(self, input_size,window_size, hidden_size, HPM_hidden_size, out_size):
        super(WindowedLSTM, self).__init__()
        self.lstm_cell = nn.LSTMCell(input_size*window_size, hidden_size)
        self.lstm0 = nn.LSTMCell(input_size, hidden_size)
        self.lstm1 = nn.LSTMCell(hidden_size, hidden_size)
        self.dropout = nn.Dropout(p=0.05)
        self.linear = nn.Linear(hidden_size, out_size)
        self.hpm = LSTMHPM(out_size + hidden_size * 2 + input_size*window_size, HPM_hidden_size, input_size*window_size)

    def forward(self, input):

        seq_len, batch_size, _ = input.size()
        outputs = []
        out_xs = []
        old_hts = []
        outputs_hiddens = []
        h_t0 = torch.zeros(batch_size, hidden_size, dtype=torch.float, requires_grad=True, device=device)  # 初始化隐藏状态
        c_t0 = torch.zeros(batch_size, hidden_size, dtype=torch.float, requires_grad=True, device=device)  # 初始化细胞状态
        h_t1 = torch.zeros(batch_size, hidden_size, dtype=torch.float, requires_grad=True, device=device)  # 初始化隐藏状态
        c_t1 = torch.zeros(batch_size, hidden_size, dtype=torch.float, requires_grad=True, device=device)  # 初始化细胞状态
        for i in range(seq_len):  # 滑动时间窗口
            old_h = h_t0
            old_hts.append(old_h)
            window_data = input[i]
            h_t0, c_t0 = self.lstm_cell(window_data, (old_h, c_t0))  # LSTMCell 前向传播
            h_t0, c_t0 = self.dropout(h_t0), self.dropout(c_t0)
            h_t1, c_t1 = self.lstm1(h_t0, (h_t1, c_t1))
            h_t1, c_t1 = self.dropout(h_t1), self.dropout(c_t1)
            output = self.linear(h_t1)  # 线性层，用于预测单个值
            output_hidden = torch.autograd.grad(
                output, old_h,
                grad_outputs=torch.ones_like(output),
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
                allow_unused=True
            )[0]
            if i == 0:
                output_hidden = torch.zeros(batch_size, hidden_size, device=device)
                outputs_hiddens.append(output_hidden)
            else:
                outputs_hiddens.append(output_hidden)
            output_x = torch.autograd.grad(
                output, window_data,
                grad_outputs=torch.ones_like(output),
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
                allow_unused=True
            )[0]
            if i == 0:
                output_x = torch.zeros(batch_size, _, device=device)
                out_xs.append(output_x)
            else:
                out_xs.append(output_x)
            outputs.append(output)
        outputs = torch.stack(outputs, dim=1).squeeze(2)  # 将预测结果拼接并去掉最后一个维度
        out_xs = torch.cat(out_xs)
        outputs_hiddens = torch.cat(outputs_hiddens)
        outputs_hiddens = min_max_normalization_per_feature(outputs_hiddens)
        outputs = torch.transpose(outputs, 0, 1)
        old_hts = torch.cat(old_hts)
        old_hts = min_max_normalization_per_feature(old_hts)
        input = torch.squeeze(input, 1)
        outputs = outputs.squeeze(1)
        output_data = torch.cat((outputs, outputs_hiddens, input, old_hts), dim=1)
        G = self.hpm(output_data)
        F = out_xs - G
        AllF_x = torch.autograd.grad(
            F, output_data,
            grad_outputs=torch.ones_like(F),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        return outputs, F, AllF_x

class LSTMHPM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMHPM, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, output_size)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.layer1(x)
        x = self.tanh(x)
        x = self.layer2(x)
        return x





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
        target = text[i + window_size:i + window_size + forecast_steps, -1]  # 预测未来20个时间步的cumulative_charge
        x.append(sequence)
        y.append(target)
    return np.array(x), np.array(y, dtype=np.float32)

def min_max_normalization_per_feature(data):
    # 如果输入数据是 NumPy 数组，将其转换为 PyTorch 张量
    if isinstance(data, np.ndarray):
        data = torch.tensor(data, dtype=torch.float32)

    # 对每个特征（列）进行最小-最大归一化
    min_vals = torch.min(data, dim=0)[0]  # 每个特征的最小值，返回最小值和索引，这里我们取最小值
    max_vals = torch.max(data, dim=0)[0]  # 每个特征的最大值，返回最大值和索引，这里我们取最大值
    return (data - min_vals) / (max_vals - min_vals + 1e-6)  # 归一化公式，防止除以0的错误
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
    last_column = df.iloc[:, -1].values # 获取最后一列特征
    # 应用小波变换去噪（软阈值处理）
    # last_column = np.apply_along_axis(wavelet_denoising, 0, last_column)
    features = np.column_stack((features[:,0:5], last_column/2))
    # 应用小波变换去噪（软阈值处理）
    features1 = np.apply_along_axis(wavelet_denoising, 0, features[:,0:5])  # 对每一列特征应用小波去噪
    features = np.column_stack((features1, last_column))
    features = min_max_normalization_per_feature(features)
    # 创建数据和目标
    data, target = build_sequences(features, window_size, forecast_steps)

    # 将每个电池的数据保存到字典中
    battery_data[file_name] = (data, target)
    # print(battery_data)


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

    train_data_tensor = torch.tensor(train_data, requires_grad=True, dtype=torch.float32).to(device)
    train_data_tensor = train_data_tensor.unsqueeze(1)
    train_data_tensor = train_data_tensor.reshape(len(train_data_tensor), 1, -1)
    train_target_tensor = torch.tensor(train_target, dtype=torch.float32).to(device)
    test_data_tensor = torch.tensor(test_data, requires_grad=True, dtype=torch.float32).to(device)
    test_data_tensor = test_data_tensor.unsqueeze(1)
    test_data_tensor = test_data_tensor.reshape(len(test_data_tensor), 1, -1)
    test_target_tensor = torch.tensor(test_target, dtype=torch.float32).to(device)

    setup_seed(0)

    model = WindowedLSTM(input_size, window_size, hidden_size, HPM_hidden_size, output_size).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learningrate)

    early_stopping = EarlyStopping(patience=100, delta=0.01)

    train_losses = []
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output, Fx, Fxs = model(train_data_tensor)

        output_cpu = output.cpu()
        train_target_cpu = train_target_tensor.cpu()
        Fx_cpu = Fx[1].cpu()
        Fxs_cpu = Fxs[1].cpu()

        loss = torch.sum((output_cpu.squeeze() - train_target_cpu.squeeze()) ** 2) + torch.sum(Fx_cpu ** 2) + torch.sum(
            Fxs_cpu ** 2)
        if (epoch % 100) == 0:
            print(loss)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

        model.eval()
        pred, fx, fxs = model(test_data_tensor)
        pred_np = pred.detach().squeeze().cpu().numpy()
        test_np = test_target_tensor.detach().squeeze().cpu().numpy()
        val_loss = np.mean((pred_np - test_np) ** 2)
        if early_stopping(val_loss):
            print(f"Early stopping triggered at epoch {epoch}")
            break

    pred_np = pred.detach().squeeze().cpu().numpy()
    test_np = test_target_tensor.detach().squeeze().cpu().numpy()
    mae, mse, rmse = evaluation(test_np, pred_np)
    print(f"RMSE: {rmse * 100:.3f}, MAE: {mae * 100:.3f}")

    maes.append(mae * 100)
    rmses.append(rmse * 100)
# 汇总交叉验证结果
print("\nCross-validation results:")
print(f"Average RMSE: {np.mean(rmses):.3f}")
print(f"Average MAE: {np.mean(maes):.3f}")
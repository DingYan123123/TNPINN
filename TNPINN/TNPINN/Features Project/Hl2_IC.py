import pandas as pd
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

# 1. 加载 CSV 数据
data = pd.read_csv('../NASA/B0018charge.csv')

# 2. 获取最大周期数，确定循环次数
max_cycle = data['Cycle'].max()

# 存储每个周期的峰值特征信息
peaks_info = []

# 3. 设置图形大小
plt.figure(figsize=(10, 5))

# 4. 循环遍历每个周期的数据并提取IC曲线的峰值信息
step = 0.004  # 电压微分最小值

for cyc in range(1, max_cycle + 1):  # 使用最大周期数
    cycle_data = data[data['Cycle'] == cyc]  # 提取当前周期的数据
    v = cycle_data['V_charge'].values
    c = cycle_data['C_charge'].values
    t = cycle_data['T_charge'].values

    # 数据清理部分，去除小幅度的电压波动
    dV = [1000]
    Vind = [3.10]
    dQ = [0]
    i = 0
    numV = len(v)

    while i < (numV - 2):
        diff = 0
        j = 0
        # 寻找符合电压微分最小值的间隔点
        while (diff < step and (i + j) < (numV - 1)):
            j += 1
            diff = abs(v[i + j] - v[i])

        # 判断是否达到结束条件
        if (diff < step or v[i] > 4.194):
            i += numV - 2
            continue
        else:
            dt = np.diff(t[i:i + j + 1]) / 3600  # 单位：小时
            ch_c1 = c[i:i + j + 1]
            dc = ch_c1[:-1] + np.diff(ch_c1) * 0.5  # 电流微分，每两个点取均值，单位：A
            dQ.append(np.sum(dc * dt))
            dV.append(diff)
            Vind.append(v[i] * 0.5 + v[i + j] * 0.5)  # 两点电压均值
            i += 1
            continue

    dV.append(dV[-1])
    Vind.append(4.20)
    dQ.append(dQ[-1])

    # 防止V小于3V的部分
    if min(Vind) < 3:
        pass

    dQdV = np.array(dQ) / np.array(dV)  # 电流/电压变化率

    # 去除重复的电压值
    unique_V, unique_indices = np.unique(Vind, return_index=True)
    unique_dQdV = []

    for i in range(len(unique_V)):
        indices = np.where(Vind == unique_V[i])[0]
        mean_dQdV = np.mean(dQdV[indices])
        unique_dQdV.append(mean_dQdV)

    # 5. 使用 find_peaks 检测峰值
    peaks, _ = find_peaks(unique_dQdV, height=0.1)  # 可以调整height参数来过滤较小的峰值

    if len(peaks) > 0:
        # 选择峰值`dQ/dV`最大值对应的电压
        peak_idx = peaks[np.argmax(np.array(unique_dQdV)[peaks])]  # 找到 `dQ/dV` 最大的峰

        peak_voltage = unique_V[peak_idx]
        peak_dQdV = unique_dQdV[peak_idx]

        # 计算峰值的斜率
        left_idx = max(peak_idx - 1, 0)
        right_idx = min(peak_idx + 1, len(unique_dQdV) - 1)
        slope = (unique_dQdV[right_idx] - unique_dQdV[left_idx]) / (unique_V[right_idx] - unique_V[left_idx])

        # 计算峰面积（通过数值积分近似）
        peak_area = np.trapz(unique_dQdV[left_idx:right_idx + 1], unique_V[left_idx:right_idx + 1])

        # 保存当前周期的峰值信息（每个周期一个特征值）
        peaks_info.append({
            'Cycle': cyc,
            'Peak_Voltage': peak_voltage,
            'Peak_dQ/dV': peak_dQdV,
            'Slope': slope,
            'Peak_Area': peak_area
        })

    else:
        # 如果没有检测到峰值，可以填充为NaN或者0
        peaks_info.append({
            'Cycle': cyc,
            'Peak_Voltage': 0,
            'Peak_dQ/dV': 0,
            'Slope': 0,
            'Peak_Area': 0
        })

    # 6. 可选：绘制IC曲线和峰值
    plt.plot(unique_V, unique_dQdV, label=f'Cycle {cyc}')  # 每个周期的IC曲线
    # plt.scatter(unique_V[peaks], np.array(unique_dQdV)[peaks], color='red', zorder=5)  # 绘制峰值

# 7. 保存峰值信息为CSV文件
peaks_df = pd.DataFrame(peaks_info)
peaks_df.to_csv('B18_ic_peaks.csv', index=False)

# 8. 绘制最终的IC曲线图
plt.xlabel('Voltage (V)')
plt.ylabel('dQ/dV (A/V)')
plt.title('IC Curve of Battery Charging')
plt.grid(True)
plt.show()

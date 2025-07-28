import numpy as np
import scipy.io
import pandas as pd

# 加载.mat文件
mat_file = '../NASA/B0018.mat'  # 替换为你的.mat文件路径
excel_file = '../NASA/B0018charge.csv'  # 目标CSV文件路径

# 初始化空列表
V = []
C = []
T = []
Cap = []  # 如果你想要保存容量数据
j = 0
index = []
charge_indices = []  # 用于记录charge过程的索引

# 加载.mat文件中的数据
mat_data = scipy.io.loadmat(mat_file)
data = mat_data['B0018']
real_data = data[0][0][0][0]

# 变量用于标记是否已经跳过第一个charge
skip_first_charge = True

# 遍历数据并将需要的值存入列表
for i in range(len(real_data)):
    if real_data[i][0] == 'charge':
        # 跳过第一个charge过程
        if skip_first_charge:
            skip_first_charge = False
            continue  # 跳过当前循环，进入下一次循环
        j = j + 1
        # 处理charge过程的数据
        C.append(real_data[i][3][0][0][1][0])  # 电流
        V.append(real_data[i][3][0][0][0][0])  # 电压
        T.append(real_data[i][3][0][0][5][0])  # 温度
        array = np.full(len(real_data[i][3][0][0][0][0]), j)  # 循环编号
        index.append(array)


# 展开数据
C = np.concatenate(C)
V = np.concatenate(V)
T = np.concatenate(T)
index = np.concatenate(index)

# 将结果存入DataFrame
df = pd.DataFrame({
    'Cycle': index,
    'C_charge': C,
    'V_charge': V,
    'T_charge': T,
    # 'Capacity': Cap  # 如果你想要使用容量数据，解开注释并保存
})
# 将 'Cycle' 列转换为整数类型
df['Cycle'] = df['Cycle'].astype(int)

# 找到最大Cycle值
max_cycle = df['Cycle'].max()

# 去掉所有Cycle等于最大值的行
df_filtered = df[df['Cycle'] != max_cycle]

# 显示去掉最大Cycle值后的数据
print(df_filtered)

# 保存为CSV文件
df_filtered.to_csv(excel_file, index=False)

print(f"数据已成功保存到 {excel_file}")

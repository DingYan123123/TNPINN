import pandas as pd

# 读取原始的 CSV 文件
df = pd.read_csv('./B18.csv')

# 选择您需要的列（这里以 'Cycle', 'current_mean', 'voltage_mean' 为例，您可以修改为实际需要的列）
selected_columns = ['Cycle','CC Time (s)','CC Time Percentage (%)','CC Area (A*s)','Peak_dQ/dV','disCC Time (s)','Capacity (Ah)']

# 选择所需的列
df_selected = df[selected_columns]

# 将所有的数值列的负值转换为正值
df_selected = df_selected.abs()

# 重新命名列（根据您的需求，这里给出新的列名）
new_column_names = ['Cycle','CC Time (s)','CC Time Percentage (%)','CC Area (A*s)','Peak_dQ/dV','disCC Time (s)','Capacity (Ah)']

# 修改列名
df_selected.columns = new_column_names

# 保存为新的 CSV 文件
df_selected.to_csv('B0018.csv', index=False)

print("新的 CSV 文件已保存。")

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 替换为你的四个文件名
file_paths = [
    './B5.csv',
    './B6.csv',
    './B7.csv',
    './B18.csv'
]

dataframes = []

# 加载并预处理每个文件
for file in file_paths:
    df = pd.read_csv(file)
    if 'Cycle' in df.columns:
        df = df.drop(columns=['Cycle'])  # 删除 Cycle 列
    dataframes.append(df)

# 按行合并所有数据（要求列结构一致）
merged_df = pd.concat(dataframes, axis=0, ignore_index=True)

# 计算 Pearson 相关系数矩阵
corr_matrix = merged_df.corr(method='pearson')

# 绘制热力图
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True)
# plt.title("Pearson Correlation Heatmap (Merged Data, Excl. 'Cycle')")
plt.tight_layout()
plt.show()

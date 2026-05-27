import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# 读取 PaySim
df = pd.read_csv("../data/paysim/ps.csv")

print("原始数据大小:", df.shape)

df['hour'] = df['step'] % 24
# df['day'] = df['step'] // 24
#
df = df.drop(columns=['step'])

# =========================
# 删除无意义列
# =========================

drop_cols = [
    'nameOrig',
    'nameDest',
    'isFlaggedFraud',
    'step'
]

existing_drop_cols = [c for c in drop_cols if c in df.columns]
df = df.drop(columns=existing_drop_cols)

# =========================
# one-hot 编码 type
# =========================

df = pd.get_dummies(
    df,
    columns=['type'],
    prefix='type'
)

# 转 float/int
for c in df.columns:
    if df[c].dtype == bool:
        df[c] = df[c].astype(int)

# =========================
# 标签改名
# =========================

df = df.rename(columns={
    'isFraud': 'Class',
    'amount': 'Amount'
})

# =========================
# 删除缺失
# =========================

df = df.fillna(0)

# =========================
# 调整列顺序
# Class 放最后
# =========================

cols = [c for c in df.columns if c != 'Class'] + ['Class']
df = df[cols]

print("处理后维度:", df.shape)

# =========================
# 构造平衡子集
# =========================

fraud_df = df[df['Class'] == 1]
normal_df = df[df['Class'] == 0]

print("原始欺诈样本:", len(fraud_df))
print("原始正常样本:", len(normal_df))

# -------------------------
# 目标欺诈比例
# -------------------------

TARGET_FRAUD_RATIO = 0.02

# fraud / total = ratio
# total = fraud / ratio

target_total = int(len(fraud_df) / TARGET_FRAUD_RATIO)

target_normal = target_total - len(fraud_df)

print("目标总样本:", target_total)
print("目标正常样本:", target_normal)

# -------------------------
# 下采样正常样本
# -------------------------

normal_subset, _ = train_test_split(
    normal_df,
    train_size=min(target_normal, len(normal_df)),
    random_state=888
)

# -------------------------
# 合并
# -------------------------

df = pd.concat([fraud_df, normal_subset])

# 打乱
df = df.sample(frac=1, random_state=888).reset_index(drop=True)

print("\n最终欺诈比例:")
print(df['Class'].mean())

print("\n最终数据形状:")
print(df.shape)

# =========================
# 保存
# =========================

save_path = "../data/paysim/paysim_processed_subset.csv"

df.to_csv(
    save_path,
    index=False
)

print("保存完成:", save_path)

print("\n最后5列:")
print(df.columns[-5:])

print("\n最后一列:")
print(df.columns[-1])

print("\n欺诈比例:")
print(df['Class'].mean())

print("\n数据形状:")
print(df.shape)
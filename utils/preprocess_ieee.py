import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

trans = pd.read_csv("../data/IEEE-CIS/train_transaction.csv")
identity = pd.read_csv("../data/IEEE-CIS/train_identity.csv")

# merge
df = trans.merge(identity, on='TransactionID', how='left')

# 只保留数值列
df = df.select_dtypes(include=[np.number])

# 删除无意义列
drop_cols = [
    'TransactionID',
    'TransactionDT'
]

existing_drop_cols = [c for c in drop_cols if c in df.columns]
df = df.drop(columns=existing_drop_cols)

missing_ratio = df.isnull().mean()
drop_cols = missing_ratio[missing_ratio > 0.9].index
df = df.drop(columns=drop_cols)

df = df.rename(columns={
    'isFraud': 'Class'
})

# fillna
df = df.fillna(0)

cols = [c for c in df.columns if c != 'Class'] + ['Class']
df = df[cols]

df, _ = train_test_split(
    df,
    train_size=50000,
    random_state=888,
    stratify=df['Class']
)

# 保存
df.to_csv(
    "../data/IEEE-CIS/ieee_cis_processed_subset50000.csv",
    index=False
)

print(df.columns[-5:])
print("最后一列:", df.columns[-1])
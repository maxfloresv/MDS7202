import pandas as pd
import os
import gc
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
import category_encoders as ce

def split_data(**kwargs) -> None:
    train_end = 36 
    val_end = 44 
    test_end = 52 
    ti = kwargs['ti']
    base_dir = ti.xcom_pull(key='base_dir')
    df = ti.xcom_pull(key='final_df')
    train = df[
    df['week'] <= train_end
    ].copy()
    validation = df[
    (df['week'] > train_end) 
    & (df['week'] <= val_end)
    ].copy()
    test = df[
    (df['week'] > val_end) 
    & (df['week'] <= test_end)
    ].copy()
    train_df = pd.concat([X_train, y_train], axis=1)
    val_df = pd.concat([X_val, y_val], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)
    train_df.to_parquet(f"{base_dir}/splits/train.parquet", engine='pyarrow', index=False)
    val_df.to_parquet(f"{base_dir}/splits/val.parquet", engine='pyarrow', index=False)
    test_df.to_parquet(f"{base_dir}/splits/test.parquet", engine='pyarrow', index=False)
    ti.xcom_push(key='train_df', value=train_df)
    ti.xcom_push(key='val_df', value=val_df)
    ti.xcom_push(key='test_df', value=test_df)

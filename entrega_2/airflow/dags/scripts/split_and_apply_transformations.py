import pandas as pd
import joblib
import gc

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer

def split_data(**kwargs) -> None:
    """
    Split the data into train (~80 %) and validation (~20 %) sets.
    """
    SPLIT_YEAR = 2024
    SPLIT_WEEK = 44

    ti = kwargs['ti']
    df_path = ti.xcom_pull(key='final_df_path')
    df = pd.read_parquet(df_path, engine='pyarrow')

    mask_train = (df['year'] == SPLIT_YEAR) & (df['week'] <= SPLIT_WEEK)
    train = df[mask_train].copy()
    val = df[~mask_train].copy()

    base_dir = ti.xcom_pull(key='base_dir')
    splits_dir = f"{base_dir}/splits"

    train_path_out = f"{splits_dir}/train.parquet"
    val_path_out = f"{splits_dir}/val.parquet"

    train.to_parquet(train_path_out, engine='pyarrow', index=False)
    val.to_parquet(val_path_out, engine='pyarrow', index=False)

    ti.xcom_push(key='train_df_path', value=train_path_out)
    ti.xcom_push(key='val_df_path', value=val_path_out)

def create_data_transformations(**kwargs) -> None:
    """
    Creates a ColumnTransformer for Feature Engineering and saves it.
    Separates X and y, and saves them to the splits directory.
    """
    ti = kwargs['ti']

    numerical = [
        'X', 
        'Y', 
        'num_deliver_per_week', 
        'size',
        'year',
        'week_sin',
        'week_cos',
        'items_lag_1',
        'items_rolling_mean_4w',
        'purchased_lag_1'
    ]
    categorical = [
        'customer_type', 
        'category',
        'sub_category', 
        'segment', 
        'package', 
        'brand'
    ]

    ti.xcom_push(key='numerical_features', value=numerical)
    ti.xcom_push(key='categorical_features', value=categorical)

    numerical_pipe = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', MinMaxScaler()) 
    ])

    categorical_pipe = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('numerical', numerical_pipe, numerical),
            ('categorical', categorical_pipe, categorical)
        ], remainder='passthrough'
    )

    base_dir = ti.xcom_pull(key='base_dir')

    train_path = ti.xcom_pull(key='train_df_path')
    val_path = ti.xcom_pull(key='val_df_path')

    train = pd.read_parquet(train_path, engine='pyarrow')
    val = pd.read_parquet(val_path, engine='pyarrow')

    # Removes week as it was used only to holdout.
    X_train = train.drop(columns=['label', 'week'])
    y_train = train['label']

    X_val = val.drop(columns=['label', 'week'])
    y_val = val['label']

    del train, val
    gc.collect()

    # Moves X and y to the splits directory.
    X_train_path_out = f"{base_dir}/splits/X_train.parquet"
    X_val_path_out = f"{base_dir}/splits/X_val.parquet"

    X_train.to_parquet(X_train_path_out, engine='pyarrow', index=False)
    X_val.to_parquet(X_val_path_out, engine='pyarrow', index=False)

    ti.xcom_push(key='X_train_path', value=X_train_path_out)
    ti.xcom_push(key='X_val_path', value=X_val_path_out)

    y_train_path_out = f"{base_dir}/splits/y_train.parquet"
    y_val_path_out = f"{base_dir}/splits/y_val.parquet"

    y_train.to_frame(name='label').to_parquet(
        y_train_path_out, 
        engine='pyarrow', 
        index=False
    )
    y_val.to_frame(name='label').to_parquet(
        y_val_path_out, 
        engine='pyarrow', 
        index=False
    )

    ti.xcom_push(key='y_train_path', value=y_train_path_out)
    ti.xcom_push(key='y_val_path', value=y_val_path_out)

    transformers_dir = f"{base_dir}/transformers"
    feature_engineering_transformer_path_out = (
        f"{transformers_dir}/feature_engineering_transformer.joblib"
    )
    joblib.dump(preprocessor, feature_engineering_transformer_path_out)
    ti.xcom_push(
        key='feature_engineering_transformer_path', 
        value=feature_engineering_transformer_path_out
    )
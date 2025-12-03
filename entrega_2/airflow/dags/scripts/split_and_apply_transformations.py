import pandas as pd
import joblib
import gc

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer

def prepare_full_data(**kwargs) -> None:
    """
    Prepares the full dataframe for Cross-Validation.
    Ensures data is chronologically sorted (important for Rolling Window Cross-Validation).
    """
    ti = kwargs['ti']
    base_dir = ti.xcom_pull(key='base_dir')

    df_path = ti.xcom_pull(key='final_df_path')
    df = pd.read_parquet(df_path, engine='pyarrow')

    # The first priority is to sort the data by year.
    # Includes customer_id and product_id for reproducibility.
    df = df.sort_values(by=[
        'year', 
        'week',
        'customer_id',
        'product_id'
    ])

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
    
    drop_cols = ['label', 'week', 'items']
    X_full = df.drop(columns=drop_cols)
    y_full = df['label']

    splits_dir = f"{base_dir}/splits"
    X_full_path_out = f"{splits_dir}/X_full.parquet"
    y_full_path_out = f"{splits_dir}/y_full.parquet"

    X_full.to_parquet(X_full_path_out, engine='pyarrow', index=False)
    y_full.to_frame(name='label').to_parquet(
        y_full_path_out, 
        engine='pyarrow', 
        index=False
    )

    del df, X_full, y_full
    gc.collect()

    ti.xcom_push(key='X_full_path', value=X_full_path_out)
    ti.xcom_push(key='y_full_path', value=y_full_path_out)

def create_preprocessor_template(**kwargs) -> None:
    """
    Creates the unfitted ColumnTransformer and saves it.
    """
    ti = kwargs['ti']
    base_dir = ti.xcom_pull(key='base_dir')

    numerical = ti.xcom_pull(key='numerical_features')
    categorical = ti.xcom_pull(key='categorical_features')

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

    transformers_dir = f"{base_dir}/transformers"
    feature_engineering_transformer_path_out = (
        f"{transformers_dir}/feature_engineering_transformer.joblib"
    )
    joblib.dump(preprocessor, feature_engineering_transformer_path_out)
    ti.xcom_push(
        key='feature_engineering_transformer_path', 
        value=feature_engineering_transformer_path_out
    )
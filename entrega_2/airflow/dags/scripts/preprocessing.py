import pandas as pd
import numpy as np
import os
import gc

from feature_engine.selection import DropConstantFeatures
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

""" 
These are the environment variables that can be set to configure the script.
If using Docker, they are set in the docker-compose.yml file.
"""
BASE_PATH = os.environ.get('AIRFLOW_HOME', '/opt/airflow')
DEBUG_MODE = os.environ.get('DEBUG_MODE', 'True').lower() == 'true'

def create_folders(**kwargs) -> None:
    """
    Create necessary folders for the current execution date.
    """
    ts = kwargs['ts']
    ti = kwargs['ti']
    safe_ts = ts.replace(":", "_").replace("+", "_").replace("T", "_")
    base_dir = f"{BASE_PATH}/{safe_ts}"
    for dir in [
        'raw', 
        'preprocessed', 
        'splits', 
        'models', 
        'transformers', 
        'studies',
        'images',
        'predictions'
    ]:
        os.makedirs(f"{base_dir}/{dir}", exist_ok=True)
    ti.xcom_push(key='base_dir', value=base_dir)

def remove_transaction_invalid_entries(trx: pd.DataFrame) -> pd.DataFrame:
    """
    Remove invalid entries from the transactions dataframe.

    Parameters
    ----------
    trx : pd.DataFrame
        The transactions dataframe to remove invalid entries from.

    Returns
    ----------
    pd.DataFrame
        The transactions dataframe with the invalid entries removed.
    """
    key_cols = ['product_id', 'order_id']
    duplicates = (
        trx.groupby(key_cols)
            .size()
            .reset_index(name='count')
    )
    duplicates = duplicates[duplicates['count'] > 1]
    negative_items = trx.query('items < 0')[key_cols]

    negative_unique = set(negative_items.apply(tuple, axis=1))
    duplicates_unique = set(duplicates[key_cols].apply(tuple, axis=1))

    # Remove negative items that aren't duplicated.
    negative_not_duped_keys = negative_unique - duplicates_unique
    mask = trx[key_cols].apply(tuple, axis=1).isin(negative_not_duped_keys)
    trx = trx[~mask]

    # Only consider the most frequent item for each product and order combination.
    max_mode = trx.groupby(key_cols)['items'].transform(
        lambda x: x.mode().max()
    )
    # Prevent floating point precision issues.
    proximity_mask = np.isclose(trx['items'], max_mode)
    trx = trx[proximity_mask]

    # Case when there are multiple modes for the same product and order combination.
    trx = trx.drop_duplicates(subset=key_cols, keep='first')

    assert trx[trx['items'] < 0].shape[0] == 0, (
        "Negative items found in the transactions dataframe."
    )

    assert (
        trx.groupby(key_cols)
            .size()
            .reset_index(name='count')
            .query('count > 1')
    ).empty, (
        "Duplicate items found in the transactions dataframe."
    )

    return trx

def remove_unheard_clients(
    clients: pd.DataFrame, 
    trx: pd.DataFrame,
    min_transactions: int = 1
) -> pd.DataFrame:
    """
    Remove clients that have not made any transactions.

    Parameters
    ----------
    clients : pd.DataFrame
        The clients dataframe to remove unheard clients from.
    trx : pd.DataFrame
        The transactions dataframe to get the client ids from.
    min_transactions : int, default=1
        Minimum number of transactions a client must have to be kept.

    Returns
    ----------
    pd.DataFrame
        The dataframe with the clients that have made at least min_transactions transactions.
    """
    counts = trx['customer_id'].value_counts()
    filtered_clients = counts[counts >= min_transactions].index
    clients = clients[clients['customer_id'].isin(filtered_clients)]

    assert clients.shape[0] == len(filtered_clients), (
        "Unheard clients filter was not applied correctly."
    )

    return clients

def remove_unbought_products(
    products: pd.DataFrame, 
    trx: pd.DataFrame,
    min_bought: int = 1
) -> pd.DataFrame:
    """
    Remove products that have not been bought.

    Parameters
    ----------
    products : pd.DataFrame
        The products dataframe to remove unbought products from.
    trx : pd.DataFrame
        The transactions dataframe to get the product ids from.
    min_bought : int, default=1
        Minimum number of times a product must be bought to be kept.

    Returns
    ----------
    pd.DataFrame
        The dataframe with the products that have been purchased at least min_bought times.
    """
    counts = trx['product_id'].value_counts()
    filtered_products = counts[counts >= min_bought].index
    products = products[products['product_id'].isin(filtered_products)]

    assert products.shape[0] == len(filtered_products), (
        "Unbought products filter was not applied correctly."
    )

    return products

def preprocess(**kwargs) -> None:
    """
    Preprocess the data, applying reproducible transformations.
    """
    ti = kwargs['ti']
    base_dir = ti.xcom_pull(key='base_dir')

    raw_dir = f"{base_dir}/raw"
    trx_path = os.path.join(raw_dir, 'transacciones.parquet')
    products_path = f'{BASE_PATH}/data/productos.parquet'
    customers_path = f'{BASE_PATH}/data/clientes.parquet'

    # Although this task is only executed once downloading status is successful, 
    # there may be issues previously. This is why we catch the FileNotFoundError 
    # and raise a custom error.
    try: 
        trx = pd.read_parquet(trx_path, engine='pyarrow')
        products = pd.read_parquet(products_path, engine='pyarrow')
        customers = pd.read_parquet(customers_path, engine='pyarrow')
    except FileNotFoundError as err:
        raise FileNotFoundError(f"Error reading data files: {err}")

    preprocessed_dir = f"{base_dir}/preprocessed"

    """
    Transactions preprocessing must be performed first.
    """
    pipe_trx = Pipeline(steps=[
        ('drop_constant_features', DropConstantFeatures(
            tol=0.85,
            missing_values='ignore'
        )),
        ('remove_transaction_invalid_entries', FunctionTransformer(
            func=remove_transaction_invalid_entries
        ))
    ])
    trx = pipe_trx.fit_transform(trx)
    trx_path_out = os.path.join(preprocessed_dir, 'transacciones.parquet')
    trx.to_parquet(trx_path_out, engine='pyarrow')
    ti.xcom_push(key='preprocessed_trx_path', value=trx_path_out)

    pipe_client = Pipeline(steps=[
        ('drop_constant_features', DropConstantFeatures(
            tol=0.85,
            missing_values='ignore'
        )),
        ('remove_unheard_clients', FunctionTransformer(
            func=remove_unheard_clients,
            kw_args={'trx': trx}
        ))
    ])
    customers = pipe_client.fit_transform(customers)
    customers_path_out = os.path.join(preprocessed_dir, 'clientes.parquet')
    customers.to_parquet(customers_path_out, engine='pyarrow')
    ti.xcom_push(key='preprocessed_customers_path', value=customers_path_out)

    pipe_prod = Pipeline(steps=[
        ('drop_constant_features', DropConstantFeatures(
            tol=0.85,
            missing_values='ignore'
        )),
        ('remove_unbought_products', FunctionTransformer(
            func=remove_unbought_products,
            kw_args={'trx': trx}
        ))
    ])
    products = pipe_prod.fit_transform(products)
    products_path_out = os.path.join(preprocessed_dir, 'productos.parquet')
    products.to_parquet(products_path_out, engine='pyarrow')
    ti.xcom_push(key='preprocessed_products_path', value=products_path_out)

def generate_base_dataframe(**kwargs) -> None:
    """
    Generate the base dataframe. This is executed when all the dataframes are loaded and cleaned.
    """
    ti = kwargs['ti']
    base_dir = ti.xcom_pull(key='base_dir')

    trx_path = ti.xcom_pull(key='preprocessed_trx_path')
    products_path = ti.xcom_pull(key='preprocessed_products_path')
    customers_path = ti.xcom_pull(key='preprocessed_customers_path')

    try:
        trx = pd.read_parquet(trx_path, engine='pyarrow')
        customers = pd.read_parquet(customers_path, engine='pyarrow')
        products = pd.read_parquet(products_path, engine='pyarrow')
    except FileNotFoundError as err:
        raise FileNotFoundError(f"Error reading preprocessed data files: {err}")

    trx['year'] = trx['purchase_date'].dt.isocalendar().year
    trx['week'] = trx['purchase_date'].dt.isocalendar().week

    existing_pairs = trx[['customer_id', 'product_id']].drop_duplicates()
    unique_periods = trx[['year', 'week']].drop_duplicates()

    base = existing_pairs.merge(
        unique_periods, 
        how='cross'
    )

    del existing_pairs, unique_periods
    gc.collect()

    period_trx = (
        trx.groupby(['customer_id', 'product_id', 'year', 'week'])
            .agg({'items': 'sum'})
            .reset_index()
    )

    df = base.merge(
        period_trx,
        on=['customer_id', 'product_id', 'year', 'week'],
        how='left'
    )

    del base, period_trx
    gc.collect()

    df['items'] = df['items'].fillna(0)
    df['label'] = (df['items'] > 0).astype(int)

    df = df.sort_values(
        by=['customer_id', 'product_id', 'year', 'week']
    )

    df['items_lag_1'] = df.groupby(['customer_id', 'product_id'])['items'].shift(1)
    df['items_rolling_mean_4w'] = df.groupby(
        ['customer_id', 'product_id']
    )['items'].transform(
        lambda x: x.shift(1).rolling(
            window=4, 
            min_periods=1
        ).mean()
    )
    df['purchased_lag_1'] = df.groupby(
        ['customer_id', 'product_id']
    )['label'].shift(1)

    df.fillna(0, inplace=True)

    # Uses sine and cosine to encode the week as a periodical variable.
    df['week_sin'] = np.sin(2 * np.pi * df['week'] / 52)
    df['week_cos'] = np.cos(2 * np.pi * df['week'] / 52)

    df = df.merge(customers, on='customer_id', how='left')
    del customers
    gc.collect()

    df = df.merge(products, on='product_id', how='left')
    del products
    gc.collect()

    preprocessed_dir = f"{base_dir}/preprocessed"
    base_path_out = os.path.join(preprocessed_dir, 'base.parquet')
    df.to_parquet(base_path_out, engine='pyarrow')
    ti.xcom_push(key='preprocessed_base_path', value=base_path_out)

def clean_base_dataframe_types(tol: float = 0.1, **kwargs) -> None:
    """
    Clean the base dataframe types, reducing the memory usage.

    Parameters
    ----------
    tol : float, default=0.1
        Tolerance for the percentage of unique values in a column to be considered as categorical.
    """
    ti = kwargs['ti']
    base_path = ti.xcom_pull(key='preprocessed_base_path')

    try:
        df = pd.read_parquet(base_path, engine='pyarrow')
    except FileNotFoundError as err:
        raise FileNotFoundError(f"Error reading base dataframe: {err}")

    # Prevent applying types to columns that don't exist.
    target_types = {
        'customer_id': 'int32',
        'product_id': 'int32',
        'Y': 'float32',
        'X': 'float32',
        'num_deliver_per_week': 'int8',
        'size': 'float16',
        'year': 'int16',  
        'week': 'int8',
        'week_sin': 'float32',
        'week_cos': 'float32',
        'items': 'float32',
        'items_lag_1': 'float32',
        'items_rolling_mean_4w': 'float32',
        'purchased_lag_1': 'int8'
    }
    types_to_apply = {
        col: dtype for col, dtype in target_types.items() if col in df.columns
    }
    if types_to_apply:
        df = df.astype(types_to_apply)
        gc.collect()

    for col in df.select_dtypes(include=['object']).columns.tolist():
        unique = df[col].nunique()
        total_rows = df.shape[0]
        if unique / total_rows < tol:
            df[col] = df[col].astype('category')

    gc.collect()

    if DEBUG_MODE:
        mem = df.memory_usage(deep=True) / 1024**2
        print(f"Memory usage after cleaning types: {mem.sum():.2f} MB")

    df.to_parquet(base_path, engine='pyarrow')
    ti.xcom_push(key='final_df_path', value=base_path)
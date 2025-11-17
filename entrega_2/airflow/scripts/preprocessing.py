import pandas as pd
import os
import gc

BASE_PATH = os.environ.get('AIRFLOW_HOME', '/opt/airflow')

def create_folders(**kwargs) -> None:
    ts = kwargs['ts']
    ti = kwargs['ti']
    safe_ts = ts.replace(":", "_").replace("+", "_").replace("T", "_")
    base_dir = f"{BASE_PATH}/{safe_ts}"
    os.makedirs(f"{base_dir}/raw", exist_ok=True)
    os.makedirs(f"{base_dir}/preprocessed", exist_ok=True)
    os.makedirs(f"{base_dir}/splits", exist_ok=True)
    os.makedirs(f"{base_dir}/models", exist_ok=True)
    ti.xcom_push(key='base_dir', value=base_dir)

def extract_and_load(**kwargs) -> None:
    ti = kwargs['ti']
    base_dir = ti.xcom_pull(key='base_dir')
    transactions_path = os.path.join(BASE_PATH, 'raw', 'transactions.parquet')
    products_path = os.path.join(BASE_PATH, 'raw', 'products.parquet')
    clients_path = os.path.join(BASE_PATH, 'raw', 'clients.parquet')    
    transactions_df = pd.read_parquet(transactions_path)
    products_df = pd.read_parquet(products_path)
    clients_df = pd.read_parquet(clients_path)
    raw_dir = f"{base_dir}/raw"
    if os.path.exists(raw_dir):
        new_files = [
            f for f in os.listdir(raw_dir)
            if f.endswith('.parquet')
        ]
        for file in new_files:
            file_path = os.path.join(raw_dir, file)
            if 'transactions' in file:
                new_df = pd.read_parquet(file_path)
                transactions_df = pd.concat([transactions_df, new_df], ignore_index=True)
            elif 'products' in file:
                new_df = pd.read_parquet(file_path)
                products_df = pd.concat([products_df, new_df], ignore_index=True)
            elif 'clients' in file:
                new_df = pd.read_parquet(file_path)
                clients_df = pd.concat([clients_df, new_df], ignore_index=True)
    clients_df.drop(
        columns=['region_id', 'zone_id', 'num_visit_per_week'],
        inplace=True,
        errors='ignore'
    )  
    ti.xcom_push(key='transactions_df', value=transactions_df)
    ti.xcom_push(key='products_df', value=products_df)
    ti.xcom_push(key='clients_df', value=clients_df)

def negative_items(**kwargs) -> None:
    ti = kwargs['ti']
    transactions_df = ti.xcom_pull(key='transactions_df')
    dupes = transactions_df.groupby(['product_id', 'order_id']).size() > 1
    dupes_keys = set(dupes[dupes].index)
    neg_keys = set(
        transactions_df[transactions_df['items'] < 0]
            [['product_id', 'order_id']]
            .apply(tuple, axis=1)
    )
    unique_neg = neg_keys - dupes_keys
    transactions_df = transactions_df[
        ~transactions_df[['product_id', 'order_id']]
            .apply(tuple, axis=1)
            .isin(unique_neg)
    ].copy()
    mode_max = transactions_df.groupby(['product_id', 'order_id'])['items'].transform(
        lambda x: x.mode().max()
    )
    transactions_df = transactions_df[
        transactions_df['items'] == mode_max
    ].drop_duplicates().copy()
    
    ti.xcom_push(key='transactions_df', value=transactions_df)
    gc.collect()

def filter_active_entities(**kwargs) -> None:
    ti = kwargs['ti']
    transactions_df = ti.xcom_pull(key='transactions_df')
    products_df = ti.xcom_pull(key='products_df')
    clients_df = ti.xcom_pull(key='clients_df')
    active_clients = transactions_df['client_id'].value_counts()
    active_clients = active_clients[active_clients >= 1].index
    clients_df = clients_df[
        clients_df['client_id'].isin(active_clients)
    ].copy()
    active_products = transactions_df['product_id'].unique()
    products_df = products_df[
        products_df['product_id'].isin(active_products)
    ].copy()
    transactions_df = transactions_df[
        transactions_df['client_id'].isin(active_clients) &
        transactions_df['product_id'].isin(active_products)
    ].copy()
    ti.xcom_push(key='transactions_df', value=transactions_df)
    ti.xcom_push(key='products_df', value=products_df)
    ti.xcom_push(key='clients_df', value=clients_df)
    gc.collect()

def create_weekly_base(**kwargs) -> None:
    ti = kwargs['ti']
    transactions_df = ti.xcom_pull(key='transactions_df')
    products_df = ti.xcom_pull(key='products_df')
    clients_df = ti.xcom_pull(key='clients_df')
    transactions_df['week'] = transactions_df['order_date'].dt.isocalendar().week
    customer_ids = clients_df['client_id'].unique()
    product_ids = products_df['product_id'].unique()
    weeks = transactions_df['week'].unique()
    
    base = pd.MultiIndex.from_product(
        [customer_ids, product_ids, weeks],
        names=['client_id', 'product_id', 'week']
    ).to_frame(index=False)
    weekly_trx = (
        transactions_df.groupby(['client_id', 'product_id', 'week'])
            .agg({'items': 'sum'})
            .reset_index()
    )
    df = base.merge(
        weekly_trx,
        on=['client_id', 'product_id', 'week'],
        how='left'
    )
    del base, weekly_trx
    gc.collect()
    ti.xcom_push(key='merged_df', value=df)
    ti.xcom_push(key='transactions_df', value=transactions_df)

def create_labels(**kwargs) -> None:
    ti = kwargs['ti']
    df = ti.xcom_pull(key='merged_df')
    df['items'] = df['items'].fillna(0)
    df['label'] = (df['items'] > 0).astype(int)
    df.drop(columns=['items'], inplace=True)
    ti.xcom_push(key='merged_df', value=df)

def merge_features(**kwargs) -> None:
    ti = kwargs['ti']
    df = ti.xcom_pull(key='merged_df')
    products_df = ti.xcom_pull(key='products_df')
    clients_df = ti.xcom_pull(key='clients_df')
    df = df.merge(clients_df, on='client_id', how='left')
    df = df.merge(products_df, on='product_id', how='left')
    df.drop(columns=['category'], inplace=True, errors='ignore')
    ti.xcom_push(key='final_df', value=df)

def save_preprocessed_data(**kwargs) -> None:
    ti = kwargs['ti']
    base_dir = ti.xcom_pull(key='base_dir')
    df = ti.xcom_pull(key='final_df')
    transactions_df = ti.xcom_pull(key='transactions_df')
    df.to_parquet(
        f"{base_dir}/preprocessed/processed_data.parquet", 
        engine='pyarrow', 
        index=False
    )
    transactions_df.to_parquet(
        f"{base_dir}/preprocessed/processed_transactions.parquet", 
        engine='pyarrow', 
        index=False
    )
    
    del df, transactions_df
    gc.collect()
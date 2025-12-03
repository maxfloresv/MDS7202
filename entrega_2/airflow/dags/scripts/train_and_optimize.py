import pandas as pd
import optuna.visualization as opt_vis
from optuna.integration.lightgbm import LightGBMPruningCallback
import os
import gc
import optuna
from optuna.samplers import TPESampler
import joblib
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
  StandardScaler, 
  MinMaxScaler, 
  OneHotEncoder
)
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from lightgbm import LGBMClassifier, early_stopping, log_evaluation

RANDOM_STATE = 42
BASE_PATH = os.environ.get('AIRFLOW_HOME', '/opt/airflow')

def construct_model(**kwargs):
  """
  Constructs a LightGBM model with preprocessing and saves it.
  """
  ti = kwargs['ti']
  base_dir = ti.xcom_pull(key='base_dir')

  preprocessor_path = ti.xcom_pull(
    key='feature_engineering_transformer_path'
  )
  preprocessor = joblib.load(preprocessor_path)

  lgbm_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LGBMClassifier(
      objective='binary',
      random_state=RANDOM_STATE
    ))
  ])

  models_dir = f"{base_dir}/models"
  model_path_out = f"{models_dir}/lgbm_model.joblib"
  joblib.dump(lgbm_model, model_path_out)
  ti.xcom_push(key='model_path', value=model_path_out)

def save_optimization_study(**kwargs):
  """
  Saves the optimization study results with LightGBM model.
  """
  def objective_function(
    trial, 
    X_train, 
    y_train, 
    X_val, 
    y_val, 
    model,
    random_state=RANDOM_STATE
  ) -> float:
    """
    Objective function for Optuna hyperparameter optimization.

    Parameters
    ----------
    trial : optuna.trial.Trial
      An Optuna trial object.
    X_train : pd.DataFrame
      Training features.
    y_train : pd.Series
      Training labels.
    X_val : pd.DataFrame
      Validation features.
    y_val : pd.Series
      Validation labels.
    model : Pipeline
      LightGBM model with preprocessing.
    random_state : int, optional
      Random state for reproducibility, by default 42.

    Returns
    -------
    float
      The macro F1-score on the validation set.
    """
    params = {
      "classifier__num_leaves": trial.suggest_int("num_leaves", 20, 200),
      "classifier__max_depth": trial.suggest_int("max_depth", 3, 12),
      "classifier__learning_rate": trial.suggest_float("learning_rate", 0.001, 0.1, log=True),
      "classifier__n_estimators": trial.suggest_int("n_estimators", 100, 1000),
      "classifier__min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 10, 100),
      "classifier__feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
      "classifier__bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
      "classifier__bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
      "classifier__lambda_l1": trial.suggest_float("lambda_l1", 0.0, 10.0),
      "classifier__lambda_l2": trial.suggest_float("lambda_l2", 0.0, 10.0),
      "classifier__scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 10.0),
      "classifier__subsample": trial.suggest_float("subsample", 0.5, 1.0),
      "classifier__colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
    }

    params.update({
      "preprocessor__numerical__imputer__strategy": trial.suggest_categorical(
        "numerical_imputer_strategy", ["mean", "median"]
      ),
    })

    # Tries different choices for numerical scaler.
    scaler_choice = trial.suggest_categorical(
      "numerical_scaler", 
      ["minmax", "standard", "none"]
    )
    if scaler_choice == "minmax":
      params["preprocessor__numerical__scaler"] = MinMaxScaler()
    elif scaler_choice == "standard":
      params["preprocessor__numerical__scaler"] = StandardScaler()
    else:
      params["preprocessor__numerical__scaler"] = "passthrough"

    # Implements pruning callback to stop unpromising trials early.
    pruning_callback = LightGBMPruningCallback(
      trial, 
      metric='auc', 
      valid_name='valid_0' 
    )

    early_stopping_callback = early_stopping(
      stopping_rounds=100,
      first_metric_only=True
    )

    # Logs the evaluation metrics every 100 boosting stages.
    log_callback = log_evaluation(period=100)

    model.set_params(**params)

    preprocessor = model.named_steps['preprocessor']
    classifier = model.named_steps['classifier']

    X_train_prepared = preprocessor.fit_transform(X_train, y_train)
    X_val_prepared = preprocessor.transform(X_val)
    
    classifier.fit(
      X_train_prepared, 
      y_train, 
      eval_set=[(X_val_prepared, y_val)],
      eval_metric='auc',
      callbacks=[
        pruning_callback, 
        early_stopping_callback, 
        log_callback
      ]
    )

    y_pred = classifier.predict(X_val_prepared)
    macro_f1 = f1_score(y_val, y_pred, average='macro')

    return macro_f1

  ti = kwargs['ti']
  base_dir = ti.xcom_pull(key='base_dir')

  X_train_path = ti.xcom_pull(key='X_train_path')
  y_train_path = ti.xcom_pull(key='y_train_path')

  X_train = pd.read_parquet(X_train_path, engine='pyarrow')
  y_train = pd.read_parquet(y_train_path, engine='pyarrow').iloc[:, 0]

  X_val_path = ti.xcom_pull(key='X_val_path')
  y_val_path = ti.xcom_pull(key='y_val_path')

  X_val = pd.read_parquet(X_val_path, engine='pyarrow')
  y_val = pd.read_parquet(y_val_path, engine='pyarrow').iloc[:, 0]

  model_path = ti.xcom_pull(key='model_path')
  model = joblib.load(model_path)

  sampler = TPESampler(seed=RANDOM_STATE)
  pruner = optuna.pruners.MedianPruner(
    n_startup_trials=10, 
    # Has to be at least the same value as the internal early stopping rounds.
    n_warmup_steps=100,
    interval_steps=10
  )

  study = optuna.create_study(
    direction="maximize", 
    sampler=sampler, 
    pruner=pruner
  )

  study.optimize(
    lambda trial: objective_function(
      trial, 
      X_train, 
      y_train, 
      X_val, 
      y_val, 
      model
    ),
    n_trials=35
  )

  print("Best hyperparameters:")
  for k, v in study.best_params.items():
    print(f"  {k}: {v}")

  studies_dir = f"{base_dir}/studies"
  study_path_out = f"{studies_dir}/optimization_study.pkl"
  joblib.dump(study, study_path_out)
  ti.xcom_push(key='study_path', value=study_path_out)

def generate_optuna_plots(**kwargs):
  """
  Generates and saves the Optuna optimization history, parallel coordinates, 
  and hyperparameter importances plots as interactive HTML files.
  """
  ti = kwargs['ti']
  base_dir = ti.xcom_pull(key='base_dir')
  study_path = ti.xcom_pull(key='study_path')
  study = joblib.load(study_path)
  
  images_dir = f"{base_dir}/images"

  try:
    fig_history = opt_vis.plot_optimization_history(study)
    path_history = os.path.join(images_dir, "optuna_history.html")
    fig_history.write_html(path_history)
    ti.xcom_push(key='optuna_history_plot_path', value=path_history)
  except Exception as e:
    print(f"Could not generate the optimization history plot: {e}")
      
  try:
    fig_parallel = opt_vis.plot_parallel_coordinate(study)
    path_parallel = os.path.join(images_dir, "optuna_parallel.html")
    fig_parallel.write_html(path_parallel)
    ti.xcom_push(key='optuna_parallel_plot_path', value=path_parallel)
  except Exception as e:
    print(f"Could not generate the parallel coordinates plot: {e}")
      
  try:
    fig_importances = opt_vis.plot_param_importances(study)
    path_importances = os.path.join(images_dir, "optuna_importances.html")
    fig_importances.write_html(path_importances)
    ti.xcom_push(key='optuna_importances_plot_path', value=path_importances)
  except Exception as e:
    print(f"Could not generate the parameter importances plot: {e}")

def setup_optimized_model(**kwargs):
  """
  Sets up the optimized model with the best parameters.
  """
  ti = kwargs['ti']
  base_dir = ti.xcom_pull(key='base_dir')

  numerical_features = ti.xcom_pull(key='numerical_features')
  categorical_features = ti.xcom_pull(key='categorical_features')

  X_train_path = ti.xcom_pull(key='X_train_path')
  y_train_path = ti.xcom_pull(key='y_train_path')

  X_train = pd.read_parquet(X_train_path, engine='pyarrow')
  y_train = pd.read_parquet(y_train_path, engine='pyarrow').iloc[:, 0]

  X_val_path = ti.xcom_pull(key='X_val_path')
  y_val_path = ti.xcom_pull(key='y_val_path')

  X_val = pd.read_parquet(X_val_path, engine='pyarrow')
  y_val = pd.read_parquet(y_val_path, engine='pyarrow').iloc[:, 0]

  # Uses all available data to fit the preprocessor (hyperparameters were already optimized).
  X_fusion = pd.concat([X_train, X_val], ignore_index=True)
  y_fusion = pd.concat([y_train, y_val], ignore_index=True)

  preprocessed_dir = f"{base_dir}/preprocessed"
  X_fusion_path_out = f"{preprocessed_dir}/X_fusion.parquet"
  X_fusion.to_parquet(X_fusion_path_out, engine='pyarrow', index=False)
  ti.xcom_push(key='X_fusion_path', value=X_fusion_path_out)

  y_fusion_path_out = f"{preprocessed_dir}/y_fusion.parquet"
  y_fusion.to_frame(name='label').to_parquet(
    y_fusion_path_out, 
    engine='pyarrow', 
    index=False
  )
  ti.xcom_push(key='y_fusion_path', value=y_fusion_path_out)

  del X_train, y_train, X_val, y_val
  gc.collect()

  study_path = ti.xcom_pull(key='study_path')
  study = joblib.load(study_path)
  best_params = study.best_trial.params

  numerical_imputer_strategy = best_params.get(
    "numerical_imputer_strategy", 
    "median"
  )
  numerical_scaler_choice = best_params.get(
    "numerical_scaler", 
    "standard"
  )

  if numerical_scaler_choice == "minmax":
    numerical_scaler = MinMaxScaler()
  elif numerical_scaler_choice == "standard":
    numerical_scaler = StandardScaler()
  else:
    numerical_scaler = "passthrough"

  numerical_pipe = Pipeline(steps=[
    ('imputer', SimpleImputer(
      strategy=numerical_imputer_strategy
    )),
    ('scaler', numerical_scaler)
  ])

  categorical_pipe = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown="ignore"))
  ])

  preprocessor = ColumnTransformer(
    transformers=[
      ('numerical', numerical_pipe, numerical_features),
      ('categorical', categorical_pipe, categorical_features)
    ]
  )

  optimized_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LGBMClassifier(
      objective='binary', 
      n_jobs=-1,
      random_state=RANDOM_STATE
    ))
  ])

  # It's necessary to prefix the classifier parameters with 'classifier__'.
  best_classifier_params = {}
  for key, value in best_params.items():
    if not key.startswith("classifier__") and key in [
      "num_leaves", "max_depth", "learning_rate", "n_estimators",
      "min_data_in_leaf", "feature_fraction", "bagging_fraction",
      "bagging_freq", "lambda_l1", "lambda_l2", "scale_pos_weight",
      "subsample", "colsample_bytree"
    ]:
      best_classifier_params[f"classifier__{key}"] = value

  optimized_model.set_params(**best_classifier_params)

  preprocessor = optimized_model.named_steps['preprocessor']
  classifier = optimized_model.named_steps['classifier']

  X_fusion_prepared = preprocessor.fit_transform(X_fusion, y_fusion)

  log_callback = log_evaluation(period=100)  
  classifier.fit(
    X_fusion_prepared,
    y_fusion,
    callbacks=[log_callback]
  )

  models_dir = f"{base_dir}/models"
  optimized_trained_model_path_out = f"{models_dir}/optimized_trained_model.joblib"
  joblib.dump(optimized_model, optimized_trained_model_path_out)
  ti.xcom_push(
    key='optimized_trained_model_path', 
    value=optimized_trained_model_path_out
  )
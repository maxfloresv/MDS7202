import os
import gc
import optuna
import joblib
import pandas as pd
import numpy as np
import optuna.visualization as opt_vis
from optuna.samplers import TPESampler
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
  StandardScaler, 
  MinMaxScaler, 
  OneHotEncoder
)
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import TimeSeriesSplit
from lightgbm import (
  LGBMClassifier, 
  early_stopping, 
  log_evaluation
)

RANDOM_STATE = 42
BASE_PATH = os.environ.get('AIRFLOW_HOME', '/opt/airflow')

def construct_model_template(**kwargs):
  """
  Constructs a LightGBM model template with preprocessing and saves it.
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
      random_state=RANDOM_STATE,
      n_jobs=-1
    ))
  ])

  models_dir = f"{base_dir}/models"
  model_path_out = f"{models_dir}/lgbm_model_template.joblib"
  joblib.dump(lgbm_model, model_path_out)
  ti.xcom_push(key='model_template_path', value=model_path_out)

def save_optimization_study(**kwargs):
  """
  Saves the optimization study results with LightGBM model.
  """
  def objective_function(
    trial, 
    X_full,
    y_full,
    model_template,
    cv_splits=3,
    random_state=RANDOM_STATE
  ) -> float:
    """
    Objective function for Optuna hyperparameter optimization.

    Parameters
    ----------
    trial : optuna.trial.Trial
      An Optuna trial object.
    X_full : pd.DataFrame
      Full dataset features.
    y_full : pd.Series
      Full dataset labels.
    model_template : Pipeline
      LightGBM model template with preprocessing.
    cv_splits : int, optional
      Number of cross-validation splits, by default 3.
    random_state : int, optional
      Random state for reproducibility, by default 42.

    Returns
    -------
    float
      The mean macro F1-score on the cross-validation sets.
    """
    params = {
      "classifier__num_leaves": trial.suggest_int("num_leaves", 20, 200),
      "classifier__max_depth": trial.suggest_int("max_depth", 3, 12),
      "classifier__learning_rate": trial.suggest_float("learning_rate", 0.001, 0.1, log=True),
      "classifier__n_estimators": trial.suggest_int("n_estimators", 100, 1000),
      "classifier__min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 50, 500),
      "classifier__feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
      "classifier__bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
      "classifier__bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
      "classifier__lambda_l1": trial.suggest_float("lambda_l1", 0.0, 10.0),
      "classifier__lambda_l2": trial.suggest_float("lambda_l2", 0.0, 10.0),
      "classifier__scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 8.0),
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

    model_template.set_params(**params)
    
    tscv = TimeSeriesSplit(n_splits=cv_splits)
    f1_scores = []

    # Implements Rolling Window Cross-Validation (no critical periods are excluded).
    for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X_full)):
      X_fold_train, y_fold_train = X_full.iloc[train_idx], y_full.iloc[train_idx]
      X_fold_val, y_fold_val = X_full.iloc[val_idx], y_full.iloc[val_idx]

      preprocessor = model_template.named_steps['preprocessor']
      classifier = model_template.named_steps['classifier']

      X_fold_train_prepared = preprocessor.fit_transform(X_fold_train, y_fold_train)
      X_fold_val_prepared = preprocessor.transform(X_fold_val)

      early_stopping_callback = early_stopping(
        stopping_rounds=100,
        first_metric_only=True
      )

      log_callback = log_evaluation(period=0)

      classifier.fit(
        X_fold_train_prepared,
        y_fold_train,
        eval_set=[(X_fold_val_prepared, y_fold_val)],
        eval_metric='auc',
        callbacks=[
          early_stopping_callback,
          log_callback
        ]
      )

      y_pred_val = classifier.predict(X_fold_val_prepared)
      # Uses binary F1-score to avoid class imbalance issues.
      fold_f1 = f1_score(y_fold_val, y_pred_val, average='binary')
      f1_scores.append(fold_f1)

      del (
        X_fold_train, 
        y_fold_train, 
        X_fold_val, 
        y_fold_val, 
        X_fold_train_prepared, 
        X_fold_val_prepared
      )
      gc.collect()

    return np.mean(f1_scores)

  ti = kwargs['ti']
  base_dir = ti.xcom_pull(key='base_dir')

  X_full_path = ti.xcom_pull(key='X_full_path')
  y_full_path = ti.xcom_pull(key='y_full_path')

  X_full = pd.read_parquet(X_full_path, engine='pyarrow')
  y_full = pd.read_parquet(y_full_path, engine='pyarrow').iloc[:, 0]

  model_template_path = ti.xcom_pull(key='model_template_path')
  model_template = joblib.load(model_template_path)

  sampler = TPESampler(seed=RANDOM_STATE)
  study = optuna.create_study(direction="maximize", sampler=sampler)

  study.optimize(
    lambda trial: objective_function(
      trial, 
      X_full, 
      y_full, 
      model_template
    ),
    n_trials=50
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
  Refits the model template with the best parameters.
  """
  ti = kwargs['ti']
  base_dir = ti.xcom_pull(key='base_dir')

  X_full_path = ti.xcom_pull(key='X_full_path')
  y_full_path = ti.xcom_pull(key='y_full_path')

  X_full = pd.read_parquet(X_full_path, engine='pyarrow')
  y_full = pd.read_parquet(y_full_path, engine='pyarrow').iloc[:, 0]

  study_path = ti.xcom_pull(key='study_path')
  study = joblib.load(study_path)
  best_params = study.best_trial.params

  numerical_features = ti.xcom_pull(key='numerical_features')
  categorical_features = ti.xcom_pull(key='categorical_features')

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

  log_callback = log_evaluation(period=100)
  optimized_model.fit(
    X_full, 
    y_full,
    classifier__callbacks=[log_callback]
  )

  models_dir = f"{base_dir}/models"
  optimized_trained_model_path_out = f"{models_dir}/optimized_trained_model.joblib"
  joblib.dump(optimized_model, optimized_trained_model_path_out)
  ti.xcom_push(
    key='optimized_trained_model_path', 
    value=optimized_trained_model_path_out
  )
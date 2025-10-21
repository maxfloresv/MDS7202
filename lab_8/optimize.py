import pandas as pd
import mlflow
import optuna
import os
import pickle
import sklearn
import plotly
import plotly.express as px

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

import xgboost as xgb
from xgboost import XGBClassifier

RANDOM_STATE = 42

df = pd.read_csv("water_potability.csv")
X = df.drop("Potability", axis=1)
y = df["Potability"]

X_train, X_test, y_train, y_test = train_test_split(
  X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

print(X_train.columns)

def get_best_model(experiment_id) -> Pipeline:
  """ 
  Gets the best model from MLflow experiment based on validation F1-score.

  Parameters
  ----------
  experiment_id : str
    The MLflow experiment ID.

  Returns
  -------
  best_model : sklearn.pipeline.Pipeline
    The best performing model.
  """
  runs = mlflow.search_runs(experiment_id)
  best_model_id = runs.sort_values("metrics.valid_f1")["run_id"].iloc[0]
  best_model = mlflow.sklearn.load_model("runs:/" + best_model_id + "/model")

  return best_model

def optimize_model():
  """
  Optimize the XGBoost model using Optuna and log results with MLflow.
  """
  experiment_name = "Water_Potability_Hyperparameter_Optimization"
  experiment = mlflow.get_experiment_by_name(experiment_name)
  if experiment is None:
    experiment_id = mlflow.create_experiment(experiment_name)
  else:
    experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
  mlflow.set_experiment(experiment_name)
  mlflow.autolog()

  def objective(trial) -> float:
    """
    Objective function for Optuna optimization.

    Parameters
    ----------
    trial : optuna.trial.Trial
      A trial object for suggesting hyperparameters.

    Returns
    -------
    float
      The F1-score of the model on the validation set.
    """
    params = {
      'learning_rate': trial.suggest_float("learning_rate", 0.001, 0.1, log=True),
      'n_estimators': trial.suggest_int("n_estimators", 50, 1000),
      'max_depth': trial.suggest_int("max_depth", 3, 10),
      'max_leaves': trial.suggest_int("max_leaves", 0, 100),
      'min_child_weight': trial.suggest_int("min_child_weight", 1, 5),
      'reg_alpha': trial.suggest_float("reg_alpha", 0.0, 1.0),
      'reg_lambda': trial.suggest_float("reg_lambda", 0.0, 1.0),
      'min_frequency': trial.suggest_float("min_frequency", 0.0, 1.0)
    }

    run_name = f"XGBoost_trial_{trial.number}"
    with mlflow.start_run(run_name=run_name, experiment_id=experiment_id):
      impute_col = ['ph', 'Sulfate', 'Trihalomethanes']

      ct = ColumnTransformer(transformers=[
        ('imputer', SimpleImputer(strategy='mean'), impute_col)
      ], remainder='passthrough')

      pipe = Pipeline(steps=[
        ('preprocessor', ct),
        ('model', XGBClassifier(
          **params, 
          use_label_encoder=False, 
          eval_metric='logloss'
        ))
      ])
      
      pipe.fit(X_train, y_train)
      y_pred = pipe.predict(X_test)
      f1 = f1_score(y_test, y_pred)

      mlflow.log_params(params)
      mlflow.log_metric("valid_f1", f1)
      mlflow.sklearn.log_model(pipe, "model")

    return f1

  study = optuna.create_study(direction="maximize")
  study.optimize(objective, n_trials=15)

  os.makedirs("plots", exist_ok=True)
  with mlflow.start_run(experiment_id=experiment_id, run_name="Optimization_Plots"):
    optimization_history = optuna.visualization.plot_optimization_history(study)
    optimization_history.write_html("plots/optimization_history.html")

    parallel_coordinate = optuna.visualization.plot_parallel_coordinate(study)
    parallel_coordinate.write_html("plots/parallel_coordinate.html")

    param_importances = optuna.visualization.plot_param_importances(study)
    param_importances.write_html("plots/param_importances.html")

    mlflow.log_artifact("plots/optimization_history.html")
    mlflow.log_artifact("plots/parallel_coordinate.html")
    mlflow.log_artifact("plots/param_importances.html")

  best_model = get_best_model(experiment_id)
  os.makedirs("models", exist_ok=True)
  with open("models/best_model.pkl", "wb") as f:
    pickle.dump(best_model, f)

  importance = best_model.named_steps['model'].get_booster().get_score(importance_type='gain')
  importance_df = pd.DataFrame({
    'feature': list(importance.keys()),
    'importance': list(importance.values())
  }).sort_values(by='importance', ascending=False)

  fig = px.bar(
    importance_df,
    x='importance',
    y='feature',
    orientation='h',
    title='Importancia de variables del mejor modelo XGBoost',
    color='importance',
    color_continuous_scale='Blues'
  )
  fig.update_layout(yaxis=dict(autorange='reversed'))

  plot_path = "plots/feature_importance.html"
  fig.write_html(plot_path)

  with mlflow.start_run(run_name="xgb_best_model"):
    mlflow.log_params(best_model.get_params())
    mlflow.sklearn.log_model(best_model, name="model")
    mlflow.log_artifact(plot_path)
    mlflow.log_dict(
      {
        "pandas": pd.__version__,
        "mlflow": mlflow.__version__,
        "optuna": optuna.__version__,
        "scikit-learn": sklearn.__version__,
        "plotly": plotly.__version__,
        "xgboost": xgb.__version__,
        "python": os.sys.version.split()[0]
      },
      "library_versions.json"
    )

  return best_model

if __name__ == "__main__":
  best_model = optimize_model()
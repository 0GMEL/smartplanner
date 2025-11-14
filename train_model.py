import mlflow
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from data_loader import *

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_train_data():
    """Загрузка подготовленных данных для обучения и валидации."""
    X_train = pd.read_csv('datatt/processed/X_train.csv')
    X_valid = pd.read_csv('datatt/processed/X_valid.csv')
    y_duration_train = pd.read_csv('datatt/processed/y_duration_train.csv').squeeze()
    y_duration_valid = pd.read_csv('datatt/processed/y_duration_valid.csv').squeeze()
    y_fatigue_train = pd.read_csv('datatt/processed/y_fatigue_train.csv').squeeze()
    y_fatigue_valid = pd.read_csv('datatt/processed/y_fatigue_valid.csv').squeeze()
    return X_train, X_valid, y_duration_train, y_duration_valid, y_fatigue_train, y_fatigue_valid


def root_mean_squared_error(y_true, y_pred):
    """
    Надёжно вычисляет RMSE без использования параметра `squared`, чтобы
    работать со старыми версиями scikit-learn.
    """
    mse = mean_squared_error(y_true, y_pred)
    return float(np.sqrt(mse))


def train_and_log_models():
    """Обучение моделей с параметрами из YAML и логирование в MLflow."""
    # Настройка MLflow: рекомендуется использовать sqlite или другой DB backend
    mlflow.set_experiment('Task Regression Models')

    X_train, X_valid, y_duration_train, y_duration_valid, y_fatigue_train, y_fatigue_valid = load_train_data()

    # Загрузка параметров из YAML
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)

    for target, y_train, y_valid in [('duration', y_duration_train, y_duration_valid),
                                     ('fatigue', y_fatigue_train, y_fatigue_valid)]:
        logging.info(f"--- Обучение моделей для: {target} ---")

        # ---------------- LightGBM ----------------
        with mlflow.start_run(run_name=f"LGBM_{target}"):
            logging.info(f"Обучение LightGBM для {target}")
            lgb_params = params.get('lgbm', {}).copy()

            train_data = lgb.Dataset(X_train, label=y_train)
            valid_data = lgb.Dataset(X_valid, label=y_valid, reference=train_data)

            model = lgb.train(
                lgb_params,
                train_data,
                valid_sets=[valid_data],
                num_boost_round=lgb_params.get('num_boost_round', 1000),
                callbacks=[lgb.early_stopping(stopping_rounds=lgb_params.get('early_stopping_rounds', 50))]
            )

            preds = model.predict(X_valid)
            rmse = root_mean_squared_error(y_valid, preds)
            mlflow.log_params(lgb_params)
            mlflow.log_metric("rmse", rmse)
            logging.info(f"LightGBM RMSE: {rmse:.4f}")

            # Сохраняем модель
            os.makedirs('mods', exist_ok=True)
            model_path = os.path.join('mods', f"lgbm_{target}.pkl")
            # Для lightgbm Booster joblib.dump обычно работает, но можно сохранять через model.save_model тоже
            joblib.dump(model, model_path)
            mlflow.log_artifact(model_path)
            logging.info(f"LightGBM {target} модель сохранена: {model_path}")

        # ---------------- XGBoost ----------------
        with mlflow.start_run(run_name=f"XGB_{target}"):
            logging.info(f"Обучение XGBoost для {target}")
            xgb_params = params.get('xgb', {}).copy()

            dtrain = xgb.DMatrix(X_train, label=y_train)
            dvalid = xgb.DMatrix(X_valid, label=y_valid)

            model = xgb.train(
                xgb_params,
                dtrain,
                num_boost_round=xgb_params.get('num_boost_round', 1000),
                evals=[(dvalid, 'validation')],
                early_stopping_rounds=xgb_params.get('early_stopping_rounds', 50)
            )

            preds = model.predict(dvalid)
            rmse = root_mean_squared_error(y_valid, preds)
            mlflow.log_params(xgb_params)
            mlflow.log_metric("rmse", rmse)
            logging.info(f"XGBoost RMSE: {rmse:.4f}")

            # Сохраняем модель
            model_path = os.path.join('mods', f"xgb_{target}.json")
            model.save_model(model_path)
            mlflow.log_artifact(model_path)
            logging.info(f"XGBoost {target} модель сохранена: {model_path}")


if __name__ == "__main__":
    train_and_log_models()
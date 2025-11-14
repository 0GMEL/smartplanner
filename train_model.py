import joblib
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.metrics import root_mean_squared_error
from data_loader import *
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_train_data():
    X_train = pd.read_csv('data/processed/X_train.csv')
    X_valid = pd.read_csv('data/processed/X_valid.csv')
    y_duration_train = pd.read_csv('data/processed/y_duration_train.csv').squeeze()
    y_duration_valid = pd.read_csv('data/processed/y_duration_valid.csv').squeeze()
    y_fatigue_train = pd.read_csv('data/processed/y_fatigue_train.csv').squeeze()
    y_fatigue_valid = pd.read_csv('data/processed/y_fatigue_valid.csv').squeeze()
    return X_train, X_valid, y_duration_train, y_duration_valid, y_fatigue_train, y_fatigue_valid


def train_and_save_models():
    logging.info("Загружаем данные...")

    X_train, X_valid, y_duration_train, y_duration_valid, y_fatigue_train, y_fatigue_valid = load_train_data()

    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)

    os.makedirs("mods", exist_ok=True)

    tasks = [
        ("duration", y_duration_train, y_duration_valid),
        ("fatigue", y_fatigue_train, y_fatigue_valid)
    ]

    for target, y_train, y_valid in tasks:
        logging.info(f"\n=== Обучение моделей для {target} ===")

        # ------------------------------------------------------
        #   LightGBM
        # ------------------------------------------------------
        logging.info(f"[LGBM] Обучение модели для {target}")
        lgb_params = params.get('lgbm', {}).copy()
        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_valid, label=y_valid, reference=train_data)

        lgb_model = lgb.train(
            params=lgb_params,
            train_set=train_data,
            valid_sets=[valid_data],
            num_boost_round=lgb_params.get('num_boost_round', 1000),
            callbacks=[lgb.early_stopping(stopping_rounds=lgb_params.get('early_stopping_rounds', 50))]
        )
        preds_lgb = lgb_model.predict(X_valid)
        rmse_lgb = root_mean_squared_error(y_valid, preds_lgb)
        logging.info(f"[LGBM] RMSE ({target}): {rmse_lgb:.4f}")
        joblib.dump(lgb_model, os.path.join("mods", f"lgbm_{target}.pkl"))

        # ------------------------------------------------------
        #   XGBoost
        # ------------------------------------------------------
        logging.info(f"[XGB] Обучение модели для {target}")
        xgb_params = params.get('xgb', {}).copy()
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dvalid = xgb.DMatrix(X_valid, label=y_valid)

        xgb_model = xgb.train(
            params=xgb_params,
            dtrain=dtrain,
            evals=[(dvalid, 'validation')],
            num_boost_round=xgb_params.get('num_boost_round', 1000),
            early_stopping_rounds=xgb_params.get('early_stopping_rounds', 50)
        )
        preds_xgb = xgb_model.predict(dvalid)
        rmse_xgb = root_mean_squared_error(y_valid, preds_xgb)
        logging.info(f"[XGB] RMSE ({target}): {rmse_xgb:.4f}")
        xgb_model.save_model(os.path.join("mods", f"xgb_{target}.json"))

        # ------------------------------------------------------
        #   CatBoost
        # ------------------------------------------------------
        logging.info(f"[CAT] Обучение модели для {target}")
        cat_params = params.get('catboost', {}).copy()
        cat_model = CatBoostRegressor(**cat_params, verbose=100)
        cat_model.fit(X_train, y_train, eval_set=(X_valid, y_valid))
        preds_cat = cat_model.predict(X_valid)
        rmse_cat = root_mean_squared_error(y_valid, preds_cat)
        logging.info(f"[CAT] RMSE ({target}): {rmse_cat:.4f}")
        cat_model.save_model(os.path.join("mods", f"catboost_{target}.cbm"))


if __name__ == "__main__":
    train_and_save_models()

import os
import json
import time
import joblib
import logging
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
from catboost import CatBoostRegressor

from optimizer import optimize_with_splitting

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def _safe_serialize(obj):
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {k: _safe_serialize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_safe_serialize(v) for v in obj]
    return obj


class Predictor:
    TRAIN_FEATURE_COLUMNS = [
        "planned_duration", "day_of_week", "category", "focus_level", "priority", "productivity",
        "year", "month", "day", "hour", "minute",
        "month_sin", "month_cos", "hour_sin", "hour_cos"
    ]

    def __init__(self,
                 duration_model_path="mods/catboost_duration.cbm",
                 fatigue_model_path="mods/catboost_fatigue.cbm",
                 encoder_path="ddm/ordinal_encoder.joblib"):
        self.duration_model = None
        self.fatigue_model = None
        self.encoder = None

        try:
            if os.path.exists(duration_model_path):
                self.duration_model = CatBoostRegressor()
                self.duration_model.load_model(duration_model_path)
                logging.info(f"Loaded CatBoost duration model: {duration_model_path}")
            else:
                logging.warning(f"Duration model not found at {duration_model_path}. Using fallback.")
        except Exception as e:
            logging.exception(f"Failed to load duration model: {e}")

        try:
            if os.path.exists(fatigue_model_path):
                self.fatigue_model = CatBoostRegressor()
                self.fatigue_model.load_model(fatigue_model_path)
                logging.info(f"Loaded CatBoost fatigue model: {fatigue_model_path}")
            else:
                logging.warning(f"Fatigue model not found at {fatigue_model_path}. Using fallback.")
        except Exception as e:
            logging.exception(f"Failed to load fatigue model: {e}")

        try:
            if os.path.exists(encoder_path):
                self.encoder = joblib.load(encoder_path)
                logging.info(f"Loaded encoder: {encoder_path}")
            else:
                logging.warning(f"Encoder not found at {encoder_path}. Using hash fallback.")
        except Exception as e:
            logging.exception(f"Failed to load encoder: {e}")

    def _simple_category_encode(self, cat):
        if pd.isna(cat):
            return -1.0
        return float(abs(hash(str(cat))) % 1000)

    def _make_feature_row(self, task_row: dict):
        planned = float(task_row.get("planned_duration", 1.0))
        dl = pd.to_datetime(task_row.get("deadline", datetime.now()))
        priority = int(task_row.get("priority", 3))
        category_raw = task_row.get("category", None)

        year = dl.year
        month = dl.month
        day = dl.day
        hour = dl.hour
        minute = dl.minute
        day_of_week = dl.weekday()

        month_sin = np.sin(2 * np.pi * month / 12)
        month_cos = np.cos(2 * np.pi * month / 12)
        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)

        focus = task_row.get("focus_level")
        if focus is None:
            if 6 <= hour < 12:
                focus = 7.5
            elif 12 <= hour < 18:
                focus = 6.5
            else:
                focus = 5.0

        if task_row.get("productivity") is not None:
            productivity = float(task_row.get("productivity"))
        else:
            productivity = float(np.clip((focus / 10.0) * (1.0 + (5 - abs(priority - 3)) * 0.02), 0.01, 1.0))

        if self.encoder:
            try:
                cat_enc = float(self.encoder.transform([[category_raw]])[0][0])
            except Exception:
                cat_enc = self._simple_category_encode(category_raw)
        else:
            cat_enc = self._simple_category_encode(category_raw)

        row = {
            "planned_duration": planned,
            "day_of_week": day_of_week,
            "category": cat_enc,
            "focus_level": focus,
            "priority": priority,
            "productivity": productivity,
            "year": year,
            "month": month,
            "day": day,
            "hour": hour,
            "minute": minute,
            "month_sin": month_sin,
            "month_cos": month_cos,
            "hour_sin": hour_sin,
            "hour_cos": hour_cos
        }

        return pd.DataFrame([row], columns=self.TRAIN_FEATURE_COLUMNS)

    def predict(self, task_row: dict):
        X = self._make_feature_row(task_row)

        pred_dur = float(task_row.get("planned_duration", 1.0))
        if self.duration_model:
            try:
                pred_dur = float(self.duration_model.predict(X)[0])
            except Exception as e:
                logging.exception("Duration CatBoost predict error, using fallback: %s", e)

        pred_fat = None
        if self.fatigue_model:
            try:
                pred_fat = float(np.clip(self.fatigue_model.predict(X)[0], 0.0, 10.0))
            except Exception as e:
                logging.exception("Fatigue CatBoost predict error, using heuristic: %s", e)

        if pred_fat is None:
            tod = X.iloc[0]["hour"] + X.iloc[0]["minute"] / 60.0
            base = 3.0 if 6 <= tod < 12 else 5.0 if 12 <= tod < 18 else 7.0
            prod = X.iloc[0]["productivity"]
            pred_fat = float(np.clip(base * (1.0 - (prod - 0.5) * 0.5), 0.0, 10.0))

        return pred_dur, pred_fat


class Planner:
    def __init__(self, predictor: Predictor = None, events_log_path="data/raw/events.csv"):
        self.predictor = predictor or Predictor()
        self.events_log_path = events_log_path
        os.makedirs(os.path.dirname(self.events_log_path) or ".", exist_ok=True)
        if not os.path.exists(self.events_log_path):
            pd.DataFrame(columns=[
                "timestamp", "event_type", "user_id", "task_id", "payload_json"
            ]).to_csv(self.events_log_path, index=False)

    def log_event(self, event_type: str, user_id: str, task_id: str, payload: dict):
        safe_payload = _safe_serialize(payload)
        rec = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "user_id": user_id,
            "task_id": task_id,
            "payload_json": json.dumps(safe_payload, ensure_ascii=False)
        }
        return pd.DataFrame([rec]).to_csv(self.events_log_path, mode="a", header=False, index=False)

    def prepare_tasks(self, user_tasks: list):
        rows = []
        for t in user_tasks:
            self.log_event("task_created", t.get("user_id", "U_unknown"), t.get("task_id", ""), t)

            pred_dur, pred_fat = self.predictor.predict(t)

            dl = pd.to_datetime(t.get("deadline", datetime.now()))
            slack_hours = max(0, (dl - datetime.now()).total_seconds() / 3600.0)

            rows.append({
                "user_id": t.get("user_id", "U_unknown"),
                "task_id": t.get("task_id", f"T_{int(time.time())}"),
                "planned_duration": float(t.get("planned_duration", pred_dur)),
                "pred_duration": float(pred_dur),
                "deadline": dl,
                "priority": int(t.get("priority", 3)),
                "focus_level": t.get("focus_level", None),
                "fatigue_level": float(pred_fat),
                "slack": slack_hours
            })

        return pd.DataFrame(rows)

    def plan(self,
             user_tasks: list,
             num_days=3,
             chunk_hours=2,
             spread_chunks_across_days=True,
             min_break_minutes=30,
             work_start_hour=9,
             work_end_hour=18,
             max_time_seconds=30):

        tasks_df = self.prepare_tasks(user_tasks)

        now = datetime.now()
        max_deadline = max(tasks_df['deadline'])
        remaining_hours = (max_deadline - now).total_seconds() / 3600.0
        available_days = max(1, int(np.ceil(remaining_hours / (work_end_hour - work_start_hour))))

        num_days_to_use = min(num_days, available_days)
        logging.info(f"Planning over {num_days_to_use} day(s) based on deadlines")

        res = optimize_with_splitting(
            tasks_df=tasks_df,
            num_days=num_days_to_use,
            work_start_hour=work_start_hour,
            work_end_hour=work_end_hour,
            chunk_hours=chunk_hours,
            min_break_minutes=min_break_minutes,
            spread_chunks_across_days=spread_chunks_across_days,
            max_time_seconds=max_time_seconds
        )

        if res is None:
            logging.error("Optimizer didn't find solution.")
            self.log_event("plan_failed", "system", "", {"reason": "no_solution"})
            return None

        for _, r in res["tasks"].iterrows():
            payload = {
                "task_id": r["task_id"],
                "task_start": r["task_start"].isoformat(),
                "task_end": r["task_end"].isoformat(),
                "priority": int(r["priority"])
            }
            self.log_event("schedule_assigned", r.get("task_id", "unknown"), r.get("task_id", "unknown"), payload)

        return {
            "chunks": res["chunks"].to_dict(orient="records"),
            "tasks": res["tasks"].to_dict(orient="records")
        }


if __name__ == "__main__":
    planner = Planner()

    tasks_one_day = [
        {
            "user_id": "U_test",
            "task_id": "LAB_001",
            "planned_duration": 6.0,
            "deadline": (datetime.now() + timedelta(hours=17)).isoformat(),
            "category": "Лабораторка",
            "priority": 3
        },
        {
            "user_id": "U_test",
            "task_id": "TASK_002",
            "planned_duration": 1.0,
            "deadline": (datetime.now() + timedelta(hours=17)).isoformat(),
            "category": "Закрыть проект на работе",
            "priority": 4
        },
        {
            "user_id": "U_test",
            "task_id": "TASK_003",
            "deadline": (datetime.now() + timedelta(hours=17)).isoformat(),
            "category": "Встреча",
            "priority": 5
        }
    ]

    schedule = planner.plan(
        tasks_one_day,
        num_days=1,
        chunk_hours=2,
        spread_chunks_across_days=False,
        min_break_minutes=60,
        work_start_hour=9,
        work_end_hour=20,
        max_time_seconds=20
    )
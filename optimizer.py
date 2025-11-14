from ortools.sat.python import cp_model
import pandas as pd
from datetime import datetime, timedelta
import math


def optimize_with_splitting(
        tasks_df,
        num_days=3,
        work_start_hour=9,
        work_end_hour=18,
        chunk_hours=2,
        min_break_minutes=30,
        spread_chunks_across_days=True,
        focus_weight_minutes=5,
        fatigue_weight_minutes=3,
        max_time_seconds=30
):
    tasks_df = tasks_df.reset_index(drop=True).copy()

    tasks_df['focus_level'] = tasks_df['focus_level'].fillna(5.0).astype(float)
    tasks_df['fatigue_level'] = tasks_df['fatigue_level'].fillna(5.0).astype(float)

    chunks = []
    for idx, row in tasks_df.iterrows():
        total_hours = float(row['planned_duration'])
        n_chunks = max(1, math.ceil(total_hours / chunk_hours))
        chunk_hours_list = [total_hours / n_chunks] * n_chunks
        for k, ch in enumerate(chunk_hours_list):
            chunks.append({
                'orig_index': idx,
                'task_id': row['task_id'],
                'chunk_index': k,
                'chunk_hours': ch,
                'priority': int(row['priority']),
                'deadline': pd.to_datetime(row['deadline']),
                'focus_level': float(row['focus_level']),
                'fatigue_level': float(row['fatigue_level'])
            })

    chunks_df = pd.DataFrame(chunks)
    n = len(chunks_df)
    if n == 0:
        return pd.DataFrame()

    day_minutes = (work_end_hour - work_start_hour) * 60
    horizon = num_days * day_minutes
    origin_day = datetime.now().replace(hour=work_start_hour, minute=0, second=0, microsecond=0)

    model = cp_model.CpModel()

    start = [model.NewIntVar(0, horizon - 1, f'start_{i}') for i in range(n)]
    end = [model.NewIntVar(0, horizon, f'end_{i}') for i in range(n)]
    interval = []
    day_var = [model.NewIntVar(0, num_days - 1, f'day_{i}') for i in range(n)]
    duration_min = [int(round(chunks_df.loc[i, 'chunk_hours'] * 60)) for i in range(n)]

    for i in range(n):
        interval.append(model.NewIntervalVar(start[i], duration_min[i], end[i], f'interval_{i}'))
        model.Add(end[i] == start[i] + duration_min[i])
        model.Add(start[i] >= day_var[i] * day_minutes)
        model.Add(start[i] <= day_var[i] * day_minutes + (day_minutes - duration_min[i]))
        model.Add(end[i] <= (day_var[i] + 1) * day_minutes)

    model.AddNoOverlap(interval)

    for orig_idx, group in chunks_df.groupby('orig_index').groups.items():
        indices = sorted(list(group), key=lambda x: chunks_df.loc[x, 'chunk_index'])
        for a, b in zip(indices, indices[1:]):
            model.Add(start[b] >= end[a] + min_break_minutes)
            if spread_chunks_across_days:
                model.Add(day_var[b] >= day_var[a] + 1)
            else:
                model.Add(day_var[b] >= day_var[a])

    orig_count = len(tasks_df)
    last_chunk_end = [None] * orig_count
    for orig_idx in range(orig_count):
        idxs = [i for i in range(n) if chunks_df.loc[i, 'orig_index'] == orig_idx]
        last_end_var = model.NewIntVar(0, horizon, f'last_end_{orig_idx}')
        for i in idxs:
            model.Add(last_end_var >= end[i])
        last_chunk_end[orig_idx] = last_end_var

    # Опоздания
    delay_vars = []
    for orig_idx in range(orig_count):
        dl_dt = pd.to_datetime(tasks_df.loc[orig_idx, 'deadline'])
        deadline_minutes = int(max(0, (dl_dt - origin_day).total_seconds() // 60))
        deadline_minutes = min(deadline_minutes, horizon)
        dvar = model.NewIntVar(0, horizon + 10000, f'delay_orig_{orig_idx}')
        model.Add(dvar >= last_chunk_end[orig_idx] - deadline_minutes)
        delay_vars.append(dvar)

    # Целевая функция
    obj_terms = []
    for orig_idx in range(orig_count):
        priority = int(tasks_df.loc[orig_idx, 'priority'])
        delay_term = priority * delay_vars[orig_idx]

        focus = float(tasks_df.loc[orig_idx, 'focus_level'])
        focus_term = int(round(focus * focus_weight_minutes))

        fatigue = float(tasks_df.loc[orig_idx, 'fatigue_level'])
        fatigue_term = int(round(fatigue * fatigue_weight_minutes))

        obj_terms.append(delay_term + fatigue_term - focus_term)

    model.Minimize(sum(obj_terms))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = max_time_seconds
    solver.parameters.num_search_workers = 8

    status = solver.Solve(model)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None

    schedule_chunks = []
    for i in range(n):
        s_min = solver.Value(start[i])
        e_min = solver.Value(end[i])
        day_idx = solver.Value(day_var[i])
        absolute_start = origin_day + timedelta(days=int(day_idx)) + timedelta(
            minutes=int(s_min - day_idx * day_minutes))
        absolute_end = origin_day + timedelta(days=int(day_idx)) + timedelta(minutes=int(e_min - day_idx * day_minutes))
        schedule_chunks.append({
            'orig_index': int(chunks_df.loc[i, 'orig_index']),
            'task_id': chunks_df.loc[i, 'task_id'],
            'chunk_index': int(chunks_df.loc[i, 'chunk_index']),
            'start_time': absolute_start,
            'end_time': absolute_end,
            'chunk_hours': chunks_df.loc[i, 'chunk_hours'],
            'priority': chunks_df.loc[i, 'priority'],
            'focus_level': chunks_df.loc[i, 'focus_level'],
            'fatigue_level': chunks_df.loc[i, 'fatigue_level'],
        })

    chunks_out = pd.DataFrame(schedule_chunks)

    agg = chunks_out.groupby('orig_index').agg({
        'task_id': 'first',
        'start_time': 'min',
        'end_time': 'max',
        'priority': 'first',
        'focus_level': 'first',
        'fatigue_level': 'first'
    }).reset_index()
    agg = agg.rename(columns={'start_time': 'task_start', 'end_time': 'task_end'})

    print("=== CHUNKS ===")
    for _, row in chunks_out.iterrows():
        print(
            f"{row['task_id']} chunk[{row['chunk_index']}] {row['start_time']} - {row['end_time']} ({row['chunk_hours']}h)")

    print("\n=== TASKS AGGREGATED ===")
    for _, row in agg.iterrows():
        print(f"{row['task_id']} {row['task_start']} -> {row['task_end']} (priority {row['priority']})")

    return {
        'chunks': chunks_out.sort_values(['start_time', 'task_id', 'chunk_index']).reset_index(drop=True),
        'tasks': agg
    }
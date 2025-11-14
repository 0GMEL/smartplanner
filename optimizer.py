# оптимизация расписания (OR-Tools)

from ortools.sat.python import cp_model
import pandas as pd
from datetime import datetime, timedelta

def optimize_schedule_ortools(tasks, day_start="08:00"):
    """
    tasks: список словарей с полями:
        - task_id
        - predicted_duration (float, в часах)
        - deadline (datetime)
        - priority (int)
        - fatigue_level (0-10)
        - focus_level (0-10)
    Возвращает DataFrame с оптимальным расписанием start/end времени.
    """

    # Переводим продолжительность в минуты
    tasks = [
        {**t, 'duration_min': int(t['predicted_duration']*60)}
        for t in tasks
    ]

    num_tasks = len(tasks)

    # Начало рабочего дня
    day_start_dt = datetime.strptime(day_start, "%H:%M")
    day_start_min = 0  # 0 минут от старта рабочего дня
    horizon = sum(t['duration_min'] for t in tasks) + 60*2  # буфер 2 часа

    model = cp_model.CpModel()

    # Переменные start и end (в минутах)
    starts = [model.NewIntVar(day_start_min, horizon, f'start_{i}') for i in range(num_tasks)]
    ends = [model.NewIntVar(day_start_min, horizon, f'end_{i}') for i in range(num_tasks)]
    durations = [t['duration_min'] for t in tasks]

    # Ограничения на продолжительность
    for i in range(num_tasks):
        model.Add(ends[i] == starts[i] + durations[i])

    # Ограничение: задачи не пересекаются
    for i in range(num_tasks):
        for j in range(i+1, num_tasks):
            model.AddNoOverlap([(starts[i], durations[i]), (starts[j], durations[j])])

    # Целевая функция (reward)
    rewards = []
    for i, task in enumerate(tasks):
        # 1. Приоритет: чем выше priority, тем лучше раннее начало
        reward_priority = task['priority'] * 10 - starts[i]  # ранний старт лучше

        # 2. Усталость: высокая fatigue → штраф
        reward_fatigue = max(0, 7 - task['fatigue_level'])*5

        # 3. Фокус: высокий focus → бонус
        reward_focus = task['focus_level']*2

        # 4. Дедлайн: штраф за просрочку
        deadline_min = int((task['deadline'] - datetime.combine(task['deadline'].date(), datetime.min.time())).total_seconds() // 60)
        reward_deadline = max(0, deadline_min - ends[i])  # если закончил до дедлайна → бонус

        total_reward = reward_priority + reward_fatigue + reward_focus + reward_deadline
        rewards.append(total_reward)

    # Максимизируем сумму reward
    model.Maximize(sum(rewards))

    # Решаем
    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    if status not in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        print("Не удалось найти решение")
        return None

    # Формируем DataFrame
    schedule = []
    for i, task in enumerate(tasks):
        start_min = solver.Value(starts[i])
        end_min = solver.Value(ends[i])
        start_time = day_start_dt + timedelta(minutes=start_min)
        end_time = day_start_dt + timedelta(minutes=end_min)

        schedule.append({
            'task_id': task['task_id'],
            'start_time': start_time,
            'end_time': end_time,
            'priority': task['priority'],
            'fatigue_level': task['fatigue_level'],
            'focus_level': task['focus_level'],
            'planned_duration': task['predicted_duration'],
            'deadline': task['deadline']
        })

    return pd.DataFrame(schedule)
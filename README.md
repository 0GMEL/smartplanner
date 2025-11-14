1. Клонировать проект

```bash
git clone <https://github.com/0GMEL/smartplanner.git>
cd smartplanner
```

2. Установить зависимости

```bash
pip install -r requirements.txt
```

3. Инициализация DVC

```bash
dvc init
```

4. Запуск обработки данных и обучения моделей

```bash
dvc repro
```

5. Запуск планировщика

```bash
python planner.py
```
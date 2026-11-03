# AI_Text_Detection
Training and evaluation for models detecting AI-generated texts

Дашборд експериментів «AI Detection Models».

Запуск (з кореня репозиторію):
    streamlit run dashboard/app.py

Шлях до логів: змінна середовища RUNS_ROOT, інакше <корінь репозиторію>/runs.
Структура даних:
    runs/<сценарій>/training_logs/<run>/{config.json, history.csv, group_metrics.csv, run_summary.json, training.log}
    runs/<сценарій>/test_results/<run>/{test_metrics.csv, confusion_matrix.csv, metrics_by_*.csv, test_predictions.csv}

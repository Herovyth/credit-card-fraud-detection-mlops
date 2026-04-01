"""
tests/test_dag.py — Перевірка коректності DAG-файлів (CI)
Лабораторна робота №5 | GitHub Actions

Запуск:
    pytest tests/test_dag.py -v
"""

from pathlib import Path
import os
import pytest

from airflow.models import DagBag

DAGS_FOLDER = str(Path(__file__).resolve().parents[1] / "dags")


def test_dag_import_no_errors():
    """DAG-файли завантажуються без помилок імпорту."""
    dag_bag = DagBag(dag_folder=DAGS_FOLDER, include_examples=False)
    assert len(dag_bag.import_errors) == 0, f"Помилки імпорту DAG:\n" + "\n".join(
        f"  {k}: {v}" for k, v in dag_bag.import_errors.items()
    )


def test_dag_exists():
    """DAG ml_training_pipeline присутній у DagBag."""
    os.environ["AIRFLOW__CORE__DAGBAG_IMPORT_TIMEOUT"] = "0"
    dag_bag = DagBag(dag_folder=DAGS_FOLDER, include_examples=False)
    assert (
        "ml_training_pipeline" in dag_bag.dags
    ), f"DAG 'ml_training_pipeline' не знайдено. Доступні: {list(dag_bag.dags.keys())}"


def test_dag_task_count():
    """DAG містить очікувану кількість задач."""
    dag_bag = DagBag(dag_folder=DAGS_FOLDER, include_examples=False)
    dag = dag_bag.dags["ml_training_pipeline"]
    assert (
        len(dag.tasks) >= 5
    ), f"Очікується мінімум 5 задач, знайдено: {len(dag.tasks)}"


def test_dag_required_tasks_present():
    """Всі ключові задачі присутні в DAG."""
    dag_bag = DagBag(dag_folder=DAGS_FOLDER, include_examples=False)
    dag = dag_bag.dags["ml_training_pipeline"]
    task_ids = {t.task_id for t in dag.tasks}

    required = {
        "check_data",
        "prepare_data",
        "train_model",
        "evaluate_model",
        "register_model",
        "notify_low_quality",
        "pipeline_end",
    }
    missing = required - task_ids
    assert not missing, f"Відсутні задачі в DAG: {missing}"


def test_dag_no_cycles():
    """DAG не містить циклів."""
    dag_bag = DagBag(dag_folder=DAGS_FOLDER, include_examples=False)
    dag = dag_bag.dags["ml_training_pipeline"]
    from airflow.utils.dag_cycle_tester import check_cycle

    try:
        check_cycle(dag)
        assert True
    except Exception as e:
        pytest.fail(f"DAG містить цикли: {e}")


def test_dag_task_dependencies():
    """Перевірка порядку залежностей між ключовими задачами."""
    dag_bag = DagBag(dag_folder=DAGS_FOLDER, include_examples=False)
    dag = dag_bag.dags["ml_training_pipeline"]

    def upstream_ids(task_id: str) -> set:
        return {t.task_id for t in dag.get_task(task_id).upstream_list}

    # prepare_data залежить від check_data
    assert "check_data" in upstream_ids("prepare_data")
    # train_model залежить від prepare_data
    assert "prepare_data" in upstream_ids("train_model")
    # evaluate_model залежить від train_model
    assert "train_model" in upstream_ids("evaluate_model")

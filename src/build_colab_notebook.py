"""Подготовка блокнота Google Colab без привязки к локальному окружению .venv."""
from copy import deepcopy
from pathlib import Path
import textwrap

import nbformat

ROOT=Path(__file__).resolve().parents[1]

BOOTSTRAP=textwrap.dedent('''\
    import importlib.util
    import os
    from pathlib import Path
    import subprocess
    import sys
    import tempfile
    import zipfile

    REPOSITORY = "https://github.com/alinaRN1/llm1.git"
    ROOT = Path(os.environ.get("LL1_PROJECT_ROOT", "/content/llm1"))
    archives = [Path("/content/LL1_masterat_ru.zip"), Path("/content/LL1_masterat.zip")]
    archive = next((p for p in archives if p.is_file()), None)

    if not (ROOT / "data/questions.json").is_file():
        ROOT.parent.mkdir(parents=True, exist_ok=True)
        if archive is not None:
            destination = Path(tempfile.mkdtemp(prefix="ll1_upload_", dir=ROOT.parent))
            with zipfile.ZipFile(archive) as bundle:
                for entry in bundle.infolist():
                    target = (destination / entry.filename).resolve()
                    if not target.is_relative_to(destination.resolve()):
                        raise ValueError("Архив содержит путь за пределами папки распаковки")
                bundle.extractall(destination)
            candidates = [destination, *[p for p in destination.iterdir() if p.is_dir()]]
            matches = [p for p in candidates if (p / "data/questions.json").is_file()]
            if len(matches) != 1:
                raise RuntimeError("В архиве не удалось однозначно найти проект лабораторной")
            ROOT = matches[0]
        else:
            if ROOT.exists() and any(ROOT.iterdir()):
                raise RuntimeError("Папка проекта занята другими файлами; выберите новую LL1_PROJECT_ROOT")
            result = subprocess.run(["git", "clone", "--depth", "1", REPOSITORY, str(ROOT)],
                                    capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError("Не удалось скачать репозиторий. Если он закрыт, загрузите "
                                   "LL1_masterat_ru.zip в панель «Файлы» Colab и повторите ячейку.")

    os.environ["LL1_PROJECT_ROOT"] = str(ROOT.resolve())
    os.chdir(ROOT)
    packages = {"numpy": "numpy", "pandas": "pandas", "matplotlib": "matplotlib"}
    missing = [package for module, package in packages.items()
               if importlib.util.find_spec(module) is None]
    if missing:
        subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", *missing], check=True)
    print("Проект подготовлен:", ROOT)
    print("Просмотр сохранённых результатов готов; скачивание весов не требуется.")
''')

NEW_EXPERIMENT=textwrap.dedent('''\
    RUN_NEW_EXPERIMENT = False

    if not RUN_NEW_EXPERIMENT:
        print("Показаны сохранённые результаты. Для нового запуска включите флаг в этой ячейке.")
    else:
        from datetime import datetime, timezone
        import shutil

        # Эта версия использует CPU, даже если в настройках Colab выбран GPU.
        environment = os.environ.copy()
        environment.update(CC="gcc", CXX="g++", CMAKE_BUILD_PARALLEL_LEVEL="2")
        subprocess.run([sys.executable, "-m", "pip", "install", "llama-cpp-python==0.3.16",
                        "psutil", "requests"], env=environment, check=True)
        subprocess.run([sys.executable, "src/pull_oci.py", "--locked", "data/models_oci.lock.json"],
                       check=True)
        RUN_ID = "colab-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        THREADS = min(2, os.cpu_count() or 1)
        subprocess.run([sys.executable, "src/benchmark_gguf.py", "--run-id", RUN_ID,
                        "--threads", str(THREADS)], check=True)
        NEW_RUN = ROOT / "results/runs" / RUN_ID
        subprocess.run([sys.executable, "src/analyze.py", str(NEW_RUN)], check=True)
        print("Новый эксперимент завершён:", NEW_RUN)
        print("Это новые измерения Colab. Проверку содержания и математических шагов нужно выполнить заново.")
        new_aggregate = pd.read_csv(NEW_RUN / "aggregate.csv")
        show_ru(new_aggregate[["model", "mode", "factual_exact", "math_accuracy",
                               "latency_median_ms", "decode_tps_median"]].to_dict("records"))
        display(Image(filename=str(NEW_RUN / "comparison.png"), width=1100))
        result_archive = shutil.make_archive(str(ROOT.parent / RUN_ID), "zip", root_dir=NEW_RUN)
        print("Архив новых результатов без весов:", result_archive)
        try:
            from google.colab import files
        except ImportError:
            pass
        else:
            files.download(result_archive)
''')


def build():
    original=nbformat.read(ROOT/'laborator.ipynb',as_version=4)
    cells=[nbformat.v4.new_markdown_cell(
        '# ЛЛ1 — запуск в Google Colab\n\n'
        'Выберите **среду Python 3, CPU**, затем **«Среда выполнения → Выполнить все»**. '
        'Начальная ячейка получает проект с GitHub. По умолчанию блокнот анализирует '
        'уже выполненный эксперимент из 360 ответов без загрузки моделей.\n\n'
        'Для нового эксперимента используйте последнюю ячейку. Все пояснения на русском. '
        'Если репозиторий закрыт, загрузите архив `LL1_masterat_ru.zip` через панель '
        '«Файлы» перед запуском первой ячейки.'),
        nbformat.v4.new_code_cell(BOOTSTRAP)]
    for original_cell in original.cells:
        cell=deepcopy(original_cell)
        if cell.cell_type=='markdown' and cell.source.startswith('## Повтор генерации'):
            continue
        if cell.cell_type=='code' and 'RUN_NEW_EXPERIMENT = False' in cell.source:
            continue
        if cell.cell_type=='code':
            cell.execution_count=None
            cell.outputs=[]
            cell.source=cell.source.replace('import json, sys, subprocess','import json, os, sys, subprocess')
            cell.source=cell.source.replace('ROOT = Path.cwd()',
                                            'ROOT = Path(os.environ["LL1_PROJECT_ROOT"])')
            cell.source=cell.source.replace('"Пройдено программных тестов:"',
                                            '"В исходном протоколе пройдено тестов:"')
        if cell.cell_type=='markdown' and cell.source.startswith('## Проверка программы'):
            cell.source=('## Сохранённый протокол проверки программы\n\n'
                         'Ниже читается протокол тестов при подготовке работы. '
                         'Это не новый запуск всех тестов в Colab. Тесты проверяют '
                         'оценщик и обработку данных, а не качество языковых моделей.')
        cells.append(cell)
    cells.extend([nbformat.v4.new_markdown_cell(
        '## Новый эксперимент в Colab — по необходимости\n\n'
        'Установите `RUN_NEW_EXPERIMENT = True` и выполните ячейку. Она установит '
        'CPU-реализацию llama.cpp, скачает около **4,2 ГБ весов**, получит **360 новых '
        'ответов** и скачает архив результатов. Сборка и генерация могут занять '
        'значительное время; оставьте сессию активной. Используется CPU с двумя '
        'потоками. Времена нового запуска относятся к аппаратуре Colab.\n\n'
        'Сохранённый отчёт остаётся отчётом исходного эксперимента. Новые ответы '
        'нуждаются в отдельной проверке содержания и шагов: предыдущие решения '
        'проверяющего допустимы только при совпадении модели, задания и хеша ответа. '
        'Локальные файлы Colab удаляются при сбросе сессии, поэтому сохраните архив.'),
        nbformat.v4.new_code_cell(NEW_EXPERIMENT)])
    notebook=nbformat.v4.new_notebook(cells=cells,metadata={
        'colab':{'name':'ЛЛ1 — Google Colab','provenance':[]},
        'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'},
        'language_info':{'name':'python'}})
    nbformat.validate(notebook)
    destination=ROOT/'laborator_colab.ipynb'
    nbformat.write(notebook,destination)
    print('Создан блокнот Google Colab:',destination)


if __name__=='__main__':
    build()

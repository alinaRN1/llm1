"""Создание и выполнение русского блокнота по измеренным и проверенным данным."""
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT=Path(__file__).resolve().parents[1]


def main():
    md=nbformat.v4.new_markdown_cell
    code=nbformat.v4.new_code_cell
    cells=[
        md('# ЛЛ1 — оценка языковых моделей с открытыми весами\n\n'
           'Исследование магистерского уровня. Здесь анализируются результаты собственного '
           'локального запуска: **360 ответов**, три модели, два режима и три начальных значения. '
           'Синтетические данные программных тестов в эти результаты не входят. '
           'Все пояснения — на русском; исходный эксперимент проведён на румынском.'),
        code('from pathlib import Path\nimport json, sys, subprocess\nimport pandas as pd\n'
             'from IPython.display import display, Image\n'
             'ROOT = Path.cwd()\nRUN = ROOT / "results/final"\n'
             'assert (ROOT / "data/questions.json").exists(), "Откройте блокнот из корня проекта"\n'
             'sys.path.insert(0, str(ROOT / "src"))\n'
             'from presentation import localize_records\n'
             'pd.set_option("display.max_colwidth", 100)\n'
             'def show_ru(records):\n    display(pd.DataFrame(localize_records(records)))\n'
             'print("Версия Python:", sys.version.split()[0])'),
        md('## Набор оценки\n\nЭталоны заданы до запуска и не передавались моделям. '
           'Для чтения ниже показан вспомогательный русский перевод; входами моделей были '
           'румынские оригиналы из `data/questions.json`.'),
        code('questions = json.loads((ROOT / "data/questions_ru.json").read_text())\n'
             'show_ru([{k:q[k] for k in ["id", "category", "question", "reference_answer"]} for q in questions])'),
        md('## Происхождение моделей\n\nGGUF F16 из Docker AI. Каждый манифест и файл проверен '
           'SHA-256. Регистр тега важен: SmolLM2 `360m-f16` содержит Instruct.'),
        code('specs = json.loads((ROOT / "data/models_oci.lock.json").read_text())\n'
             'show_ru([{\n'
             '    "model": m["model"], "manifest_digest": m["manifest_digest"],\n'
             '    "GGUF_bytes": next(x["size"] for x in m["layers"] if x["file"] == "model.gguf")\n'
             '} for m in specs])'),
        md('## Повтор генерации — по необходимости\n\n'
           'По умолчанию анализируются сохранённые реальные приложения. Флаг включается только '
           'для нового запуска, которому потребуется новая проверка, если ответы изменятся.'),
        code('RUN_NEW_EXPERIMENT = False\n'
             'if RUN_NEW_EXPERIMENT:\n'
             '    subprocess.run([sys.executable, "src/pull_oci.py", "--locked", "data/models_oci.lock.json"], check=True)\n'
             '    subprocess.run([sys.executable, "src/benchmark_gguf.py", "--run-id", "notebook-new-run"], check=True)\n'
             '    print("Новый запуск сохранён отдельно; до итогового сравнения выполните проверку ответов.")\n'
             'else:\n    print("Анализ сохранённого собственного эксперимента без повторной генерации.")'),
        md('## Полнота и целостность данных'),
        code('import hashlib\n'
             'manifest = json.loads((RUN / "manifest.json").read_text())\n'
             'rows = json.loads((RUN / "scored.json").read_text())\n'
             'assert manifest["status"] == "complete" and len(rows) == 360\n'
             'assert len({(r["model"], r["id"], r["mode"], r["repetition"]) for r in rows}) == 360\n'
             'assert all("audit" in r for r in rows)\n'
             'assert manifest["dataset_sha256"] == hashlib.sha256((ROOT / "data/questions.json").read_bytes()).hexdigest()\n'
             'assert all(r["answer_sha256"] == hashlib.sha256(r["answer"].encode()).hexdigest() for r in rows)\n'
             'print("Все 360 ответов проверены; хеши набора и текстов совпадают.")\n'
             'print("Процессор:", manifest["cpu"], "| версия реализации:", manifest["backend_version"])'),
        md('## Строгая оценка и проверка содержания\n\n'
           'Правильный ответ может не содержать маркер. Нарушение формата рассматривается '
           'отдельно от фактической ошибки.'),
        code('aggregate = pd.read_csv(RUN / "aggregate.csv")\n'
             'show_ru(aggregate[["model", "mode", "factual_exact", "factual_content", '
             '"math_accuracy", "math_content", "steps_valid_rate", "complete_solution_rate"]].to_dict("records"))'),
        code('display(Image(filename=str(RUN / "audited_accuracy.png"), width=1100))'),
        md('## Задержка, скорость и перплексия\n\n'
           'Среда: CPU, 4 потока, F16. Низкая PPL не доказывает правильность. '
           'Референсная PPL считает все тексты одной Qwen и имеет стилевое смещение.'),
        code('show_ru(aggregate[["model", "mode", "ttft_median_ms", "latency_median_ms", '
             '"latency_p95_ms", "decode_tps_median", "reference_ppl_token_weighted", "truncation_rate"]].to_dict("records"))\n'
             'display(Image(filename=str(RUN / "comparison.png"), width=1100))'),
        code('resources = json.loads((RUN / "resources.json").read_text())\n'
             'show_ru([{\n'
             '    "model": r["model"], "weights_GiB": r["weight_bytes"] / 2**30,\n'
             '    "peak_RSS_GiB": r["sampled_peak_rss_bytes"] / 2**30,\n'
             '    "native_PPL": r["corpus_native"]["ppl"],\n'
             '    "native_bits_per_byte": r["corpus_native"]["bits_per_byte"]\n'
             '} for r in resources])'),
        md('## Неопределённость и парные сравнения\n\n'
           'Единица перевыборки — вопрос; его повторы остаются вместе. '
           'Интервалы Уилсона используют первый повтор с n=12/8. Бутстреп разведочный.'),
        code('show_ru(json.loads((RUN / "content_comparisons.json").read_text()))\n'
             'show_ru(json.loads((RUN / "content_decoding_comparisons.json").read_text()))'),
        md('## Проверка математических решений\n\n'
           'Верный результат ≠ верные шаги ≠ полное решение. Проверка одна, выполнена '
           'ИИ-ассистентом без независимого оценщика-человека. Решения связаны с хешами текстов.'),
        code('review = pd.read_csv(RUN / "manual_review.csv")\n'
             'assert len(review) == 144\n'
             'show_ru(review[["model", "mode", "id", "repetition", "content_correct", '
             '"steps_valid", "solution_complete", "note"]].head(16).to_dict("records"))'),
        md('## Обсуждение и ограничения'),
        code('from IPython.display import Markdown\n'
             'display(Markdown((RUN / "discussion.md").read_text()))'),
        md('## Проверка программы\n\n'
           'Тесты подтверждают работу оценщика и обработки данных, а не качество языковых моделей.'),
        code('import xml.etree.ElementTree as ET\n'
             'suite = ET.parse(ROOT / "reports/tests.xml").find("testsuite")\n'
             'assert int(suite.attrib["failures"]) == 0 and int(suite.attrib["errors"]) == 0\n'
             'print("Пройдено программных тестов:", suite.attrib["tests"])\n'
             'assert (ROOT / "reports/raport.pdf").exists()\n'
             'print("Русский отчёт PDF/Word и приложения сохранены в проекте.")')]
    notebook=nbformat.v4.new_notebook(cells=cells,metadata={
        'kernelspec':{'name':'llm1','display_name':'ЛЛ1 (.venv)','language':'python'},
        'language_info':{'name':'python','version':'3.12'}})
    client=NotebookClient(notebook,timeout=120,kernel_name='llm1',
                          resources={'metadata':{'path':str(ROOT)}})
    client.execute()
    nbformat.validate(notebook)
    nbformat.write(notebook,ROOT/'laborator.ipynb')
    executed=sum(c.cell_type=='code' and c.execution_count is not None for c in notebook.cells)
    print(f'Выполнено ячеек блокнота: {executed}')


if __name__=='__main__':
    main()

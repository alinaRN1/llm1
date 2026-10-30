"""Сборка русского отчёта из завершённого эксперимента и проверенных ответов."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
from zoneinfo import ZoneInfo

ROOT=Path(__file__).resolve().parents[1]
NAMES={'ai/qwen2.5:0.5B-F16':'Qwen 0,5B','ai/smollm2:360m-f16':'Smol 360M',
       'ai/llama3.2:1B-F16':'Llama 1B'}
MODES={'greedy':'Жадный','sampled':'Выборка'}


def table(headers,rows):
    def cell(value):
        return str(value).replace('|','/').replace('\n',' ')
    return ('| '+' | '.join(headers)+' |\n|'+ '|'.join('---' for _ in headers)+'|\n'+
            ''.join('| '+' | '.join(cell(v) for v in row)+' |\n' for row in rows)+'\n')


def pct(x):
    return f'{100*x:.1f}%'


def interval(values):
    return f'[{100*values[0]:.1f}; {100*values[1]:.1f}]'


def empirical(run):
    manifest=json.loads((run/'manifest.json').read_text(encoding="utf-8"))
    rows=json.loads((run/'scored.json').read_text(encoding="utf-8"))
    aggregates=json.loads((run/'aggregate.json').read_text(encoding="utf-8"))
    resources=json.loads((run/'resources.json').read_text(encoding="utf-8"))
    audit=json.loads((run/'audit_metadata.json').read_text(encoding="utf-8"))
    if manifest['status']!='complete' or audit['responses_reviewed']!=360 or len(rows)!=360:
        raise ValueError('Отчёт требует полного эксперимента и проверки 360 ответов')
    if any('audit' not in r for r in rows):
        raise ValueError('Один из ответов не прошёл проверку содержания')
    started=datetime.fromisoformat(manifest['started_utc'].replace('Z','+00:00'))
    ended=datetime.fromisoformat(manifest['finished_utc'].replace('Z','+00:00'))
    local=ZoneInfo('Europe/Chisinau')
    seconds=sum(r['latency_ms'] for r in rows)/1000
    output='# Экспериментальные результаты\n\n'
    output+=(f'Получены и сохранены **{len(rows)} реальных ответов**, по 120 на модель. '
             f'Проверены все {audit["responses_reviewed"]} ответов '
             f'({audit["unique_model_item_answers"]} уникальных сочетаний модели, задания и текста). '
             'Синтетические ответы в результаты эксперимента не включены.\n\n')
    output+=table(['Параметр','Значение'],[
        ['Реализация',manifest['backend']+' '+manifest['backend_version']],
        ['Python',manifest['python']],['Процессор',manifest['cpu']],
        ['Квота CPU (cpu.max)',manifest['cpu_quota']],['Потоков',manifest['arguments']['threads']],
        ['Доступная память',f"{manifest['ram_bytes']/2**30:.2f} ГиБ"],
        ['Контекст / пакет','1024 / 512; logits_all=True; без GPU'],
        ['Начало, Кишинёв',started.astimezone(local).strftime('%d.%m.%Y %H:%M:%S %Z')],
        ['Окончание, Кишинёв',ended.astimezone(local).strftime('%d.%m.%Y %H:%M:%S %Z')],
        ['Сумма времени запросов',f'{seconds:.2f} с'],
        ['Полный период',f'{(ended-started).total_seconds():.1f} с, включая диагностику и возобновление']])
    output+=('После первого блока обнаружен пакет базовой SmolLM2 без шаблона диалога. '
             'До измерений SmolLM2 он заменён на Instruct. Готовый блок Qwen использован '
             'повторно после проверки набора, настроек и контрольной суммы. Манифест '
             'первой попытки сохранён; полный период включает диагностическую паузу.\n\n')
    output+='## Строгая правильность и проверка содержания\n\n'
    output+=table(['Модель','Режим','Факты: строго','Факты: проверка','Мат.: строго','Мат.: проверка'],[
        [NAMES[r['model']],MODES[r['mode']],pct(r['factual_exact']),pct(r['factual_content']),
         pct(r['math_accuracy']),pct(r['math_content'])] for r in aggregates])
    output+=('Среднее по начальным значениям использует 36 фактологических и 24 математических '
             'ответа на сочетание модели и режима. Независимых вопросов по-прежнему 12 и 8. '
             'Разрыв между строгой оценкой и проверкой содержания показывает влияние формата.\n\n')
    output+='### Разбиение по предметным областям\n\n'
    categories=[]
    for a in aggregates:
        item=[NAMES[a['model']],MODES[a['mode']]]
        for category in ['istorie','romana','matematica']:
            selected=[r for r in rows if r['model']==a['model'] and r['mode']==a['mode'] and r['category']==category]
            item.append(pct(sum(r['audit']['content_correct'] for r in selected)/len(selected)))
        categories.append(item)
    output+=table(['Модель','Режим','История','Румынский язык','Математика'],categories)
    output+='### Интервалы Уилсона по первому повтору\n\n'
    ci=[]
    for a in aggregates:
        subset=[r for r in rows if r['model']==a['model'] and r['mode']==a['mode'] and r['repetition']==0]
        f=[r for r in subset if r['category']!='matematica']
        m=[r for r in subset if r['category']=='matematica']
        ci.append([NAMES[a['model']],MODES[a['mode']],
            f"{sum(r['audit']['content_correct'] for r in f)}/12 "+interval(a['factual_content_wilson_rep0']),
            f"{sum(r['audit']['content_correct'] for r in m)}/8 "+interval(a['math_content_wilson_rep0'])])
    output+=table(['Модель','Режим','Факты: k/n, ДИ 95%','Математика: k/n, ДИ 95%'],ci)
    output+='Интервалы относятся к повтору 0, а не к среднему по трём повторам.\n\n'
    output+='## Математика: верные шаги и полные решения\n\n'
    output+=table(['Модель','Режим','Верный результат','Верные шаги','Полное решение'],[
        [NAMES[r['model']],MODES[r['mode']],pct(r['math_content']),pct(r['steps_valid_rate']),
         pct(r['complete_solution_rate'])] for r in aggregates])
    output+='Все 144 математических ответа проверены; пояснения и хеши приведены в `manual_review_ru.csv`. Оценка шагов не выводится из одного присутствия слова «шаг».\n\n'
    output+='![Правильность содержания и полнота решений]('+os.path.relpath(run/'audited_accuracy.png',ROOT/'reports')+'){width=100%}\n\n'
    output+='## Задержка и скорость\n\n'
    output+=table(['Модель','Режим','TTFT, мс','Задержка, мс','P95, мс','Токен/с','Обрывы'],[
        [NAMES[r['model']],MODES[r['mode']],f"{r['ttft_median_ms']:.1f}",f"{r['latency_median_ms']:.1f}",
         f"{r['latency_p95_ms']:.1f}",f"{r['decode_tps_median']:.2f}",pct(r['truncation_rate'])] for r in aggregates])
    output+='TTFT, задержка и токены/с представлены медианами; P95 — 95-й процентиль задержки. Прогрев исключён. Скорость рассматривается совместно с длиной и правильностью ответов.\n\n'
    output+='![Автоматические метрики реальных запросов]('+os.path.relpath(run/'comparison.png',ROOT/'reports')+'){width=100%}\n\n'
    output+='## Перплексия\n\n'
    output+=table(['Модель','Целевых токенов','Собственная PPL','Бит/байт'],[
        [NAMES[r['model']],r['corpus_native']['targets'],f"{r['corpus_native']['ppl']:.3f}",
         f"{r['corpus_native']['bits_per_byte']:.3f}"] for r in resources])
    output+=table(['Модель','Режим','Взвешенная референсная PPL'],[
        [NAMES[r['model']],MODES[r['mode']],f"{r['reference_ppl_token_weighted']:.3f}"] for r in aggregates])
    output+='Разное число целевых токенов на одном тексте отражает влияние токенизации. Референсная PPL использует Qwen и включает маркеры и шаги. Малая величина не доказывает правильность и не служит основанием выбора победителя.\n\n'
    output+='## Ресурсы и стоимость\n\n'
    output+=table(['Модель','Веса, ГиБ','RSS, ГиБ','Запуск, с','CPU генерации, с'],[
        [NAMES[r['model']],f"{r['weight_bytes']/2**30:.3f}",f"{r['sampled_peak_rss_bytes']/2**30:.3f}",
         f"{r['load_seconds']:.3f}",f"{sum(x['cpu_seconds'] for x in rows if x['model']==r['model']):.2f}"] for r in resources])
    output+=(f'Сумма астрономического времени 360 запросов — {seconds:.2f} с. При условном '
             f'тарифе r денежных единиц в час только эти измеренные запросы соответствуют '
             f'коэффициенту {seconds/3600:.6f} × r. Это не счёт за инфраструктуру: запуск, '
             'PPL, хранение и остальное резервирование добавляются по реальным правилам '
             'оплаты. Плата API за токены при локальном запуске равна 0; облачный тариф '
             'и энергопотребление не предоставлены и не измерены.\n\n')
    output+='## Парные сравнения\n\n'
    comparisons=json.loads((run/'content_comparisons.json').read_text(encoding="utf-8"))
    output+=table(['Режим','A − B','Разность, п.п.','ДИ 95%, п.п.'],[
        [MODES[r['mode']],NAMES[r['model_a']]+' − '+NAMES[r['model_b']],f"{100*r['difference']:.2f}",
         interval(r['cluster_bootstrap_95'])] for r in comparisons])
    output+='Метрика — правильность содержания на 20 вопросах с группировкой повторов. П.п. — процентные пункты. Интервалы разведочные, без поправки на множественные сравнения.\n\n'
    mode_rows=[]
    mode_intervals=json.loads((run/'content_decoding_comparisons.json').read_text(encoding="utf-8"))
    for model in manifest['models']:
        g=next(a for a in aggregates if a['model']==model and a['mode']=='greedy')
        s=next(a for a in aggregates if a['model']==model and a['mode']=='sampled')
        mode_ci=next(r['cluster_bootstrap_95'] for r in mode_intervals if r['model']==model)
        mode_rows.append([NAMES[model],pct(g['overall_content']),pct(s['overall_content']),
                          f"{100*(s['overall_content']-g['overall_content']):+.2f}",interval(mode_ci)])
    output+=table(['Модель','Жадный: содержание','Выборка: содержание','Разность, п.п.','ДИ 95%, п.п.'],mode_rows)
    output+='Каждый вопрос имеет одинаковый вес: 60 ответов на модель и режим. Эффект температуры не отделён от top-p и штрафа повторения.\n\n'
    output+='## Примеры ошибок и чувствительность оценщика\n\n'
    output+='Ниже приведены русские переводы фрагментов для чтения; исходные ответы с теми же числовыми утверждениями и хешами сохранены в данных.\n\n'
    examples=[(('ai/qwen2.5:0.5B-F16',1,'greedy',0),
        'Объединение румынских княжеств произошло в 1918 году.'),
        (('ai/qwen2.5:0.5B-F16',18,'sampled',1),
        'Ответ: 1\n\nПроверка:\n3(1) - 4 = 2\n3 - 4 = 2\n-1 = 2 (семь)\n\nРешение: x = 1.'),
        (('ai/qwen2.5:0.5B-F16',20,'greedy',0),
        'ИТОГОВЫЙ ОТВЕТ: 7² + 5² = 49 + 25 = 74'),
        (('ai/llama3.2:1B-F16',13,'greedy',0),
        '245 + 378 = 623\n\nПромежуточные вычисления:\n1. 245 + 200 = 445\n2. 445 + 78 = 523\n3. 523 + 56 = 579\n\nИтоговый ответ: 579')]
    for key,translation in examples:
        r=next(r for r in rows if (r['model'],r['id'],r['mode'],r['repetition'])==key)
        output+=f"### {NAMES[r['model']]}, задание {r['id']}, {MODES[r['mode']]}, повтор {r['repetition']}\n\n"
        output+='```text\n'+translation+'\n```\n\n'+r['audit']['note']+'\n\n'
    discussion=run/'discussion.md'
    if not discussion.exists():
        raise ValueError('Нужны выводы по измеренным результатам')
    output+=discussion.read_text(encoding="utf-8")+'\n\n'
    return output


def appendices(run):
    output='# Приложение А. Вопросы и эталоны\n\n'
    output+='Ниже дан вспомогательный русский перевод. При измерении модели получали румынские оригиналы из `data/questions.json`. Перевод не использовался для повторного расчёта вероятностей или оценок.\n\n'
    original={q['id']:q for q in json.loads((ROOT/'data/questions.json').read_text(encoding="utf-8"))}
    for item in json.loads((ROOT/'data/questions_ru.json').read_text(encoding="utf-8")):
        output+=f"## Задание {item['id']} — {item['category']}\n\n{item['question']}\n\n"
        output+='Эталон: '+item['reference_answer']+'.\n\n'
        source=original[item['id']]
        if 'reference_steps' in source:
            output+='Эталонные вычисления:\n\n'+ '\n'.join('- `'+s+'`' for s in source['reference_steps'])+'\n\n'
        else:
            output+='Источник: '+source['source'].replace('intrarea','статья').replace('intrările','статьи')+'\n\n'
    output+='# Приложение Б. Воспроизводимость\n\n'
    manifest=json.loads((run/'manifest.json').read_text(encoding="utf-8"))
    output+='SHA-256 исходного набора: `'+manifest['dataset_sha256']+'`\n\n'
    output+='SHA-256 корпуса: `'+manifest['corpus_sha256']+'`\n\n'
    for model in manifest['artifacts']:
        output+='**'+model['model']+'**\n\nМанифест: `'+model['manifest_digest']+'`\n\n'
        layer=next(x for x in model['layers'] if x['file']=='model.gguf')
        output+='GGUF: `'+layer['digest']+'`\n\n'
    output+='Команды выполняются из корня проекта:\n\n```bash\nbash install.sh\n.venv/bin/python src/pull_oci.py --locked data/models_oci.lock.json\n.venv/bin/python src/benchmark_gguf.py --run-id replicare-01\n.venv/bin/python src/analyze.py results/runs/replicare-01\n```\n\n'
    output+='Если новые ответы отличаются, требуется новая проверка. Существующее заключение принимается только для той же модели, задания и хеша текста. Для сохранённых итоговых результатов:\n\n```bash\n.venv/bin/python src/audit.py results/final data/review_decisions.json\n.venv/bin/python src/analyze.py results/final\n.venv/bin/python src/build_report.py --run results/final\n```\n\n'
    output+='Цифровые приложения содержат все ответы, пояснения проверки, ресурсы, таблицы CSV/JSON, интервалы бутстрепа, графики, выполненный блокнот и протокол тестов. Читаемые таблицы с суффиксом `_ru` оформлены по-русски; машинные идентификаторы исходной схемы сохранены. Веса скачиваются отдельно по контрольным суммам.\n'
    return output


def build(run):
    source=(ROOT/'reports/template.md').read_text(encoding="utf-8")
    source=source.replace('<!-- EMPIRICAL_RESULTS -->',empirical(run))
    source=source.replace('<!-- APPENDICES -->',appendices(run))
    target=ROOT/'reports/raport.md'
    target.write_text(source, encoding="utf-8")
    if not shutil.which('pandoc') or not shutil.which('xelatex'):
        raise RuntimeError('Для экспорта Word/PDF нужны Pandoc и XeLaTeX')
    subprocess.run(['pandoc',str(target),'-o',str(target.with_suffix('.docx')),'--toc'],cwd=ROOT/'reports',check=True)
    cache=Path('/tmp/llm1-font-cache');cache.mkdir(exist_ok=True)
    environment=os.environ.copy();environment['XDG_CACHE_HOME']=str(cache)
    subprocess.run(['pandoc',str(target),'-o',str(target.with_suffix('.pdf')),
                    '--pdf-engine=xelatex'],cwd=ROOT/'reports',env=environment,check=True)
    print(target.with_suffix('.pdf'))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description='Сборка русского отчёта лабораторной')
    parser.add_argument('--run',type=Path,default=ROOT/'results/final',help='Папка завершённого эксперимента')
    build(parser.parse_args().run.resolve())

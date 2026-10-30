"""Русское оформление таблиц без изменения машинной схемы и измерений."""
import csv

MODES={'greedy':'Жадный','sampled':'Выборка'}
CATEGORIES={'istorie':'История','romana':'Румынский язык','matematica':'Математика'}
HEADERS={
    'model':'Модель','mode':'Режим','id':'Задание','repetition':'Повтор',
    'category':'Категория','question':'Вопрос','reference_answer':'Эталон',
    'original_question':'Исходный вопрос (рум.)','answer':'Исходный ответ (рум.)',
    'answer_sha256':'SHA-256 ответа','note':'Пояснение проверки',
    'content_correct':'Содержание верно','steps_valid':'Шаги верны','solution_complete':'Решение полное',
    'factual_exact':'Факты: строгая точность','factual_partial':'Факты: частичный Жаккар',
    'math_accuracy':'Математика: строгая точность',
    'factual_wilson_rep0':'Факты: строгий ДИ Уилсона, повтор 0',
    'math_wilson_rep0':'Математика: строгий ДИ Уилсона, повтор 0',
    'latency_median_ms':'Медианная задержка, мс','latency_p95_ms':'P95 задержки, мс',
    'ttft_median_ms':'Медианное время первого токена, мс','decode_tps_median':'Медианная скорость, токен/с',
    'generation_tokens_mean':'Среднее число выходных токенов','truncation_rate':'Доля обрывов',
    'format_success':'Доля наличия итогового маркера','cpu_seconds_sum':'Время CPU генерации, с',
    'reference_ppl_token_weighted':'Взвешенная референсная PPL',
    'arithmetic_equations_checked':'Проверено числовых равенств',
    'arithmetic_equations_valid':'Верных числовых равенств','manual_steps_review':'Состояние проверки шагов',
    'factual_content':'Факты: проверка содержания','math_content':'Математика: проверка результата',
    'overall_content':'Общая правильность содержания','steps_valid_rate':'Доля верных шагов',
    'complete_solution_rate':'Доля полных решений',
    'factual_content_wilson_rep0':'Факты: ДИ содержания, повтор 0',
    'math_content_wilson_rep0':'Математика: ДИ содержания, повтор 0',
    'accuracy':'Строгая точность','independent_questions':'Независимых вопросов',
    'wilson_rep0':'Строгий ДИ Уилсона, повтор 0','content_accuracy':'Точность содержания',
    'content_wilson_rep0':'ДИ Уилсона содержания, повтор 0',
    'model_a':'Модель A','model_b':'Модель B','difference':'Разность A−B / выборка−жадный',
    'cluster_bootstrap_95':'Групповой бутстреп: ДИ 95%',
    'metric':'Метрика','questions':'Вопросов','unit':'Единица анализа',
    'interpretation':'Интерпретация','sampled_minus_greedy':'Выборка минус жадный режим',
    'manifest_digest':'SHA-256 манифеста','GGUF_bytes':'Размер GGUF, байт',
    'weights_GiB':'Веса, ГиБ','peak_RSS_GiB':'Пиковая RSS, ГиБ',
    'native_PPL':'Собственная PPL','native_bits_per_byte':'Собственная оценка, бит/байт'
}


def localize_records(records):
    """Переводятся заголовки и условные обозначения, но не ответы моделей."""
    translated=[]
    for row in records:
        result={}
        for key,value in row.items():
            if key=='mode':value=MODES.get(value,value)
            elif key=='category':value=CATEGORIES.get(value,value)
            elif key=='manual_steps_review':value={'reviewed':'Проверено','pending':'Ожидает проверки'}.get(value,value)
            elif key=='unit' and value=='question':value='Вопрос'
            elif key=='metric' and value=='content_correct':value='Правильность содержания'
            result[HEADERS.get(key,key)]=value
        translated.append(result)
    return translated


def write_russian_csv(path,records):
    rows=localize_records(records)
    if not rows:
        return
    with path.open('w',encoding='utf-8-sig',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

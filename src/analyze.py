"""Таблицы, интервалы и графики строятся только по завершённому реальному запуску."""
import argparse
import itertools
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from metrics import wilson
from presentation import MODES,write_russian_csv,localize_records

LABELS={'ai/qwen2.5:0.5B-F16':'Qwen 0.5B','ai/smollm2:360m-f16':'Smol 360M',
        'ai/llama3.2:1B-F16':'Llama 1B'}


def analyze(run):
    manifest=json.loads((run/'manifest.json').read_text(encoding="utf-8"))
    if manifest['status']!='complete':
        raise ValueError('Незавершённый запуск: эмпирический рейтинг построить нельзя')
    rows=json.loads((run/'scored.json').read_text(encoding="utf-8"))
    data=pd.DataFrame(rows)
    expected=3*20*2*manifest['arguments']['repetitions']
    if len(data)!=expected or data.duplicated(['model','mode','id','repetition']).any():
        raise ValueError('Наблюдения отсутствуют либо дублируются')
    if set(data.id)!=set(range(1,21)) or set(data['mode'])!={'greedy','sampled'}:
        raise ValueError('Набор вопросов или режимов не соответствует протоколу')
    audited='audit' in data and all(isinstance(r.get('audit'),dict) for r in rows)
    if audited:
        data['content_correct']=[r['audit']['content_correct'] for r in rows]
        data['steps_valid']=[r['audit']['steps_valid'] for r in rows]
        data['solution_complete']=[r['audit']['solution_complete'] for r in rows]
    grouped=[]
    for (model,mode),df in data.groupby(['model','mode']):
        factual=df[df.category!='matematica']
        math=df[df.category=='matematica']
        # Интервалы Уилсона используют независимые вопросы; три повтора не утраивают n.
        first=factual[factual.repetition==0]
        first_math=math[math.repetition==0]
        nll=sum(r['reference_ppl']['nll'] or 0 for r in df.to_dict('records'))
        targets=sum(r['reference_ppl']['targets'] for r in df.to_dict('records'))
        group=dict(model=model,mode=mode,factual_exact=float(factual.exact.mean()),
            factual_partial=float(factual.partial.mean()),math_accuracy=float(math.math_correct.mean()),
            factual_wilson_rep0=wilson(first.exact.sum(),len(first)),
            math_wilson_rep0=wilson(first_math.math_correct.sum(),len(first_math)),
            latency_median_ms=float(df.latency_ms.median()),latency_p95_ms=float(df.latency_ms.quantile(.95)),
            ttft_median_ms=float(df.ttft_ms.median()),decode_tps_median=float(df.decode_tps.median()),
            generation_tokens_mean=float(df.generated_tokens.mean()),
            truncation_rate=float(df.truncated.mean()),format_success=float(df.format_ok.mean()),
            cpu_seconds_sum=float(df.cpu_seconds.sum()),
            reference_ppl_token_weighted=float(np.exp(nll/targets)) if targets else None,
            arithmetic_equations_checked=int(math.equations_checked.sum()),
            arithmetic_equations_valid=int(math.equations_valid.sum()),
            manual_steps_review='reviewed' if audited else 'pending')
        if audited:
            audited_fact=df[df.category!='matematica']
            audited_math=df[df.category=='matematica']
            f0=audited_fact[audited_fact.repetition==0]
            m0=audited_math[audited_math.repetition==0]
            group.update(factual_content=float(audited_fact.content_correct.mean()),
                math_content=float(audited_math.content_correct.mean()),
                overall_content=float(df.content_correct.mean()),
                steps_valid_rate=float(audited_math.steps_valid.mean()),
                complete_solution_rate=float(audited_math.solution_complete.mean()),
                factual_content_wilson_rep0=wilson(f0.content_correct.sum(),len(f0)),
                math_content_wilson_rep0=wilson(m0.content_correct.sum(),len(m0)))
        grouped.append(group)
    (run/'aggregate.json').write_text(json.dumps(grouped,ensure_ascii=False,indent=2), encoding="utf-8")
    pd.DataFrame(grouped).to_csv(run/'aggregate.csv',index=False)
    write_russian_csv(run/'aggregate_ru.csv',grouped)
    # Парный бутстреп: выбираем вопросы, сохраняя вместе все их повторы.
    rng=np.random.default_rng(20261030)
    comparisons=[]
    for mode in ['greedy','sampled']:
        frame=data[data['mode']==mode].copy()
        frame['correct']=frame['exact'].fillna(frame['math_correct'])
        matrix=frame.groupby(['id','model']).correct.mean().unstack()
        for a,b in itertools.combinations(matrix.columns,2):
            delta=(matrix[a]-matrix[b]).to_numpy()
            samples=delta[rng.integers(0,len(delta),size=(10000,len(delta)))].mean(axis=1)
            comparisons.append(dict(mode=mode,model_a=a,model_b=b,
                difference=float(delta.mean()),cluster_bootstrap_95=np.quantile(samples,[.025,.975]).tolist(),
                questions=len(delta),unit='question',interpretation='Разведочное сравнение без поправки на множественность'))
    (run/'paired_comparisons.json').write_text(json.dumps(comparisons,indent=2), encoding="utf-8")
    write_russian_csv(run/'paired_comparisons_ru.csv',comparisons)
    category_rows=[]
    frame=data.copy()
    frame['correct']=frame['exact'].fillna(frame['math_correct'])
    for (model,mode,category),df in frame.groupby(['model','mode','category']):
        first=df[df.repetition==0]
        category_result=dict(model=model,mode=mode,category=category,
            accuracy=float(df.correct.mean()),independent_questions=len(first),
            wilson_rep0=wilson(first.correct.sum(),len(first)))
        if audited:
            category_result.update(content_accuracy=float(df.content_correct.mean()),
                content_wilson_rep0=wilson(first.content_correct.sum(),len(first)))
        category_rows.append(category_result)
    pd.DataFrame(category_rows).to_csv(run/'category_summary.csv',index=False)
    write_russian_csv(run/'category_summary_ru.csv',category_rows)
    mode_comparisons=[]
    for model,df in frame.groupby('model'):
        matrix=df.groupby(['id','mode']).correct.mean().unstack()
        delta=(matrix['sampled']-matrix['greedy']).to_numpy()
        samples=delta[rng.integers(0,len(delta),size=(10000,len(delta)))].mean(axis=1)
        mode_comparisons.append(dict(model=model,sampled_minus_greedy=float(delta.mean()),
            cluster_bootstrap_95=np.quantile(samples,[.025,.975]).tolist(),questions=len(delta)))
    (run/'decoding_comparisons.json').write_text(json.dumps(mode_comparisons,indent=2), encoding="utf-8")
    if audited:
        audited_comparisons=[]
        for mode in ['greedy','sampled']:
            matrix=data[data['mode']==mode].groupby(['id','model']).content_correct.mean().unstack()
            for a,b in itertools.combinations(matrix.columns,2):
                delta=(matrix[a]-matrix[b]).to_numpy()
                samples=delta[rng.integers(0,len(delta),size=(10000,len(delta)))].mean(axis=1)
                audited_comparisons.append(dict(mode=mode,model_a=a,model_b=b,
                    difference=float(delta.mean()),cluster_bootstrap_95=np.quantile(samples,[.025,.975]).tolist(),
                    metric='content_correct',questions=len(delta)))
        (run/'content_comparisons.json').write_text(json.dumps(audited_comparisons,indent=2), encoding="utf-8")
        write_russian_csv(run/'content_comparisons_ru.csv',audited_comparisons)
        content_modes=[]
        for model,df in data.groupby('model'):
            matrix=df.groupby(['id','mode']).content_correct.mean().unstack()
            delta=(matrix['sampled']-matrix['greedy']).to_numpy()
            samples=delta[rng.integers(0,len(delta),size=(10000,len(delta)))].mean(axis=1)
            content_modes.append(dict(model=model,difference=float(delta.mean()),
                cluster_bootstrap_95=np.quantile(samples,[.025,.975]).tolist(),questions=len(delta)))
        (run/'content_decoding_comparisons.json').write_text(json.dumps(content_modes,indent=2), encoding="utf-8")
        write_russian_csv(run/'content_decoding_comparisons_ru.csv',content_modes)
    agg=pd.DataFrame(grouped)
    fig,axes=plt.subplots(2,3,figsize=(15,8))
    metrics=[('factual_exact','Факты: строгая точность'),('math_accuracy','Математика: строгая точность'),
             ('factual_partial','Частичная оценка Жаккара'),('latency_median_ms','Медианная задержка, мс'),
             ('decode_tps_median','Декодирование, токен/с'),('reference_ppl_token_weighted','Референсная PPL')]
    for ax,(key,title) in zip(axes.ravel(),metrics):
        table=agg.pivot(index='model',columns='mode',values=key)
        table.index=[LABELS.get(x,x.split('/')[-1]) for x in table.index]
        table=table.rename(columns=MODES)
        table.plot.bar(ax=ax,rot=20)
        ax.set_title(title)
        ax.set_xlabel('')
        ax.legend(title='Режим')
        if key in ['factual_exact','math_accuracy','factual_partial']:
            ax.set_ylim(0,1)
    fig.tight_layout()
    fig.savefig(run/'comparison.png',dpi=180)
    fig.savefig(run/'comparison.svg')
    plt.close(fig)
    if audited:
        fig,axes=plt.subplots(1,3,figsize=(13,4.5))
        for ax,(key,title) in zip(axes,[('factual_content','Факты: проверка содержания'),
            ('math_content','Математика: верный результат'),('complete_solution_rate','Математика: полное решение')]):
            table=agg.pivot(index='model',columns='mode',values=key)
            table.index=[LABELS.get(x,x.split('/')[-1]) for x in table.index]
            table=table.rename(columns=MODES)
            table.plot.bar(ax=ax,rot=15)
            ax.set_title(title)
            ax.set_ylim(0,1)
            ax.set_xlabel('')
            ax.legend(title='Режим')
        fig.tight_layout()
        fig.savefig(run/'audited_accuracy.png',dpi=180)
        fig.savefig(run/'audited_accuracy.svg')
        plt.close(fig)
    # Стоимость условна: тариф и мощность CPU не выдумываются.
    (run/'results_summary.md').write_text(
        '# Измеренные результаты\n\n```csv\n'+pd.DataFrame(localize_records(
        pd.DataFrame(grouped)[['model','mode','factual_exact','math_accuracy',
        'latency_median_ms','decode_tps_median','truncation_rate']].to_dict('records'))).to_csv(index=False)+
        '```\n\nИнтервалы Уилсона относятся только к повтору 0 (12/8 вопросов). '
        'Бутстреп сохраняет группировку по вопросам. Проверка шагов: '+
        ('выполнена, заключения связаны с хешами.' if audited else 'ожидается.')+' '
        'Денежная стоимость = оплачиваемые часы × реальный тариф; тариф не предоставлен.\n', encoding="utf-8")
    print(run/'results_summary.md')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('run',type=Path)
    analyze(parser.parse_args().run)

"""Присоединение явных заключений, связанных с хешами; оценки не угадываются."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from presentation import write_russian_csv


def response_hash(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def audit(run,decisions):
    rows=json.loads((run/'scored.json').read_text(encoding="utf-8"))
    decisions=json.loads(decisions.read_text(encoding="utf-8"))
    lookup={}
    for decision in decisions:
        key=(decision['model'],decision['id'],decision['answer_sha256'])
        if key in lookup:
            raise ValueError('Повторяющееся заключение проверки')
        for field in ['content_correct','steps_valid','solution_complete']:
            value=decision.get(field)
            if field=='content_correct' or value is not None:
                if type(value) is not int or value not in (0,1):
                    raise ValueError('Недопустимая оценка проверки')
        if not decision.get('note'):
            raise ValueError('Для оценки требуется пояснение')
        lookup[key]=decision
    csv_rows=[]
    for row in rows:
        key=(row['model'],row['id'],response_hash(row['answer']))
        if key not in lookup:
            raise ValueError(f"Нет заключения проверки: {key}")
        decision=lookup[key]
        if row['category']=='matematica' and any(decision.get(k) is None for k in ['steps_valid','solution_complete']):
            raise ValueError('Для математического ответа не проверены шаги')
        row['audit']={k:decision[k] for k in ['content_correct','steps_valid','solution_complete','note']}
        row['answer_sha256']=key[2]
        if row['category']=='matematica':
            row['manual_steps_status']='reviewed'
        csv_rows.append({k:row[k] for k in ['model','mode','id','repetition','category','answer','answer_sha256']}|row['audit'])
    (run/'scored.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2,allow_nan=False), encoding="utf-8")
    with (run/'audit.csv').open('w', encoding="utf-8", newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    with (run/'manual_review.csv').open('w', encoding="utf-8", newline="") as f:
        selected=[r for r in csv_rows if r['category']=='matematica']
        writer=csv.DictWriter(f,fieldnames=list(selected[0]))
        writer.writeheader()
        writer.writerows(selected)
    write_russian_csv(run/'audit_ru.csv',csv_rows)
    write_russian_csv(run/'manual_review_ru.csv',selected)
    (run/'audit_metadata.json').write_text(json.dumps(dict(
        responses_reviewed=len(rows),unique_model_item_answers=len(lookup),
        reviewer='ИИ-ассистент; одна качественная проверка без независимой оценки человеком',
        rubric={'content_correct':'Правильное содержание независимо от итогового маркера',
                'steps_valid':'Хотя бы один содержательный верный шаг по задаче без математических противоречий',
                'solution_complete':'Правильный ответ со всеми требуемыми вычислениями и проверкой'},
        decisions_sha256=hashlib.sha256(json.dumps(decisions,sort_keys=True,ensure_ascii=False).encode()).hexdigest()),
        ensure_ascii=False,indent=2), encoding="utf-8")
    print(f'Проверено {len(rows)} ответов; {len(lookup)} уникальных сочетаний модели, задания и текста')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('run',type=Path)
    parser.add_argument('decisions',type=Path)
    args=parser.parse_args()
    audit(args.run,args.decisions)

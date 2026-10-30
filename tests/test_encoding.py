"""Регрессия чтения и записи Unicode при системной кодировке Windows-1251."""
import json
from pathlib import Path
import sys

import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from audit import audit
from analyze import analyze


def test_notebook_and_evaluation_pipeline_with_cp1251_default(tmp_path,monkeypatch):
    original_open=Path.open

    def windows_open(self,mode='r',buffering=-1,encoding=None,errors=None,newline=None):
        if 'b' not in mode and encoding in (None,'locale'):
            encoding='cp1251'
        return original_open(self,mode,buffering,encoding,errors,newline)

    monkeypatch.setattr(Path,'open',windows_open)
    probe=tmp_path/'проверка.txt'
    probe.write_bytes('История и Ștefan'.encode('utf-8'))
    # Контроль воспроизводит ту же ошибку, что возникла у пользователя на Windows.
    with pytest.raises(UnicodeDecodeError):
        probe.read_text()

    notebook=json.loads((ROOT/'laborator.ipynb').read_text(encoding='utf-8'))
    cell=next(c['source'] for c in notebook['cells']
              if c['cell_type']=='code' and 'questions = json.loads' in ''.join(c['source']))
    source=''.join(cell) if isinstance(cell,list) else cell
    displayed=[]
    exec(compile(source,'<ячейка набора оценки>','exec'),
         {'ROOT':ROOT,'json':json,'show_ru':displayed.extend})
    assert len(displayed)==20 and displayed[0]['category']=='История'

    for name in ['manifest.json','scored.json']:
        (tmp_path/name).write_bytes((ROOT/'results/final'/name).read_bytes())
    decisions=tmp_path/'решения.json'
    decisions.write_bytes((ROOT/'data/review_decisions.json').read_bytes())
    audit(tmp_path,decisions)
    analyze(tmp_path)
    original=json.loads((ROOT/'results/final/scored.json').read_text(encoding='utf-8'))
    written=json.loads((tmp_path/'scored.json').read_text(encoding='utf-8'))
    assert [r['answer'] for r in written]==[r['answer'] for r in original]
    assert [r['audit'] for r in written]==[r['audit'] for r in original]
    assert 'Модель' in (tmp_path/'audit_ru.csv').read_text(encoding='utf-8-sig')
    assert 'Измеренные результаты' in (tmp_path/'results_summary.md').read_text(encoding='utf-8')

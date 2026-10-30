import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from analyze import analyze


def test_failed_run_cannot_be_ranked(tmp_path):
    (tmp_path/'manifest.json').write_text(json.dumps({'status':'failed'}))
    with pytest.raises(ValueError,match='Незавершённый'):
        analyze(tmp_path)


def test_analysis_pipeline_on_labelled_synthetic_fixture(tmp_path):
    # Синтетическая выборка проверяет только агрегацию и не выдаётся за эксперимент.
    (tmp_path/'manifest.json').write_text(json.dumps({'status':'complete','arguments':{'repetitions':1}}))
    rows=[]
    for model in ['fixture/a','fixture/b','fixture/c']:
        for mode in ['greedy','sampled']:
            for i in range(1,21):
                math=i>12
                rows.append(dict(model=model,mode=mode,id=i,repetition=0,
                    category='matematica' if math else 'istorie',exact=None if math else 1,
                    partial=None if math else 1,math_correct=1 if math else None,
                    reference_ppl={'nll':4,'targets':2},latency_ms=100,ttft_ms=10,
                    decode_tps=20,generated_tokens=8,truncated=False,format_ok=True,
                    cpu_seconds=.1,equations_checked=2 if math else None,
                    equations_valid=1 if math else None))
    (tmp_path/'scored.json').write_text(json.dumps(rows))
    analyze(tmp_path)
    aggregate=json.loads((tmp_path/'aggregate.json').read_text())
    assert len(aggregate)==6
    assert all(r['math_accuracy']==1 and r['factual_exact']==1 for r in aggregate)
    assert all(r['manual_steps_review']=='pending' for r in aggregate)
    comparison=json.loads((tmp_path/'paired_comparisons.json').read_text())
    assert len(comparison)==6
    assert all(r['cluster_bootstrap_95']==[0,0] for r in comparison)
    assert (tmp_path/'comparison.png').stat().st_size>1000

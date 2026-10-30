import json
import math
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from benchmark_gguf import perplexity
from audit import audit,response_hash


def test_gguf_likelihood_uniform_distribution_and_overlapping_windows():
    class UniformLM:
        def tokenize(self,text,**kwargs):
            assert kwargs=={'add_bos':False,'special':False}
            return [1,2,3,4,5,6,7,8,9,10,11]
        def reset(self):
            pass
        def eval(self,tokens):
            self.scores=np.zeros((len(tokens),16),dtype=np.float32)
    result=perplexity(UniformLM(),'test text',context=6,stride=3)
    assert result['targets']==10
    assert result['ppl']==pytest.approx(16)
    assert result['nll']==pytest.approx(10*math.log(16))


def test_audit_cannot_attach_to_a_changed_answer(tmp_path):
    rows=[{'model':'fixture/model','id':13,'category':'matematica','answer':'changed',
           'mode':'greedy','repetition':0}]
    (tmp_path/'scored.json').write_text(json.dumps(rows))
    decisions=[{'model':'fixture/model','id':13,'answer_sha256':response_hash('original'),
                'content_correct':1,'steps_valid':0,'solution_complete':0,'note':'Проверка синтетического примера'}]
    path=tmp_path/'decisions.json'
    path.write_text(json.dumps(decisions))
    with pytest.raises(ValueError,match='Нет заключения проверки'):
        audit(tmp_path,path)


def test_complete_audit_roundtrip(tmp_path):
    answer='RĂSPUNS FINAL: 623'
    rows=[{'model':'fixture/model','id':13,'category':'matematica','answer':answer,
           'mode':'greedy','repetition':0}]
    (tmp_path/'scored.json').write_text(json.dumps(rows))
    decisions=[{'model':'fixture/model','id':13,'answer_sha256':response_hash(answer),
                'content_correct':1,'steps_valid':0,'solution_complete':0,
                'note':'Правильный результат без промежуточных вычислений'}]
    path=tmp_path/'decisions.json'
    path.write_text(json.dumps(decisions))
    audit(tmp_path,path)
    reviewed=json.loads((tmp_path/'scored.json').read_text())[0]
    assert reviewed['audit']['content_correct']==1
    assert reviewed['manual_steps_status']=='reviewed'
    assert reviewed['audit']['solution_complete']==0

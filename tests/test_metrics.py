import json
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from metrics import arithmetic, equation_audit, factual, final_answer, math_scores, numeric, target_windows, wilson


def test_unicode_and_aliases():
    assert factual('RĂSPUNS FINAL: Ştefan cel Mare.', ['Ștefan cel Mare'])['exact'] == 1
    assert factual('RĂSPUNS FINAL: Cuza', ['Alexandru Ioan Cuza','Cuza'])['exact'] == 1


def test_negation_not_exact():
    assert factual('RĂSPUNS FINAL: nu 1859', ['1859'])['exact'] == 0


def test_order_and_multiplicity_not_erased_for_exact():
    assert factual('RĂSPUNS FINAL: Mare cel Ștefan', ['Ștefan cel Mare'])['exact'] == 0


def test_missing_final_does_not_guess_last_number():
    assert numeric('2x + 5 = 17. Nu știu, poate 6') is None
    assert factual('1859', ['1859'])['exact'] == 0


@pytest.mark.parametrize('value,expected', [('x = 6',6), ('-2,5',-2.5), ('1/2',.5)])
def test_numeric(value,expected):
    assert float(numeric('RĂSPUNS FINAL: '+value)) == expected


def test_multiple_markers_last_wins():
    assert numeric('RĂSPUNS FINAL: 5\nCorectez.\nRASPUNS FINAL: 6') == 6


def test_extra_numbers_rejected():
    assert numeric('RĂSPUNS FINAL: 6 sau 7') is None


def test_steps_are_not_just_labels():
    score = math_scores('Pas 1: 2 + 2 = 5\nRĂSPUNS FINAL: 6','6')
    assert score['math_correct'] == 1
    assert score['equations_valid'] == 0
    assert score['equations_checked'] == 1
    assert score['manual_steps_status'] == 'pending'


def test_valid_calculations():
    audit = equation_audit('36 × 10 = 360\n36 × 2 = 72\n360 + 72 = 432')
    assert len(audit) == 3 and all(x['valid'] for x in audit)


def test_symbolic_suffix_not_misclassified():
    assert equation_audit('2x + 5 = 17') == []


def test_no_expression_execution():
    with pytest.raises(ValueError):
        arithmetic('__import__("os").system("echo no")')
    with pytest.raises(ValueError):
        arithmetic('2**1000000')


@pytest.mark.parametrize('length',[2,511,512,513,1024,1500])
def test_ppl_targets_once(length):
    targets=[]
    for start,end,first in target_windows(length):
        assert end-start <= 512
        assert first > start
        targets += list(range(first,end))
    assert targets == list(range(1,length))


def test_wilson_small_sample():
    lo,hi=wilson(8,8)
    assert .67 < lo < .68 and hi == 1


def test_dataset_and_reference_math():
    data=json.loads((Path(__file__).resolve().parents[1]/'data/questions.json').read_text(encoding="utf-8"))
    assert len(data)==20 and len({x['id'] for x in data})==20
    for item in data:
        if item['category']=='matematica':
            audit=equation_audit('\n'.join(item['reference_steps']))
            assert len(audit)==len(item['reference_steps'])
            assert all(x['valid'] for x in audit)
            assert math_scores('RĂSPUNS FINAL: '+item['value'],item['value'])['math_correct']==1


def test_ppl_uniform_logits_and_real_transformer():
    import torch
    from transformers import GPT2Config, GPT2LMHeadModel
    from benchmark import perplexity
    # Уменьшенная случайная архитектура проверяет код, а не качество предобученной модели.
    model=GPT2LMHeadModel(GPT2Config(vocab_size=16,n_positions=16,n_embd=8,n_layer=1,n_head=1)).eval()
    for p in model.parameters():
        p.data.zero_()
    class Tokenizer:
        def __call__(self,text,**kwargs):
            return {'input_ids':torch.tensor([[1,2,3,4,5,6,7,8,9]])}
    result=perplexity(model,Tokenizer(),'test text',context=6,stride=3)
    assert result['targets']==8
    assert result['ppl']==pytest.approx(16,rel=1e-5)
    assert result['nll']==pytest.approx(8*math.log(16),rel=1e-5)


def test_generation_timing_with_local_synthetic_transformer():
    import torch
    from transformers import GPT2Config, GPT2LMHeadModel
    from benchmark import generate
    model=GPT2LMHeadModel(GPT2Config(vocab_size=16,n_positions=32,n_embd=8,
                                    n_layer=1,n_head=1,eos_token_id=None)).eval()
    class ChatTokenizer:
        pad_token_id=0
        def apply_chat_template(self,messages,**kwargs):
            assert len(messages)==2 and messages[1]['role']=='user'
            return {'input_ids':torch.tensor([[1,3,4]]),
                    'attention_mask':torch.ones(1,3,dtype=torch.long)}
        def decode(self,tokens,**kwargs):
            return ' '.join(str(int(x)) for x in tokens)
    for mode in ['greedy','sampled']:
        row=generate(model,ChatTokenizer(),'test',mode,123,5)
        assert row['generated_tokens']==5
        assert row['prompt_tokens']==3
        assert 0 < row['ttft_ms'] <= row['latency_ms']
        assert row['decode_tps']>0 and row['truncated']

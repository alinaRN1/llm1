"""Только реальная локальная генерация; неудачный или неполный запуск не даёт итогов."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import threading
import time
import traceback

import numpy as np
import psutil
import torch
from huggingface_hub import HfApi, snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList

from metrics import factual, math_scores, target_windows

ROOT = Path(__file__).resolve().parents[1]
MODELS = ['Qwen/Qwen2.5-0.5B-Instruct', 'HuggingFaceTB/SmolLM2-360M-Instruct',
          'TinyLlama/TinyLlama-1.1B-Chat-v1.0']
SYSTEM = ('Ești un asistent educațional. Răspunde în limba română. '
          'Pentru istorie și limba română scrie doar RĂSPUNS FINAL: <răspuns>. '
          'Pentru matematică arată calculele și verificarea, apoi scrie pe o linie '
          'separată RĂSPUNS FINAL: <valoare>.')


def dump(path, obj):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False))
    temporary.replace(path)


def pinned_snapshot(model, cache, revision=None):
    # Сохраняем неизменяемый коммит источника; trust_remote_code остаётся выключенным.
    info = HfApi().model_info(model, revision=revision)
    sha = info.sha
    path = snapshot_download(model, revision=sha, cache_dir=cache,
                             allow_patterns=['*.json', '*.safetensors', '*.model', '*.txt', '*.jinja'])
    return path, sha


def load(path):
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(path, local_files_only=True,
                trust_remote_code=False, torch_dtype=torch.float32).eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer, model


class TokenClock(StoppingCriteria):
    def __init__(self):
        self.times = []

    def __call__(self, input_ids, scores, **kwargs):
        self.times.append(time.perf_counter())
        return False


def generate(model, tokenizer, question, mode, seed, max_tokens):
    torch.manual_seed(seed)
    encoded = tokenizer.apply_chat_template(
        [{'role':'system','content':SYSTEM}, {'role':'user','content':question}],
        add_generation_prompt=True, return_tensors='pt', return_dict=True)
    clock = TokenClock()
    options = dict(max_new_tokens=max_tokens, pad_token_id=tokenizer.pad_token_id,
                   do_sample=mode == 'sampled', stopping_criteria=StoppingCriteriaList([clock]))
    if mode == 'sampled':
        options.update(temperature=0.2, top_p=0.9, repetition_penalty=1.1)
    # Явно задаём жадный режим без наследования параметров случайной выборки.
    else:
        options.update(temperature=None, top_p=None, top_k=None, repetition_penalty=1.0)
    start = time.perf_counter()
    cpu_start = time.process_time()
    with torch.inference_mode():
        output = model.generate(**encoded, **options)
    end = time.perf_counter()
    tokens = output[0, encoded['input_ids'].shape[-1]:]
    count = len(tokens)
    # Исключаем обработку входа из скорости декодирования; для одного токена скорость не определена.
    decode_s = clock.times[-1]-clock.times[0] if len(clock.times) > 1 else None
    eos = model.generation_config.eos_token_id
    eos = eos if isinstance(eos, list) else [eos]
    return dict(answer=tokenizer.decode(tokens, skip_special_tokens=True),
        prompt_tokens=encoded['input_ids'].shape[-1], generated_tokens=count,
        ttft_ms=(clock.times[0]-start)*1000 if clock.times else None,
        latency_ms=(end-start)*1000, cpu_seconds=time.process_time()-cpu_start,
        decode_tps=(count-1)/decode_s if decode_s and count > 1 else None,
        end_to_end_tps=count/(end-start),
        truncated=bool(count >= max_tokens and int(tokens[-1]) not in eos))


def perplexity(model, tokenizer, text, context=512, stride=256):
    ids = tokenizer(text, return_tensors='pt', add_special_tokens=False)['input_ids']
    length = ids.shape[-1]
    if length < 2:
        return dict(nll=None, targets=0, ppl=None, bits_per_byte=None)
    total_nll, total = 0., 0
    for start, end, first_target in target_windows(length, context, stride):
        chunk = ids[:, start:end]
        labels = chunk.clone()
        labels[:, :first_target-start] = -100
        count = int((labels[:, 1:] != -100).sum())
        with torch.inference_mode():
            loss = model(chunk, labels=labels).loss
        total_nll += float(loss) * count
        total += count
    mean = total_nll/total
    return dict(nll=total_nll, targets=total, ppl=math.exp(mean) if mean < 700 else None,
                bits_per_byte=total_nll / math.log(2) / len(text.encode('utf-8')))


class MemoryMonitor:
    def __init__(self):
        self.peak = psutil.Process().memory_info().rss
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.sample, daemon=True)

    def sample(self):
        while not self.stop.wait(.05):
            self.peak = max(self.peak, psutil.Process().memory_info().rss)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repetitions', type=int, default=3)
    parser.add_argument('--max-new-tokens', type=int, default=160)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--cache', default=str(ROOT/'.cache/huggingface'))
    parser.add_argument('--run-id', default=time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()))
    parser.add_argument('--replay-manifest', type=Path,
                        help='Использовать точные версии моделей из предыдущего манифеста')
    args = parser.parse_args()
    if args.repetitions < 1 or not 1 <= args.threads <= 64:
        parser.error('Недопустимое число повторов или потоков')
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    revisions = {}
    if args.replay_manifest:
        previous = json.loads(args.replay_manifest.read_text())
        revisions = {model:previous['artifacts'][model]['revision'] for model in MODELS}
    out = ROOT/'results/runs'/args.run_id
    out.mkdir(parents=True, exist_ok=False)
    questions = json.loads((ROOT/'data/questions.json').read_text())
    corpus = (ROOT/'data/corpus.txt').read_text()
    arguments = vars(args).copy()
    arguments['replay_manifest'] = str(args.replay_manifest) if args.replay_manifest else None
    manifest = dict(status='running', arguments=arguments, models=MODELS,
        system_prompt=SYSTEM, python=platform.python_version(), torch=torch.__version__,
        platform=platform.platform(), cpu=platform.processor(),
        logical_cpus=psutil.cpu_count(), ram_bytes=psutil.virtual_memory().total,
        gpu_used=False, dtype='float32', seed=20261030,
        dataset_sha256=hashlib.sha256((ROOT/'data/questions.json').read_bytes()).hexdigest(),
        corpus_sha256=hashlib.sha256((ROOT/'data/corpus.txt').read_bytes()).hexdigest(),
        started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), artifacts={})
    dump(out/'manifest.json', manifest)
    rows, resources, pinned = [], [], {}
    try:
        for model_id in MODELS:
            path, sha = pinned_snapshot(model_id, args.cache, revisions.get(model_id))
            pinned[model_id] = (path, sha)
            manifest['artifacts'][model_id] = {'revision':sha}
            dump(out/'manifest.json', manifest)
            with MemoryMonitor() as memory:
                started = time.perf_counter()
                tokenizer, model = load(path)
                load_s = time.perf_counter()-started
                parameters = sum(p.numel() for p in model.parameters())
                # Прогреваем оба режима и исключаем эти запросы из всех агрегатов.
                for mode in ['greedy', 'sampled']:
                    generate(model, tokenizer, 'Cât este 1 + 1?', mode, 123, 8)
                tasks = [(item, mode, rep) for item in questions
                         for mode in ['greedy','sampled'] for rep in range(args.repetitions)]
                random.Random(20261030).shuffle(tasks)
                for item, mode, rep in tasks:
                    seed = 20261030 + rep
                    answer = generate(model, tokenizer, item['question'], mode, seed, args.max_new_tokens)
                    scores = (math_scores(answer['answer'], item['value'])
                              if item['category']=='matematica'
                              else factual(answer['answer'], item['answers']))
                    row = dict(model=model_id, revision=sha, id=item['id'], category=item['category'],
                               mode=mode, repetition=rep, seed=seed, **answer, **scores)
                    rows.append(row)
                    with (out/'responses.jsonl').open('a') as f:
                        f.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
                    print(f"{model_id} {len(rows)}: задание {item['id']} {mode}, повтор {rep}", flush=True)
                native = perplexity(model, tokenizer, corpus)
                resource = dict(model=model_id, revision=sha, parameters=parameters,
                    weight_bytes=sum(f.stat().st_size for f in Path(path).glob('*.safetensors')),
                    load_seconds=load_s, corpus_native=native)
                del model, tokenizer
                import gc
                gc.collect()
            resource['sampled_peak_rss_bytes'] = memory.peak
            resources.append(resource)
            dump(out/'resources.json', resources)
        # Все ответы оценивает одна референсная модель с одним токенизатором.
        # Это референсная PPL, а не псевдоперплексия маскированной модели или собственная PPL участника.
        tokenizer, model = load(pinned[MODELS[0]][0])
        for row in rows:
            row['reference_lm'] = MODELS[0]
            row['reference_revision'] = pinned[MODELS[0]][1]
            row['reference_ppl'] = perplexity(model, tokenizer, row['answer'])
        dump(out/'scored.json', rows)
        with (out/'manual_review.csv').open('w') as f:
            writer = csv.writer(f)
            writer.writerow(['model','mode','id','repetition','answer','final_correct_review',
                             'steps_valid_0_1','complete_0_1','note'])
            for r in rows:
                if r['category']=='matematica':
                    writer.writerow([r[k] for k in ['model','mode','id','repetition','answer']]+['','','',''])
        manifest['status']='complete'
    except Exception as e:
        manifest['status']='failed'
        manifest['error_type']=type(e).__name__
        # Трассировка содержит диагностику библиотек; окружение и учётные данные не выводятся.
        traceback.print_exc()
        raise
    finally:
        manifest['finished_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        dump(out/'manifest.json', manifest)
    print(f'Завершено: {out}', flush=True)


if __name__ == '__main__':
    main()

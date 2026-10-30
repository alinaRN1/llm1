"""Локальные CPU-эксперименты на GGUF Docker AI, закреплённых SHA-256.

Локально измеряются генерация, время токенов и условное правдоподобие заданного текста.
Отдельные процессы изолируют память моделей. Внешний API генерации не используется.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import subprocess
import sys
import threading
import time

import llama_cpp
from llama_cpp import Llama
from llama_cpp.llama_chat_format import Jinja2ChatFormatter
import numpy as np
import psutil

from metrics import factual, math_scores, target_windows
from pull_oci import digest_file

ROOT=Path(__file__).resolve().parents[1]
SYSTEM=('Ești un asistent educațional. Răspunde în limba română. '
        'Pentru istorie și limba română scrie doar RĂSPUNS FINAL: <răspuns>. '
        'Pentru matematică arată calculele și verificarea, apoi scrie pe o linie '
        'separată RĂSPUNS FINAL: <valoare>.')


def dump(path,obj):
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False))
    temp.replace(path)


class MemoryMonitor:
    def __init__(self):
        self.peak=psutil.Process().memory_info().rss
        self.stop=threading.Event()
        self.thread=threading.Thread(target=self.sample,daemon=True)

    def sample(self):
        while not self.stop.wait(.05):
            self.peak=max(self.peak,psutil.Process().memory_info().rss)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self,*args):
        self.stop.set()
        self.thread.join()


def load(spec,threads):
    path=ROOT/spec['model_path']
    layer=next(x for x in spec['layers'] if x['file']=='model.gguf')
    if digest_file(path)!=layer['digest']:
        raise ValueError('Не пройдена проверка целостности модели')
    return Llama(model_path=str(path),n_ctx=1024,n_batch=512,n_threads=threads,
                 n_threads_batch=threads,n_gpu_layers=0,logits_all=True,
                 use_mmap=True,seed=20261030,verbose=False)


def chat_prompt(llm,question):
    template=llm.metadata.get('tokenizer.chat_template')
    if not template:
        raise ValueError('В модели отсутствует официальный шаблон диалога')
    eos=llm.detokenize([llm.token_eos()],special=True).decode('utf-8')
    bos=llm.detokenize([llm.token_bos()],special=True).decode('utf-8') if llm.token_bos()>=0 else ''
    formatter=Jinja2ChatFormatter(template=template,bos_token=bos,eos_token=eos,
                                  add_generation_prompt=True)
    result=formatter(messages=[{'role':'system','content':SYSTEM},
                                {'role':'user','content':question}])
    return llm.tokenize(result.prompt.encode('utf-8'),add_bos=False,special=True)


def generate(llm,question,mode,seed,max_new_tokens):
    prompt=chat_prompt(llm,question)
    if len(prompt)+max_new_tokens>llm.n_ctx():
        raise ValueError('Запрос и генерация превышают контекст; скрытое усечение запрещено')
    llm.reset()  # Также запрещает повторное использование префикса и KV-кеша между вопросами.
    llm.set_seed(seed)
    tokens,times=[],[]
    start=time.perf_counter()
    cpu_start=time.process_time()
    generator=llm.generate(prompt,top_k=0,top_p=.9 if mode=='sampled' else 1.,min_p=0.,
        temp=.2 if mode=='sampled' else 0.,repeat_penalty=1.1 if mode=='sampled' else 1.,
        reset=True)
    stopped=False
    try:
        for token in generator:
            times.append(time.perf_counter())
            tokens.append(token)
            if llama_cpp.llama_token_is_eog(llm._model.vocab,token):
                stopped=True
                break
            if len(tokens)>=max_new_tokens:
                break
    finally:
        generator.close()
    end=time.perf_counter()
    cpu=time.process_time()-cpu_start
    answer_tokens=tokens[:-1] if stopped else tokens
    answer=llm.detokenize(answer_tokens).decode('utf-8',errors='replace')
    count=len(tokens)
    duration=times[-1]-times[0] if count>1 else None
    return dict(answer=answer,prompt_tokens=len(prompt),generated_tokens=count,
        output_utf8_bytes=len(answer.encode('utf-8')),ttft_ms=(times[0]-start)*1000 if times else None,
        latency_ms=(end-start)*1000,cpu_seconds=cpu,
        decode_tps=(count-1)/duration if duration else None,
        end_to_end_tps=count/(end-start),truncated=not stopped,finish_reason='eog' if stopped else 'length')


def perplexity(llm,text,context=512,stride=256):
    tokens=llm.tokenize(text.encode('utf-8'),add_bos=False,special=False)
    if len(tokens)<2:
        return dict(nll=None,targets=0,ppl=None,bits_per_byte=None)
    nll,total=0.,0
    for start,end,first in target_windows(len(tokens),context,stride):
        llm.reset()
        llm.eval(tokens[start:end])
        for target in range(first,end):
            logits=llm.scores[target-start-1,:]
            maximum=float(logits.max())
            log_z=maximum+math.log(float(np.exp(logits-maximum,dtype=np.float64).sum()))
            nll+=log_z-float(logits[tokens[target]])
            total+=1
    if not math.isfinite(nll):
        raise ValueError('Получено нечисловое или бесконечное значение потери')
    return dict(nll=nll,targets=total,ppl=math.exp(nll/total) if nll/total<700 else None,
                bits_per_byte=nll/math.log(2)/len(text.encode('utf-8')))


def worker(run,index):
    manifest=json.loads((run/'manifest.json').read_text())
    args=manifest['arguments']
    spec=manifest['artifacts'][index]
    questions=json.loads((ROOT/'data/questions.json').read_text())
    with MemoryMonitor() as memory:
        started=time.perf_counter()
        llm=load(spec,args['threads'])
        load_s=time.perf_counter()-started
        metadata=dict(llm.metadata)
        # Прогреваем оба режима; эти ответы не входят в измеренные результаты.
        for mode in ['greedy','sampled']:
            generate(llm,'Cât este 1 + 1?',mode,123,8)
        tasks=[(item,mode,rep) for item in questions for mode in ['greedy','sampled']
               for rep in range(args['repetitions'])]
        random.Random(20261030).shuffle(tasks)
        path=run/f'responses-{index}.jsonl'
        with path.open('x') as file:
            for counter,(item,mode,rep) in enumerate(tasks,1):
                answer=generate(llm,item['question'],mode,20261030+rep,args['max_new_tokens'])
                scores=(math_scores(answer['answer'],item['value']) if item['category']=='matematica'
                        else factual(answer['answer'],item['answers']))
                row=dict(model=spec['model'],revision=spec['manifest_digest'],id=item['id'],
                    category=item['category'],mode=mode,repetition=rep,seed=20261030+rep,**answer,**scores)
                file.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
                file.flush()
                print(f"{spec['model']}: {counter}/{len(tasks)}, задание {item['id']} {mode}",flush=True)
        native=perplexity(llm,(ROOT/'data/corpus.txt').read_text())
        llm.close()
    gguf=next(x for x in spec['layers'] if x['file']=='model.gguf')
    dump(run/f'resource-{index}.json',dict(model=spec['model'],revision=spec['manifest_digest'],
        parameters_nominal=spec['config'].get('config',{}).get('parameters',metadata.get('general.size_label')),
        weight_bytes=gguf['size'],
        load_seconds=load_s,sampled_peak_rss_bytes=memory.peak,corpus_native=native,
        gguf_metadata=metadata,process_isolated=True))


def reference_worker(run):
    manifest=json.loads((run/'manifest.json').read_text())
    rows=[json.loads(line) for i in range(3) for line in (run/f'responses-{i}.jsonl').read_text().splitlines()]
    spec=manifest['artifacts'][0]
    llm=load(spec,manifest['arguments']['threads'])
    cache={}
    for i,row in enumerate(rows,1):
        text=row['answer']
        if text not in cache:
            cache[text]=perplexity(llm,text)
        row['reference_lm']=spec['model']
        row['reference_revision']=spec['manifest_digest']
        row['reference_ppl']=cache[text]
        if i%20==0:
            print(f'Референсная PPL: {i}/{len(rows)}',flush=True)
    llm.close()
    dump(run/'scored.json',rows)
    with (run/'responses.jsonl').open('w') as file:
        for row in rows:
            file.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
    with (run/'manual_review.csv').open('w') as file:
        writer=csv.writer(file)
        writer.writerow(['model','mode','id','repetition','answer','final_correct_review',
                         'steps_valid_0_1','complete_0_1','note'])
        for row in rows:
            if row['category']=='matematica':
                writer.writerow([row[k] for k in ['model','mode','id','repetition','answer']]+['','','',''])


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-id',default=time.strftime('%Y%m%dT%H%M%SZ',time.gmtime()))
    parser.add_argument('--repetitions',type=int,default=3)
    parser.add_argument('--threads',type=int,default=4)
    parser.add_argument('--max-new-tokens',type=int,default=160)
    parser.add_argument('--worker',type=int)
    parser.add_argument('--reference-worker',action='store_true')
    parser.add_argument('--resume',action='store_true',help='Повторно использовать совпадающие завершённые блоки после диагностики сбоя')
    args=parser.parse_args()
    run=ROOT/'results/runs'/args.run_id
    if args.worker is not None:
        return worker(run,args.worker)
    if args.reference_worker:
        return reference_worker(run)
    if args.repetitions<1 or not 1<=args.threads<=32 or args.max_new_tokens<1:
        parser.error('Недопустимые параметры эксперимента')
    models=json.loads((ROOT/'data/models_oci.lock.json').read_text())
    if len(models)!=3:
        raise ValueError('Требуются ровно три проверенных артефакта моделей')
    run.mkdir(parents=True,exist_ok=args.resume)
    cpu_info=Path('/proc/cpuinfo').read_text()
    cpu=next((line.split(':',1)[1].strip() for line in cpu_info.splitlines() if line.startswith('model name')),'unknown')
    manifest=dict(status='running',backend='llama-cpp-python',backend_version=llama_cpp.__version__,
        arguments={'repetitions':args.repetitions,'threads':args.threads,'max_new_tokens':args.max_new_tokens},
        models=[m['model'] for m in models],artifacts=models,system_prompt=SYSTEM,
        python=platform.python_version(),platform=platform.platform(),cpu=cpu,
        logical_cpus=psutil.cpu_count(),ram_bytes=psutil.virtual_memory().total,
        cpu_quota=(Path('/sys/fs/cgroup/cpu.max').read_text().strip()
                   if Path('/sys/fs/cgroup/cpu.max').is_file() else 'недоступна'),
        gpu_used=False,weight_format='GGUF F16',context=1024,batch=512,logits_all=True,
        prefix_cache=False,seed=20261030,model_order_randomized=False,
        dataset_sha256=hashlib.sha256((ROOT/'data/questions.json').read_bytes()).hexdigest(),
        corpus_sha256=hashlib.sha256((ROOT/'data/corpus.txt').read_bytes()).hexdigest(),
        started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
    reused=[]
    if args.resume:
        previous=json.loads((run/'manifest.json').read_text())
        if previous['status']!='failed':
            raise ValueError('Возобновление допустимо только после остановленного неудачного запуска')
        for key in ['arguments','system_prompt','dataset_sha256','corpus_sha256','backend_version']:
            if previous[key]!=manifest[key]:
                raise ValueError(f'Настройки возобновления не совпадают: {key}')
        dump(run/'attempt-1-manifest.json',previous)
        manifest['started_utc']=previous['started_utc']
        manifest['resumed_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
        for i,spec in enumerate(models):
            resource=run/f'resource-{i}.json'
            responses=run/f'responses-{i}.jsonl'
            if resource.exists() and responses.exists():
                metadata=json.loads(resource.read_text())
                rows=[json.loads(s) for s in responses.read_text().splitlines()]
                if (metadata['model']==spec['model'] and metadata['revision']==spec['manifest_digest']
                    and len(rows)==40*args.repetitions and
                    len({(r['id'],r['mode'],r['repetition']) for r in rows})==len(rows)
                    and all(r['model']==spec['model'] and r['revision']==spec['manifest_digest'] for r in rows)):
                    reused.append(i)
                else:
                    raise ValueError('Существующий блок нельзя безопасно использовать повторно')
        manifest['reused_worker_blocks']=reused
    dump(run/'manifest.json',manifest)
    try:
        for i in range(3):
            if i in reused:
                print(f'Повторно используем проверенный завершённый блок {i}',flush=True)
                continue
            subprocess.run([sys.executable,__file__,'--run-id',args.run_id,'--worker',str(i)],check=True)
        resources=[json.loads((run/f'resource-{i}.json').read_text()) for i in range(3)]
        dump(run/'resources.json',resources)
        subprocess.run([sys.executable,__file__,'--run-id',args.run_id,'--reference-worker'],check=True)
        manifest['status']='complete'
    except Exception as error:
        manifest['status']='failed'
        manifest['error_type']=type(error).__name__
        raise
    finally:
        manifest['finished_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
        dump(run/'manifest.json',manifest)
    print(f'ЗАВЕРШЕНО: {run}',flush=True)


if __name__=='__main__':
    main()

"""Скачивание публичных моделей Docker AI через унаследованный HTTPS-прокси.

Все манифесты и файлы проверяются SHA-256; токены авторизации не сохраняются.
Контейнеры не запускаются, Docker daemon не требуется.
"""
import argparse
import hashlib
import json
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
SPECS = [('ai/qwen2.5', '0.5B-F16'), ('ai/smollm2', '360m-f16'), ('ai/llama3.2', '1B-F16')]
ACCEPT = 'application/vnd.oci.image.manifest.v1+json'


def digest_file(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):
            h.update(block)
    return 'sha256:'+h.hexdigest()


def public_token(session, repo):
    response=session.get('https://auth.docker.io/token',
        params={'service':'registry.docker.io','scope':f'repository:{repo}:pull'},timeout=60)
    response.raise_for_status()
    return response.json()['token']


def pull(repo, tag, destination, expected_manifest=None):
    session=requests.Session()  # Наследуются прокси и доверенные сертификаты; проверка TLS включена.
    headers={'Authorization':'Bearer '+public_token(session,repo), 'Accept':ACCEPT}
    ref=expected_manifest or tag
    response=session.get(f'https://registry-1.docker.io/v2/{repo}/manifests/{ref}',
                         headers=headers,timeout=60)
    response.raise_for_status()
    digest='sha256:'+hashlib.sha256(response.content).hexdigest()
    expected=expected_manifest or response.headers.get('Docker-Content-Digest')
    if expected and expected!=digest:
        raise ValueError('Не совпала контрольная сумма манифеста реестра')
    manifest=response.json()
    destination.mkdir(parents=True,exist_ok=True)
    (destination/'oci_manifest.json').write_bytes(response.content)
    metadata={'model':repo+':'+tag,'registry':'registry-1.docker.io','manifest_digest':digest,
              'layers':[]}
    layers=[manifest['config'],*manifest['layers']]
    for i,layer in enumerate(layers):
        media=layer['mediaType']
        annotation=layer.get('annotations',{}).get('org.cncf.model.filepath','')
        name='model.gguf' if 'gguf' in media or annotation.endswith('.gguf') else ('config.json' if i==0 else f'license-{i}.txt')
        target=destination/name
        if not (target.exists() and target.stat().st_size==layer['size'] and digest_file(target)==layer['digest']):
            temporary=target.with_suffix(target.suffix+'.partial')
            headers={'Authorization':'Bearer '+public_token(session,repo)}
            with session.get(f"https://registry-1.docker.io/v2/{repo}/blobs/{layer['digest']}",
                             headers=headers,stream=True,timeout=(60,180)) as response:
                response.raise_for_status()
                checksum=hashlib.sha256()
                size=0
                with temporary.open('wb') as f:
                    for chunk in response.iter_content(8*1024*1024):
                        f.write(chunk)
                        checksum.update(chunk)
                        size+=len(chunk)
            if 'sha256:'+checksum.hexdigest()!=layer['digest'] or size!=layer['size']:
                temporary.unlink(missing_ok=True)
                raise ValueError('Не совпали контрольная сумма или размер скачанного файла')
            temporary.replace(target)
        metadata['layers'].append({'file':name,'digest':layer['digest'],'size':layer['size'],'media_type':media})
        print(f'{repo}:{tag}: проверен {name} ({layer["size"]} байт)',flush=True)
    if not (destination/'model.gguf').exists():
        raise ValueError('Пакет OCI не содержит модель GGUF')
    metadata['model_path']=str((destination/'model.gguf').relative_to(ROOT))
    metadata['config']=json.loads((destination/'config.json').read_text(encoding="utf-8"))
    metadata['annotations']=manifest.get('annotations',{})
    (destination/'provenance.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2), encoding="utf-8")
    return metadata


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--locked',type=Path,help='Скачать неизменяемые версии по существующему файлу фиксации')
    args=parser.parse_args()
    locked=json.loads(args.locked.read_text(encoding="utf-8")) if args.locked else []
    expected={m['model']:m['manifest_digest'] for m in locked}
    models=[]
    for repo,tag in SPECS:
        destination=ROOT/'.cache/oci'/repo.replace('/','_')/tag
        models.append(pull(repo,tag,destination,expected.get(repo+':'+tag)))
        (ROOT/'data/models_oci.lock.json').write_text(json.dumps(models,ensure_ascii=False,indent=2), encoding="utf-8")


if __name__=='__main__':
    main()

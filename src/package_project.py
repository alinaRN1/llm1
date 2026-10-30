"""Сборка проверяемого архива проекта для скачивания с GitHub."""
import hashlib
from pathlib import Path
import zipfile

ROOT=Path(__file__).resolve().parents[1]
ROOT_FILES={'.gitignore','README.md','install.sh','laborator.ipynb','laborator_colab.ipynb',
            'requirements.txt','requirements.lock.txt'}
ROOT_DIRS={'data','licenses','reports','results','src','tests'}
EXCLUDED={'__pycache__','.pytest_cache','.cache','.git','.venv'}


def package():
    files=[]
    for path in sorted(ROOT.rglob('*')):
        relative=path.relative_to(ROOT)
        if relative.parts[0] not in ROOT_DIRS and str(relative) not in ROOT_FILES:
            continue
        if any(part in EXCLUDED for part in relative.parts):
            continue
        if relative.parts[:2]==('results','runs'):
            continue
        if path.is_symlink():
            raise ValueError(f'Символическая ссылка не включается в архив: {relative}')
        if path.is_file():
            files.append((path,relative))
    checksums=''.join(hashlib.sha256(path.read_bytes()).hexdigest()+'  '+str(relative)+'\n'
                      for path,relative in files)
    checksum_file=ROOT/'checksums.sha256'
    checksum_file.write_text(checksums,encoding='utf-8')
    files.append((checksum_file,Path('checksums.sha256')))
    output=ROOT/'downloads/LL1_masterat_ru.zip'
    output.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(output,'w',zipfile.ZIP_DEFLATED) as archive:
        for path,relative in files:
            archive.write(path,Path('LL1_masterat')/relative)
    with zipfile.ZipFile(output) as archive:
        if archive.testzip() is not None:
            raise ValueError('Архив не прошёл проверку целостности')
        for line in checksums.splitlines():
            expected,name=line.split('  ',1)
            if hashlib.sha256(archive.read('LL1_masterat/'+name)).hexdigest()!=expected:
                raise ValueError(f'Не совпала контрольная сумма: {name}')
        if any('/downloads/' in name for name in archive.namelist()):
            raise ValueError('Архив не должен включать папку собственных сборок')
    digest=hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_name(output.name+'.sha256').write_text(digest+'  '+output.name+'\n',encoding='utf-8')
    print(f'Готов архив: {output}; файлов: {len(files)}; размер: {output.stat().st_size} байт')


if __name__=='__main__':
    package()

"""Консервативная оценка без угадывания последнего числа и исполнения кода."""
import ast
import math
import operator
import re
import unicodedata
from fractions import Fraction


def normalize(text):
    text = ''.join(c for c in unicodedata.normalize('NFD', text.lower())
                   if unicodedata.category(c) != 'Mn')
    return ' '.join(re.findall(r'\w+', text))


def final_answer(text):
    matches = list(re.finditer(r'raspuns\s+final\s*:', strip_accents(text), re.I))
    if not matches:
        return None
    # Удаление диакритики не сдвигает индексы предсоставленных румынских букв Unicode.
    return strip_accents(text)[matches[-1].end():].strip().split('\n')[0].strip()


def strip_accents(text):
    return ''.join(c for c in unicodedata.normalize('NFD', text)
                   if unicodedata.category(c) != 'Mn')


def factual(text, aliases):
    answer = final_answer(text)
    if answer is None:
        return {'exact': 0, 'partial': 0.0, 'format_ok': False}
    a = normalize(answer)
    tokens = set(a.split())
    partial = max((len(tokens & set(normalize(b).split())) /
                   max(1, len(tokens | set(normalize(b).split()))) for b in aliases))
    return {'exact': int(a in [normalize(b) for b in aliases]),
            'partial': partial, 'format_ok': True}


def numeric(text):
    answer = final_answer(text)
    if answer is None:
        return None
    answer = re.sub(r'^x\s*=\s*', '', answer, flags=re.I).strip().rstrip('.;')
    if not re.fullmatch(r'[+-]?\d+(?:[.,]\d+)?(?:\s*/\s*[+-]?\d+)?', answer):
        return None
    try:
        return Fraction(answer.replace(',', '.').replace(' ', ''))
    except (ValueError, ZeroDivisionError):
        return None


OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
       ast.Div: operator.truediv, ast.Pow: operator.pow}


def arithmetic(expression):
    """Ограниченный AST-проверяющий числовых равенств без eval и sympify."""
    if len(expression) > 120:
        raise ValueError('Слишком длинное выражение')
    expression = expression.replace('×', '*').replace('÷', '/').replace('²', '**2')
    expression = expression.replace('^', '**').replace(',', '.').replace('−', '-')
    nodes = ast.parse(expression.strip(), mode='eval')
    if len(list(ast.walk(nodes))) > 50:
        raise ValueError('Слишком сложное выражение')

    def visit(node):
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return Fraction(str(node.value))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            return (-1 if isinstance(node.op, ast.USub) else 1) * visit(node.operand)
        if isinstance(node, ast.BinOp) and type(node.op) in OPS:
            a, b = visit(node.left), visit(node.right)
            if isinstance(node.op, ast.Pow) and (b.denominator != 1 or abs(b) > 12):
                raise ValueError('Недопустимая степень')
            value = OPS[type(node.op)](a, b)
            if abs(value) > 10**15:
                raise ValueError('Слишком большое значение')
            return value
        raise ValueError('Неподдерживаемая арифметическая операция')
    return visit(nodes)


def equation_audit(text):
    """Автоматически проверяются числовые равенства; символьные и текстовые шаги требуют разбора."""
    checked = []
    for line in text.splitlines():
        line = re.sub(r'^\s*(?:Pas(?:ul)?\s*\d*[:.)-]?|\d+[.)])\s*', '', line, flags=re.I)
        # Принимаются только целые числовые строки: фрагмент символьного уравнения
        # вида '2x + 5 = 17' нельзя ошибочно читать как '5 = 17'.
        for match in re.finditer(r'^[\s]*([\d\s+*/().,²×÷^−-]+(?:=[\d\s+*/().,²×÷^−-]+)+)[\s]*$', line):
            parts = match.group(1).strip().rstrip('.;,').split('=')
            try:
                values = [arithmetic(p) for p in parts]
                checked.append({'equation': match.group(1).strip(),
                                'valid': all(x == values[0] for x in values[1:])})
            except (ValueError, SyntaxError, ZeroDivisionError, OverflowError):
                continue
    return checked


def math_scores(text, value):
    number = numeric(text)
    audit = equation_audit(text)
    return {'math_correct': int(number is not None and number == Fraction(value)),
            'format_ok': final_answer(text) is not None,
            'parsed_value': str(number) if number is not None else None,
            'equations_checked': len(audit),
            'equations_valid': sum(x['valid'] for x in audit),
            'equation_audit': audit,
            'manual_steps_status': 'pending'}


def wilson(k, n, z=1.959963984540054):
    k,n=int(k),int(n)
    if not n:
        return [None, None]
    if not 0<=k<=n:
        raise ValueError('Число успехов должно находиться между нулём и n')
    p = k / n
    denominator = 1 + z*z/n
    middle = (p + z*z/(2*n)) / denominator
    radius = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / denominator
    return [0.0 if k==0 else float(max(0,middle-radius)),
            1.0 if k==n else float(min(1,middle+radius))]


def target_windows(length, context=512, stride=256):
    """Каждый целевой индекс от 1 до length−1 учитывается ровно один раз."""
    if not 1 <= stride < context:
        raise ValueError('Требуется 1 <= stride < context')
    previous_end = 1
    for end in range(min(context, length), length + stride, stride):
        end = min(end, length)
        start = max(0, end-context)
        yield start, end, previous_end
        previous_end = end
        if end == length:
            break

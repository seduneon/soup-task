"""Verifiable rewards. Correctness and format are separate metrics (H9): a rising reward must come from
correctness, not from learning the format alone or gaming the parser.

math_boxed: the completion must contain a final answer in \\boxed{...}.
  - the LAST balanced \\boxed{} is the answer; several boxed answers that disagree score 0 (hedging is not
    rewarded); no box scores 0 (no fallback to "the last number", which is easy to game);
  - numbers are compared exactly as fractions after normalisation (1/2 == 0.5 == \\frac{1}{2}, 1,000 == 1000);
    currency, percent, degree signs, a trailing unit word ("18 dollars", "10 \\text{ hours}") and a leading
    "x =" are stripped; \\frac12 and -\\frac{1}{14} (MATH style) parse; anything else is compared as a
    normalised string (the MATH source keeps only items whose gold answer is a number, see prepare_data.py).
"""
import ast
import re
from fractions import Fraction

PROMPT_SUFFIXES = {
    "cot": "\nPut your final answer within \\boxed{}.",
    # answer-only: ~10-token completions, so max_new can be ~32 and decode stops dominating the step; the
    # CoT style at max_new 512 truncated 41 % of MATH 3-5 completions and turned the reward into a length test
    "direct": "\nAnswer with only the final number in \\boxed{}. Do not explain or show any work.",
    "plain": "",  # the item's prompt already carries the full instruction (countdown)
}
PROMPT_SUFFIX = PROMPT_SUFFIXES["cot"]


def boxed_all(text):
    out, i = [], 0
    while (j := text.find("\\boxed{", i)) != -1:
        k, depth = j + len("\\boxed{"), 1
        while k < len(text) and depth:
            depth += {"{": 1, "}": -1}.get(text[k], 0)
            k += 1
        if depth:
            break  # unbalanced (truncated) box: ignored
        out.append(text[j + len("\\boxed{"):k - 1])
        i = k
    return out


def normalize(s):
    s = s.strip()
    s = s.replace("\\$", "").replace("$", "").replace("\\%", "").replace("%", "").replace("{,}", ",")
    s = re.sub(r"\^\{?\\circ\}?", "", s)                       # degrees
    s = re.sub(r"\\(?:text|mathrm|textbf)\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\[,;! ]", " ", s)
    s = s.replace("\\left", "").replace("\\right", "").replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    s = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"(\1)/(\2)", s)
    s = re.sub(r"\\frac\s*(\d)\s*(\d)", r"(\1)/(\2)", s)       # MATH writes \frac12
    s = re.sub(r"\\frac\s*\{([^{}]+)\}\s*(\d)", r"(\1)/(\2)", s)   # and \frac{10}3
    s = re.sub(r"^[a-zA-Z]\s*=\s*", "", s.strip())               # "x = 5" -> "5"
    m = re.fullmatch(r"(-?[\d.,/()]*\d[\d.,/()]*)\s*[a-zA-Z][a-zA-Z .]*", s.strip())  # "18 dollars", "10 hours"
    if m:
        s = m[1]
    s = re.sub(r"\s+", "", s).rstrip(".")
    if re.fullmatch(r"-?\d{1,3}(,\d{3})+(\.\d+)?", s):
        s = s.replace(",", "")
    return s


def as_number(s):
    if s.startswith("-(") or s.startswith("-\\("):  # "-(1)/(14)", from "-\frac{1}{14}"
        n = as_number(s[1:])
        return None if n is None else -n
    m = re.fullmatch(r"\(?(-?\d+(?:\.\d+)?)\)?(?:/\(?(-?\d+(?:\.\d+)?)\)?)?", s)
    if not m:
        return None
    try:
        num = Fraction(m[1])
        return num / Fraction(m[2]) if m[2] else num
    except (ZeroDivisionError, ValueError):
        return None


def same_answer(a, b):
    a, b = normalize(a), normalize(b)
    na, nb = as_number(a), as_number(b)
    return na == nb if na is not None and nb is not None else a == b


def math_boxed(completion, answer):
    """-> (reward, info). reward 1.0 iff the single final boxed answer equals `answer`."""
    boxes = boxed_all(completion)
    info = {"has_box": bool(boxes), "n_boxes": len(boxes)}
    if not boxes:
        return 0.0, info
    final = boxes[-1]
    if any(not same_answer(b, final) for b in boxes[:-1]):
        info["conflicting_boxes"] = True
        return 0.0, info
    return float(same_answer(final, answer)), info


def toy_ascii(completion, answer):
    """Test-only reward for random models (real rewards are ~always 0 there, so GRPO would never update):
    fraction of ASCII alphanumeric characters. Varies within a group, so advantages are non-zero."""
    return sum(ch.isascii() and ch.isalnum() for ch in completion) / max(len(completion), 1), {"has_box": False}


_ANSWER = re.compile(r"<answer>(.*?)</answer>", re.S)
_EXPR_CHARS = re.compile(r"[0-9+\-*/() ]+")
_OPS = {ast.Add: lambda a, b: a + b, ast.Sub: lambda a, b: a - b, ast.Mult: lambda a, b: a * b,
        ast.Div: lambda a, b: a / b}


def _eval_expr(node):
    """Exact evaluation (Fraction) of a binary +,-,*,/ tree over integer literals; anything else raises."""
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval_expr(node.left), _eval_expr(node.right))
    if isinstance(node, ast.Constant) and type(node.value) is int:
        return Fraction(node.value)
    raise ValueError(f"not allowed: {ast.dump(node)}")


def countdown(completion, answer):
    """answer = "a,b,c|target". reward 1.0 iff the LAST <answer>...</answer> is an expression that uses every
    given number exactly once (no concatenation, no reuse, no extra numbers), only binary + - * / and
    parentheses, and equals the target exactly (fractions, no float tolerance; 0 on division by zero).
    has_box = an <answer> tag is present: format compliance, kept separate from correctness (H9)."""
    nums_s, target_s = answer.split("|")
    nums, target = sorted(int(x) for x in nums_s.split(",")), Fraction(int(target_s))
    found = _ANSWER.findall(completion)
    info = {"has_box": bool(found)}
    if not found:
        return 0.0, info
    expr = found[-1].strip()
    if not _EXPR_CHARS.fullmatch(expr) or "**" in expr or "//" in expr:
        info["bad_chars"] = True
        return 0.0, info
    if sorted(int(x) for x in re.findall(r"\d+", expr)) != nums:
        info["wrong_numbers"] = True
        return 0.0, info
    try:
        value = _eval_expr(ast.parse(expr, mode="eval").body)
    except (ZeroDivisionError, ValueError, SyntaxError, RecursionError):
        info["invalid_expr"] = True
        return 0.0, info
    return float(value == target), info


_UNICODE_OPS = str.maketrans({"÷": "/", "×": "*", "−": "-"})


def normalize_ops(completion):
    """Map the Unicode operators ÷ × − to / * -. NOT part of the training reward (the prompt asks for +, -, *, /,
    so countdown() scores ÷ × as bad_chars). analyze.py uses it to measure arithmetic skill apart from operator
    spelling: the shuffled-reward control writes ÷ in 12 of 500 held-out answers, all of them correct."""
    return completion.translate(_UNICODE_OPS)


REWARDS = {"math_boxed": math_boxed, "toy_ascii": toy_ascii, "countdown": countdown}

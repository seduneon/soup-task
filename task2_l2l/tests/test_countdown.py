"""Countdown reward: correct answers pass, every known way to game the checker scores 0, solver and scorer agree."""
import pytest

from prepare_countdown import solutions
from reward import countdown

ANS = "68,98,14|61"  # 68 - 98 / 14 = 61


@pytest.mark.parametrize("text", [
    "<answer>68 - 98 / 14</answer>",
    "<answer> 68 - (98 / 14) </answer>",
    "work: 98/14 = 7, 68 - 7 = 61\n<answer>68-98/14</answer>",
    "<answer>1 + 1</answer> no wait <answer>68 - 98 / 14</answer>",  # the last tag counts
])
def test_correct(text):
    assert countdown(text, ANS)[0] == 1.0


@pytest.mark.parametrize("text,flag", [
    ("68 - 98 / 14 = 61", None),                          # no tag: format failure
    ("<answer>61</answer>", "wrong_numbers"),              # just the target
    ("<answer>68 - 98 / 14 + 0</answer>", "wrong_numbers"),  # an extra number
    ("<answer>68 - 7</answer>", "wrong_numbers"),          # a number not given
    ("<answer>6898 - 14</answer>", "wrong_numbers"),       # concatenation
    ("<answer>68 - 98 / 14 / 1 * 1</answer>", "wrong_numbers"),
    ("<answer>(68 - 98) / 14</answer>", None),             # right numbers, wrong value
    ("<answer>68 ** 1 - 98 / 14</answer>", "bad_chars"),   # power
    ("<answer>68 - 98 // 14</answer>", "bad_chars"),       # floor division
    ("<answer>68 - 98 / 14 = 61</answer>", "bad_chars"),   # '=' inside the tag
    ("<answer>-(14 - 98) - 68 + 45</answer>", "wrong_numbers"),
    ("<answer>__import__('os')</answer>", "bad_chars"),
    ("<answer>68 - 98 / 14</answer", None),                # unclosed tag
])
def test_rejected(text, flag):
    r, info = countdown(text, ANS)
    assert r == 0.0
    if flag:
        assert info.get(flag), info


def test_unary_minus_and_div_by_zero():
    assert countdown("<answer>-68 + 98 + 14</answer>", "68,98,14|44")[1].get("invalid_expr")
    r, info = countdown("<answer>5 / (3 - 3)</answer>", "5,3,3|1")
    assert r == 0.0 and info.get("invalid_expr")


def test_exact_fractions():
    assert countdown("<answer>(1 + 2) / 3</answer>", "1,2,3|1")[0] == 1.0
    assert countdown("<answer>10 / 4 * 2</answer>", "10,4,2|5")[0] == 1.0     # 2.5 * 2, no rounding needed
    assert countdown("<answer>10 / 3 * 3</answer>", "10,3,3|10")[0] == 1.0    # float would give 9.999...


def test_solver_and_scorer_agree():
    for nums, target in [((68, 98, 14), 61), ((5, 40, 83), 38), ((2, 3, 4), 24), ((1, 1, 1), 3)]:
        sols = solutions(list(nums), target)
        assert sols, (nums, target)
        ans = f"{','.join(map(str, nums))}|{target}"
        assert all(countdown(f"<answer>{s}</answer>", ans)[0] == 1.0 for s in sols)
    assert solutions([1, 1, 1], 100) == []

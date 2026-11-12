# ---- provided: the R1-Zero training template
TEMPLATE = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. "
    "The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. "
    "The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, "
    "i.e., <think> reasoning process here </think> <answer> answer here </answer>. User: {prompt}. Assistant:"
)
# ---- end provided
import re


def build_prompt(question):
    """The template with the question filled in."""
    return TEMPLATE.format(prompt=question)


def format_reward(response):
    """1.0 for exactly <think>...</think> then <answer>...</answer> (each once, non-empty, only whitespace around), else 0.0."""
    pattern = r"\s*<think>(.*?)</think>\s*<answer>(.*?)</answer>\s*"

    match = re.fullmatch(pattern, response, flags=re.DOTALL)

    if not match:
        return 0.0

    if not match.group(1).strip() or not match.group(2).strip():
        return 0.0

    tags = ["<think>", "</think>", "<answer>", "</answer>"]
    return 1.0 if all(response.count(tag) == 1 for tag in tags) else 0.0

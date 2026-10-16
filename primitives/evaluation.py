import math_verify
from vllm.outputs import CompletionOutput  # type: ignore


def evaluate_math500(
    outputs: list[CompletionOutput],
    answers: list[str],
    think_end_id: int,
) -> dict:
    golds = [math_verify.parse(f"${answer}$") for answer in answers]
    predictions = [math_verify.parse(output.text.split("</think>")[-1]) if "</think>" in output.text else None for output in outputs]
    corrects = [prediction is not None and math_verify.verify(gold, prediction) for gold, prediction in zip(golds, predictions)]
    return {
        "accuracy": sum(corrects) / len(outputs),
        "corrects": corrects,
        "think_lens": [list(output.token_ids).index(think_end_id) if think_end_id in output.token_ids else len(output.token_ids) for output in outputs],
        "output_lens": [len(output.token_ids) for output in outputs],
        "truncateds": [output.finish_reason == "length" for output in outputs],
        "texts": [output.text for output in outputs],
    }

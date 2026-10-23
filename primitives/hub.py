import json
from pathlib import Path
from tempfile import TemporaryDirectory

from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download, snapshot_download
from safetensors.torch import load_file, save, save_file
from torch import Tensor
from transformers import PreTrainedModel


def get_repo_name() -> str:
    return f"{HfApi().whoami()['name']}/where-quantization-hurts"


def make_numina_path(
    model_name: str,
    num_samples: int,
    quantized_path: str | None = None,
) -> str:
    if quantized_path is None:
        return f"numina/{model_name}/num_samples={num_samples}"
    return f"numina/{quantized_path},numina_num_samples={num_samples}"


def make_gptq_path(
    model_name: str,
    num_bits: int,
    group_size: int,
    num_samples: int,
) -> str:
    return f"gptq/{model_name}/num_bits={num_bits},group_size={group_size},num_samples={num_samples}"


def make_mxfp4_path(
    model_name: str,
    group_size: int,
) -> str:
    return f"mxfp4/{model_name}/group_size={group_size}"


def make_qat_path(
    model_name: str,
    num_bits: int,
    group_size: int,
    num_samples: int,
    qat_num_samples: int,
    high_entropy_ratio: float,
    high_entropy_weight: float,
    learning_rate: float,
    num_epochs: int,
    threshold_num_samples: int,
    on_policy: bool,
) -> str:
    return f"qat/{model_name}/num_bits={num_bits},group_size={group_size},num_samples={num_samples},qat_num_samples={qat_num_samples},high_entropy_ratio={high_entropy_ratio:g},high_entropy_weight={high_entropy_weight:g},learning_rate={learning_rate:g},num_epochs={num_epochs},threshold_num_samples={threshold_num_samples},on_policy={on_policy}"


def make_routing_path(
    quantized_path: str,
    route_ratio: float,
    route: str,
) -> str:
    return f"routing/{quantized_path},route_ratio={route_ratio:g},route={route}"


def save_generations(
    path: str,
    rows: list[dict],
    tensors: dict[str, Tensor],
) -> None:
    api = HfApi()
    repo_name = get_repo_name()
    api.create_repo(repo_name, exist_ok=True)
    operations = [
        CommitOperationAdd(path_in_repo=f"{path}/rows.json", path_or_fileobj=json.dumps(rows, indent=2).encode()),
        CommitOperationAdd(path_in_repo=f"{path}/outputs.safetensors", path_or_fileobj=save(tensors)),
    ]
    api.create_commit(repo_name, operations=operations, commit_message=f"Save {path}")
    print(f"saved {path}/rows.json, {path}/outputs.safetensors")


def save_quantized(
    path: str,
    model: PreTrainedModel,
    grid: dict[str, Tensor],
) -> None:
    api = HfApi()
    repo_name = get_repo_name()
    api.create_repo(repo_name, exist_ok=True)
    with TemporaryDirectory() as directory:
        model.save_pretrained(directory)
        save_file({name: tensor.cpu().contiguous() for name, tensor in grid.items()}, f"{directory}/grid.safetensors")
        api.upload_folder(folder_path=directory, path_in_repo=path, repo_id=repo_name, commit_message=f"Save {path}")
    print(f"saved {path}")


def save_merged(
    path: str,
    directory: str,
) -> None:
    api = HfApi()
    repo_name = get_repo_name()
    api.create_repo(repo_name, exist_ok=True)
    api.upload_folder(folder_path=directory, path_in_repo=path, repo_id=repo_name, commit_message=f"Save {path}")
    print(f"saved {path}")


def download_model(
    path: str,
) -> str:
    # vllm은 폴더 안 safetensors를 모두 weight로 읽음.
    # 따라서 grid는 받지 않음.
    return str(Path(snapshot_download(get_repo_name(), allow_patterns=[f"{path}/*"], ignore_patterns=[f"{path}/grid.safetensors"])) / path)


def load_grid(
    path: str,
) -> dict[str, Tensor]:
    return load_file(hf_hub_download(get_repo_name(), f"{path}/grid.safetensors"))


def get_pending_targets(
    entry_name: str,
    model_name: str,
    file_name: str,
    file_extension: str = "json",
) -> list[str]:
    files = set(HfApi().list_repo_files(get_repo_name()))
    targets = [f"bf16/{model_name}"]
    for file in sorted(files):
        if file.startswith((f"gptq/{model_name}/", f"mxfp4/{model_name}/", f"qat/{model_name}/")) and file.endswith("/grid.safetensors"):
            targets.append(str(Path(file).parent))
    targets = [target for target in targets if f"{entry_name}/{target}/{file_name}.{file_extension}" not in files]
    print(f"{entry_name} pending {len(targets)}", *targets, sep="\n  ")
    return targets


def save_result(
    path: str,
    file_name: str,
    result: dict,
) -> None:
    api = HfApi()
    repo_name = get_repo_name()
    api.create_repo(repo_name, exist_ok=True)
    api.upload_file(path_or_fileobj=json.dumps(result, indent=2).encode(), path_in_repo=f"{path}/{file_name}.json", repo_id=repo_name, commit_message=f"Save {path}/{file_name}.json")
    print(f"saved {path}/{file_name}.json")


def load_result(
    path: str,
    file_name: str,
) -> dict | None:
    repo_name = get_repo_name()
    if not HfApi().file_exists(repo_name, f"{path}/{file_name}.json"):
        return None
    return json.loads(Path(hf_hub_download(repo_name, f"{path}/{file_name}.json")).read_text())


def save_tensors(
    path: str,
    file_name: str,
    tensors: dict[str, Tensor],
) -> None:
    api = HfApi()
    repo_name = get_repo_name()
    api.create_repo(repo_name, exist_ok=True)
    api.upload_file(path_or_fileobj=save(tensors), path_in_repo=f"{path}/{file_name}.safetensors", repo_id=repo_name, commit_message=f"Save {path}/{file_name}.safetensors")
    print(f"saved {path}/{file_name}.safetensors")


def load_tensors(
    path: str,
    file_name: str,
) -> dict[str, Tensor]:
    return load_file(hf_hub_download(get_repo_name(), f"{path}/{file_name}.safetensors"))

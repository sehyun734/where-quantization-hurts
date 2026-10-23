import json
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download
from safetensors.torch import load_file, save
from torch import Tensor


def get_repo_name() -> str:
    return f"{HfApi().whoami()['name']}/where-quantization-hurts"


def upload_file(
    path: str,
    data: bytes,
) -> None:
    HfApi().upload_file(path_or_fileobj=data, path_in_repo=path, repo_id=get_repo_name(), commit_message=f"Save {path}")
    print(f"saved {path}")


def upload_folder(
    path: str,
    directory: str,
) -> None:
    HfApi().upload_folder(folder_path=directory, path_in_repo=path, repo_id=get_repo_name(), commit_message=f"Save {path}")
    print(f"saved {path}")


def download_file(
    path: str,
) -> str:
    return hf_hub_download(get_repo_name(), path)


def get_pending_targets(
    entry_name: str,
    model_name: str,
    file_name: str,
    file_extension: str = "json",
    include_bf16: bool = True,
) -> list[str]:
    files = set(HfApi().list_repo_files(get_repo_name()))
    targets = [f"bf16/{model_name}"] if include_bf16 else []
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
    upload_file(f"{path}/{file_name}.json", json.dumps(result, indent=2).encode())


def load_result(
    path: str,
    file_name: str,
) -> dict:
    return json.loads(Path(download_file(f"{path}/{file_name}.json")).read_text())


def save_tensors(
    path: str,
    file_name: str,
    tensors: dict[str, Tensor],
) -> None:
    upload_file(f"{path}/{file_name}.safetensors", save(tensors))


def load_tensors(
    path: str,
    file_name: str,
) -> dict[str, Tensor]:
    return load_file(download_file(f"{path}/{file_name}.safetensors"))

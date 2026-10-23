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
    scale_learning_rate: float,
) -> str:
    return f"qat/{model_name}/num_bits={num_bits},group_size={group_size},num_samples={num_samples},qat_num_samples={qat_num_samples},high_entropy_ratio={high_entropy_ratio:g},high_entropy_weight={high_entropy_weight:g},learning_rate={learning_rate:g},num_epochs={num_epochs},threshold_num_samples={threshold_num_samples},on_policy={on_policy},scale_learning_rate={scale_learning_rate:g}"


def make_routing_path(
    quantized_path: str,
    route_ratio: float,
    route: str,
) -> str:
    return f"routing/{quantized_path},route_ratio={route_ratio:g},route={route}"

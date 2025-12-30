# baselines
import importlib
from .model_utils import get_template, load_model_and_tokenizer, load_vllm_model, _init_ray

_method_mapping = {
    "JCB": "baselines.JCB",
}


def get_method_class(method):
    """
    Returns the method class given the method name. This is used to access static methods.
    """
    if method not in _method_mapping:
        raise ValueError(f"Can not find method {method}")
    module_path = _method_mapping[method]
    module = importlib.import_module(module_path)
    method_class = getattr(module, method)  # Class name assumed to be the same as method name
    return method_class


def init_method(method_class, method_config):
    if method_class.use_ray:
        _init_ray(num_cpus=8)
    output = method_class(**method_config)
    return output


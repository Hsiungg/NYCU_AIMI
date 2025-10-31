import random as _random
import numpy as _np
import torch as _torch
import os as _os
from itertools import product as _product
from typing import Callable as _Callable, Dict as _Dict, Any as _Any, List as _List, Tuple as _Tuple


def set_seed(seed: int) -> None:
    # Set CuBLAS workspace config for deterministic behavior
    _os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

    _random.seed(seed)
    _np.random.seed(seed)
    _torch.manual_seed(seed)
    if _torch.cuda.is_available():
        _torch.cuda.manual_seed_all(seed)
    _torch.backends.cudnn.deterministic = True
    _torch.backends.cudnn.benchmark = False
    try:
        _torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def param_grid_to_configs(param_grid: _Dict[str, _List[_Any]]) -> _List[_Dict[str, _Any]]:
    """Expand a dict of lists into a list of all param combinations.

    Example:
        {"lr": [1e-2, 1e-3], "dropout": [0.2, 0.5]}
        -> [{"lr":1e-2, "dropout":0.2}, {"lr":1e-2, "dropout":0.5}, ...]
    """
    if not param_grid:
        return [{}]
    keys = list(param_grid.keys())
    values_product = _product(*(param_grid[k] for k in keys))
    return [dict(zip(keys, values)) for values in values_product]


def grid_search(
    param_grid: _Dict[str, _List[_Any]],
    run_fn: _Callable[[_Dict[str, _Any]], _Dict[str, _Any] | float],
    *,
    metric_key: str = "metric",
    maximize: bool = True,
    verbose: bool = True,
) -> _Tuple[_Dict[str, _Any], _Dict[str, _Any] | float, _List[_Tuple[_Dict[str, _Any], _Dict[str, _Any] | float]]]:
    """Generic grid search utility.

    Args:
        param_grid: Mapping of hyperparameter name -> list of candidate values.
        run_fn: Callback that takes a config dict and returns either a float metric
                or a result dict containing metric under `metric_key`.
        metric_key: If run_fn returns a dict, which key to read as the metric.
        maximize: Whether to maximize (True) or minimize (False) the metric.
        verbose: Print progress and best-so-far.

    Returns:
        best_config, best_result, all_results
        - best_result is whatever run_fn returned for the best config
        - all_results is a list of (config, result) for all runs
    """
    configs = param_grid_to_configs(param_grid)
    if not configs:
        raise ValueError("param_grid resulted in no configurations")

    best_config: _Dict[str, _Any] | None = None
    best_result: _Dict[str, _Any] | float | None = None
    best_metric: float | None = None
    all_results: _List[_Tuple[_Dict[str, _Any], _Dict[str, _Any] | float]] = []

    for idx, config in enumerate(configs, start=1):
        result = run_fn(config)
        all_results.append((config, result))

        if isinstance(result, dict):
            if metric_key not in result:
                raise KeyError(
                    f"metric_key '{metric_key}' not found in run_fn result: {list(result.keys())}")
            metric = float(result[metric_key])
        else:
            metric = float(result)

        is_better = (best_metric is None) or (
            metric > best_metric if maximize else metric < best_metric)
        if is_better:
            best_metric = metric
            best_config = config
            best_result = result
        if verbose:
            prefix = f"[{idx}/{len(configs)}]"
            status = "best" if is_better else "done"
            print(f"{prefix} metric={metric:.6f} config={config} -> {status}")

    # mypy/typing guard
    assert best_config is not None and best_result is not None and best_metric is not None
    if verbose:
        direction = "max" if maximize else "min"
        print(
            f"Grid search complete. Best ({direction} {metric_key}): {best_metric:.6f} with config={best_config}")

    return best_config, best_result, all_results

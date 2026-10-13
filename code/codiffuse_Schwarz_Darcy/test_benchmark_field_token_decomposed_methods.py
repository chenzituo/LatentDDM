import inspect

from .benchmark_field_token_decomposed_methods import physical_zero, target_free_context


def test_target_free_context_function_is_explicitly_target_independent() -> None:
    source = inspect.getsource(target_free_context)
    assert 'batch["solution"]' not in source
    assert 'batch["mask"]' in source
    assert 'batch["solution"]' not in inspect.getsource(physical_zero)

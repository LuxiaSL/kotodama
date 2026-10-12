"""--data_seed decouples the training-data order from the init seed (init x data-order studies)."""
import argparse

from kotodama.training.train import training_data_seed


def test_data_seed_defaults_to_seed() -> None:
    assert training_data_seed(argparse.Namespace(seed=43, data_seed=None)) == 43
    assert training_data_seed(argparse.Namespace(seed=43)) == 43  # namespaces from before the flag existed


def test_data_seed_overrides_order_only() -> None:
    args = argparse.Namespace(seed=42, data_seed=43)
    assert training_data_seed(args) == 43
    assert args.seed == 42  # the init seed is untouched

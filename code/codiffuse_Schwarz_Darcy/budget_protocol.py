"""Explicit training subsets and shared field losses for the ICLR Darcy study."""
import hashlib
import json
from pathlib import Path

import torch


def apply_training_manifest(dataset, path, budget):
    if path is None:
        return None
    path = Path(path)
    raw = path.read_bytes()
    manifest = json.loads(raw)
    if dataset.split != 'train' or manifest['split'] != 'train':
        raise ValueError('label manifests are restricted to training data')
    if Path(manifest['dataset']).resolve() != dataset.path:
        raise ValueError('label manifest refers to another dataset')
    permutation = manifest['permutation']
    if sorted(permutation) != list(range(len(dataset))):
        raise ValueError('manifest must permute the entire training pool exactly once')
    if type(budget) is not int or not 1 <= budget <= len(dataset):
        raise ValueError('manifest needs an explicit positive budget within the pool')
    indices = permutation[:budget]
    dataset.case_indices = indices
    dataset.length = budget
    return {'manifest_sha256': hashlib.sha256(raw).hexdigest(),
            'subset_seed': manifest['subset_seed'], 'case_indices': indices,
            'dataset': str(dataset.path), 'split': 'train'}


def validate_resume_protocol(saved_args, args):
    for key in ('training_subset', 'selection_metric', 'field_loss', 'max_train_cases'):
        previous = saved_args.get(key)
        current = getattr(args, key, None)
        if previous != current:
            raise ValueError(f'resume changes scientific protocol: {key}')


def shared_field_loss(prediction, target, mask, sdf):
    """Masked field + .2 gradient + .25 near-boundary MSE in source units."""
    error = (prediction - target).square()
    field = (error * mask).sum() / mask.sum().clamp_min(1.)
    numerator = prediction.new_zeros(())
    denominator = prediction.new_zeros(())
    for dimension in (-2, -1):
        left = [slice(None)] * mask.ndim
        right = [slice(None)] * mask.ndim
        left[dimension] = slice(None, -1)
        right[dimension] = slice(1, None)
        pair = mask[tuple(left)] * mask[tuple(right)]
        diff = torch.diff(prediction, dim=dimension) - torch.diff(target, dim=dimension)
        numerator = numerator + (diff.square() * pair).sum()
        denominator = denominator + pair.sum()
    gradient = numerator / denominator.clamp_min(1.)
    boundary = torch.exp(-8. * sdf.abs()) * mask
    near = (error * boundary).sum() / boundary.sum().clamp_min(1.)
    return field + .2 * gradient + .25 * near, field

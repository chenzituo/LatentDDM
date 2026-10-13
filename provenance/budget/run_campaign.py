#!/usr/bin/env python3
"""Staged Darcy campaign; training phases never invoke protected evaluation."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
DARCY = PROJECT / 'codiffuse_Schwarz_Darcy'
OUTPUT = DARCY / 'outputs/iclr2027_boost_20260918'
LOCAL = DARCY / 'outputs/sni_face_local_seed20260806/checkpoint.pt'
COMPOSER = DARCY / 'outputs/input_correction_modes4_spatial4_trace0p5_depth4_warm_seed20260811/checkpoint_best.pt'
OLD_PATCH = DARCY / 'outputs/global_direct96_pilot_v1/selected/patch/checkpoint.pt'
POSITIONS = ['fixed_tile_fourier_v1', 'fixed_tile_fourier_locality_v1']
SEEDS = [20260918, 20260919, 20260920]
LRS = [3e-5, 1e-4, 3e-4]


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, default=str) + '\n')
    temporary.replace(path)


def dataset(scale):
    return DARCY / 'data' / f'darcy_domain_extension_fixed_dx_{scale}x{scale}_v1.h5'


def run(module, arguments, name):
    logs = OUTPUT / 'driver_logs'
    logs.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, '-m', module] + [str(v) for v in arguments]
    record = dict(command=command, started=time.time(), phase=name,
                  slurm_job_id=os.environ.get('SLURM_JOB_ID'))
    save(logs / f'{name}.command.json', record)
    print('RUN', name, flush=True)
    with (logs / f'{name}.log').open('a') as stream:
        completed = subprocess.run(command, cwd=PROJECT, stdout=stream,
                                   stderr=subprocess.STDOUT, check=False)
    record.update(ended=time.time(), exit_code=completed.returncode)
    save(logs / f'{name}.command.json', record)
    if completed.returncode:
        raise RuntimeError(f'{name} failed: see {logs / (name + ".log")}')


def source(position, smoke=False):
    target = OUTPUT / ('smoke/source' if smoke else f'sources/{position}')
    checkpoint = target / 'checkpoint.pt'
    if (target/'summary.json').exists():
        return checkpoint
    args = ['--data', DARCY/'data/darcy_full_domain_face_96_v2.h5',
            '--output-dir', target, '--statistics-checkpoint', LOCAL,
            '--architecture', 'global_vit', '--position-encoding', position,
            '--patch-size', 32, '--width', 256, '--depth', 8, '--heads', 8,
            '--mlp-ratio', 4, '--training-stage', 'full_domain',
            '--current-policy', 'zero', '--no-use-discrete-residual',
            '--selection-residual-weight', 0, '--validation-sweeps', 1,
            '--epochs', 1 if smoke else 80, '--samples-per-epoch', 8 if smoke else 4096,
            '--batch-size', 2 if smoke else 16, '--workers', 2 if smoke else 4,
            '--validation-batches', 1 if smoke else 16,
            '--field-loss', 'shared_field_v1', '--lr', 2e-4, '--min-lr', 2e-6,
            '--weight-decay', 1e-4, '--seed', SEEDS[0]]
    if smoke:
        args += ['--max-train-cases', 8, '--max-val-cases', 2]
    if (target/'last_checkpoint.pt').exists():
        args += ['--resume', target/'last_checkpoint.pt']
    run('codiffuse_Schwarz_Darcy.train_global', args, 'source_smoke' if smoke else 'source_'+position)
    return checkpoint


def adapt(method, checkpoint, position, scale, budget, seed, lr, folder, smoke=False):
    target = OUTPUT / folder
    if (target/'summary.json').exists():
        return target
    batch = 2 if scale == 8 else 1
    args = ['--data', dataset(scale), '--output-dir', target,
            '--train-manifest', HERE/'manifests'/f'darcy_domain_extension_fixed_dx_{scale}x{scale}_v1_seed{seed}.json',
            '--max-train-cases', budget, '--max-val-cases', 2 if smoke else 64,
            '--epochs', 1 if smoke else 24, '--samples-per-epoch', 4 if smoke else 256,
            '--batch-size', batch, '--workers', 2,
            '--validation-batches', 1 if smoke else 64//batch,
            '--lr', lr, '--weight-decay', 1e-4, '--seed', seed]
    if method == 'latent':
        module = 'codiffuse_Schwarz_Darcy.train_input_correction_latentddm'
        args += ['--local-checkpoint', LOCAL, '--core', 32, '--halo', 4,
                 '--token-width', 128, '--heads', 4, '--communication-depth', 4,
                 '--trace-modes', 4, '--spatial-token-bins', 4,
                 '--assembly', 'normalized_overlap', '--trace-weight', .5,
                 '--gradient-weight', .2, '--near-boundary-weight', .25,
                 '--selection-metric', 'relative_l2']
        last = target/'checkpoint_last.pt'
        args += ['--resume', last] if last.exists() else ['--initial-composer-checkpoint', COMPOSER]
    else:
        module = 'codiffuse_Schwarz_Darcy.train_global'
        args += ['--architecture', 'global_vit', '--position-encoding', position,
                 '--patch-size', 32, '--width', 256, '--depth', 8, '--heads', 8,
                 '--mlp-ratio', 4, '--training-stage', 'full_domain',
                 '--current-policy', 'zero', '--no-use-discrete-residual',
                 '--selection-residual-weight', 0, '--validation-sweeps', 1,
                 '--field-loss', 'shared_field_v1', '--min-lr', lr*.01,
                 '--adaptation-provenance', HERE/'PROTOCOL.md']
        if method == 'lora':
            args += ['--adaptation-mode', 'low_rank', '--low-rank', 24,
                     '--low-rank-placement', 'attention_ffn']
        last = target/'last_checkpoint.pt'
        args += ['--resume', last] if last.exists() else ['--initial-checkpoint', checkpoint]
    run(module, args, folder.replace('/', '_'))
    return target


def score(path):
    q = json.loads((path/'summary.json').read_text())
    value = q['best_selection_score']
    if not math.isfinite(value):
        raise ValueError(f'non-finite validation score: {path}')
    return value


def screen():
    frozen = OUTPUT/'selection.json'
    if frozen.exists():
        return
    candidates = [('normalized_linear_fourier_v1', OLD_PATCH)] + [
        (position, OUTPUT/'sources'/position/'checkpoint.pt') for position in POSITIONS]
    trials = []
    for position, checkpoint in candidates:
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
        for lr in LRS:
            path = adapt('full', checkpoint, position, 8, 16, SEEDS[0], lr,
                         f'screen/full_{position}_lr{lr:g}')
            trials.append(dict(method='full', position=position, checkpoint=str(checkpoint.resolve()),
                               lr=lr, score=score(path), run=str(path.resolve())))
    selected = min(trials, key=lambda row: row['score'])
    choices = {'full': selected}
    for method in ('latent', 'lora'):
        rows = []
        for lr in LRS:
            path = adapt(method, Path(selected['checkpoint']), selected['position'],
                         8, 16, SEEDS[0], lr, f'screen/{method}_lr{lr:g}')
            rows.append(dict(method=method, lr=lr, score=score(path), run=str(path.resolve())))
        choices[method] = min(rows, key=lambda row: row['score'])
        trials += rows
    save(frozen, dict(choices=choices, trials=trials, criterion='8x8 m16 seed20260918 validation relative L2',
                     protocol_sha256=hashlib.sha256((HERE/'PROTOCOL.md').read_bytes()).hexdigest(),
                     protected_test_opened=False, frozen_at=time.time()))


def sweep(scale):
    selected = json.loads((OUTPUT/'selection.json').read_text())['choices']
    for budget in (4, 16, 64):
        for seed in SEEDS:
            for method in ('latent', 'full', 'lora'):
                adapt(method, Path(selected['full']['checkpoint']), selected['full']['position'],
                      scale, budget, seed, selected[method]['lr'],
                      f'sweep/{scale}x{scale}/m{budget}/{method}_seed{seed}')


def smoke():
    checkpoint = source(POSITIONS[1], smoke=True)
    for method in ('full', 'lora', 'latent'):
        adapt(method, checkpoint, POSITIONS[1], 8, 4, SEEDS[0], 1e-4,
              f'smoke/{method}', smoke=True)
    # Exercise inference deserialization and merged LoRA on validation only.
    for method in ('full', 'lora'):
        target = OUTPUT/'smoke'/method
        if not (target/'validation/summary.json').exists():
            run('codiffuse_Schwarz_Darcy.evaluate_global', [
                '--data', dataset(8), '--checkpoint', target/'checkpoint.pt',
                '--output-dir', target/'validation', '--groups', 'validation',
                '--sweeps', 1, '--batch-size', 2, '--workers', 0,
                '--initialization', 'zero', '--max-cases', 2], f'smoke_evaluate_{method}')
    target = OUTPUT/'smoke/latent'
    if not (target/'validation.json').exists():
        run('codiffuse_Schwarz_Darcy.evaluate_input_correction_latentddm', [
            '--data', dataset(8), '--local-checkpoint', LOCAL,
            '--composer-checkpoint', target/'checkpoint_best.pt',
            '--output', target/'validation.json', '--split', 'validation',
            '--max-cases', 2, '--batch-size', 2, '--workers', 0, '--sni-sweeps', 0],
            'smoke_evaluate_latent')
    save(OUTPUT/'smoke/PASSED.json', {'time':time.time(), 'protected_test_opened':False})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['smoke', 'source', 'screen', 'sweep'])
    parser.add_argument('--index', type=int, default=0)
    parser.add_argument('--scale', type=int, choices=[8, 16], default=8)
    args = parser.parse_args()
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('GPU campaign requires an allocated CUDA device')
    frozen_code = HERE/'code_sha256.json'
    if frozen_code.exists():
        for name, expected in json.loads(frozen_code.read_text()).items():
            if hashlib.sha256((PROJECT/name).read_bytes()).hexdigest() != expected:
                raise RuntimeError(f'campaign source changed after freezing: {name}')
    hardware = dict(torch_version=torch.__version__, cuda_version=torch.version.cuda,
                    gpu=torch.cuda.get_device_name(), phase=args.phase,
                    index=args.index, scale=args.scale, started=time.time(),
                    slurm_job_id=os.environ.get('SLURM_JOB_ID'),
                    hostname=os.uname().nodename)
    save(OUTPUT/'hardware'/f'{args.phase}_{args.index}_{args.scale}_{os.environ.get("SLURM_JOB_ID", "local")}.json', hardware)
    if args.phase != 'smoke' and not (OUTPUT/'smoke/PASSED.json').exists():
        raise RuntimeError('Complete GPU smoke test before production')
    if args.phase == 'source':
        source(POSITIONS[args.index])
    elif args.phase == 'screen':
        screen()
    elif args.phase == 'sweep':
        sweep(args.scale)
    else:
        smoke()


if __name__ == '__main__':
    main()

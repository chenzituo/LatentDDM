#!/usr/bin/env python3
"""Audit completed training and score frozen checkpoints; never train or select."""
import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from run_campaign import HERE, PROJECT, OUTPUT, LOCAL, COMPOSER, SEEDS, dataset, save

FINAL = OUTPUT / 'final_test'
METHODS = ('latent', 'full', 'lora')
COUNTS = dict(latent=796400, full=9729024, lora=786432)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def frozen_check():
    records = read(HERE / 'code_sha256.json')
    for name, expected in records.items():
        require(sha(PROJECT / name) == expected, f'Frozen code changed: {name}')
    return len(records)


def run_specs(scale, selection):
    for method, checkpoint in [('latent', COMPOSER), ('full', Path(selection['full']['checkpoint']))]:
        yield dict(scale=scale, budget=0, seed=None, method=method,
                   checkpoint=str(checkpoint.resolve()), key=f'{scale}x{scale}/m0/{method}')
    for budget in (4, 16, 64):
        for seed in SEEDS:
            for method in METHODS:
                relative = f'{scale}x{scale}/m{budget}/{method}_seed{seed}'
                folder = OUTPUT / 'sweep' / relative
                yield dict(scale=scale, budget=budget, seed=seed, method=method,
                           checkpoint=str((folder / ('checkpoint_best.pt' if method == 'latent' else 'checkpoint.pt')).resolve()),
                           key=relative, training_dir=str(folder.resolve()))


def audit(allow_partial=False):
    import torch
    checked = frozen_check()
    selection = read(OUTPUT / 'selection.json')
    require(selection['protocol_sha256'] == sha(HERE / 'PROTOCOL.md'), 'Protocol changed')
    choices = selection['choices']
    provenance = read(HERE / 'input_provenance.json')
    local_hash = sha(LOCAL)
    composer_hash = sha(COMPOSER)
    for path, digest in [(LOCAL, local_hash), (COMPOSER, composer_hash)]:
        expected = next(v['sha256'] for v in provenance.values() if Path(v['resolved_path']).resolve() == path.resolve())
        require(digest == expected, f'Source changed: {path}')
    local_stats = torch.load(LOCAL, map_location='cpu', weights_only=False)['statistics']
    source = Path(choices['full']['checkpoint'])
    source_hash = sha(source)
    source_saved = torch.load(source, map_location='cpu', weights_only=False)
    require(source_saved['statistics'] == local_stats, 'Selected Patch source normalization mismatch')
    data_hashes = {}
    # Hash whole containers; this does not score or inspect test fields.
    for scale in (8, 16):
        path = dataset(scale)
        data_hashes[str(scale)] = sha(path)
        require(data_hashes[str(scale)] == provenance[str(path.relative_to(PROJECT))]['sha256'], 'Data changed')
    rows, missing = [], []
    for scale in (8, 16):
        for spec in run_specs(scale, choices):
            if not spec['budget']:
                continue
            folder = Path(spec['training_dir'])
            if not (folder / 'summary.json').exists():
                missing.append(spec['key'])
                continue
            q = read(folder / 'summary.json')
            checkpoint = Path(spec['checkpoint'])
            c = torch.load(checkpoint, map_location='cpu', weights_only=False)
            a = c['args']
            method, budget, seed = spec['method'], spec['budget'], spec['seed']
            batch = 2 if scale == 8 else 1
            for key, expected in dict(train_cases=budget, validation_cases=64, epochs_completed=24,
                                      samples_per_epoch=256, sample_exposures=6144,
                                      trainable_parameter_count=COUNTS[method]).items():
                require(q[key] == expected, f'{spec["key"]}: summary {key}')
            for key, expected in dict(seed=seed, max_train_cases=budget, max_val_cases=64,
                                      epochs=24, samples_per_epoch=256, batch_size=batch,
                                      validation_batches=64//batch, lr=choices[method]['lr'],
                                      weight_decay=1e-4).items():
                require(a[key] == expected, f'{spec["key"]}: args {key}')
            require(Path(a['data']).resolve() == dataset(scale).resolve(), 'Wrong dataset')
            require(a['resume'] is None, 'Unexpected resumed RNG state')
            manifest = HERE / 'manifests' / f'darcy_domain_extension_fixed_dx_{scale}x{scale}_v1_seed{seed}.json'
            subset = a['training_subset']
            require(subset['case_indices'] == read(manifest)['permutation'][:budget], 'Wrong subset')
            require(subset['manifest_sha256'] == sha(manifest), 'Manifest changed')
            require(subset['split'] == 'train', 'Wrong training split')
            history = q['history']
            require([int(r['epoch']) for r in history] == list(range(1, 25)), 'Incomplete epochs')
            scores = [r['selection_score'] for r in history]
            require(all(math.isfinite(x) for x in scores), 'Nonfinite selection')
            require(math.isclose(min(scores), q['best_selection_score'], abs_tol=1e-10), 'Best score mismatch')
            require(int(c['epoch']) == scores.index(min(scores)) + 1, 'Best checkpoint mismatch')
            require(c['statistics'] == local_stats, 'Normalizer mismatch')
            if method == 'latent':
                require(c['local_checkpoint_sha256'] == local_hash, 'Local changed')
                require(c['initial_composer_checkpoint_sha256'] == composer_hash, 'Composer changed')
                require(a['selection_metric'] == 'relative_l2', 'Wrong selection metric')
                require(all(r['validation_examples'] == 64 for r in history), 'Incomplete validation')
                require(sum(r['cases'] for r in c['validation_rows']) == 64, 'Checkpoint validation incomplete')
                score = sum(r['relative_l2'] * r['cases'] for r in c['validation_rows']) / 64
            else:
                require(Path(c['initial_checkpoint']).resolve() == source.resolve(), 'Wrong source')
                require(c['model_spec']['position_encoding'] == choices['full']['position'], 'Wrong positions')
                require(a['field_loss'] == 'shared_field_v1' and a['selection_residual_weight'] == 0, 'Wrong loss/selection')
                require(c['adaptation_mode'] == ('low_rank' if method == 'lora' else 'full'), 'Wrong adaptation')
                if method == 'lora':
                    require(c['low_rank']['rank'] == 24 and c['low_rank']['placement'] == 'attention_ffn', 'Wrong LoRA')
                    require(not any('lora_' in k for k in c['model']), 'Unmerged model')
                score = c['validation']['final_relative_l2']
            require(math.isclose(score, q['best_selection_score'], rel_tol=1e-6, abs_tol=1e-8), 'Checkpoint score mismatch')
            command = read(OUTPUT / 'driver_logs' / ('sweep_' + spec['key'].replace('/', '_') + '.command.json'))
            require(command.get('exit_code') == 0, 'Training command incomplete')
            rows.append(dict(**spec, checkpoint_sha256=sha(checkpoint), best_epoch=int(c['epoch']),
                             validation_relative_l2=score, process_seconds=command['ended']-command['started'],
                             trainer_seconds=q['wall_seconds'], slurm_job_id=command['slurm_job_id']))
    result = dict(checked_at=time.time(), frozen_files_checked=checked, selection_sha256=sha(OUTPUT/'selection.json'),
                  local_sha256=local_hash, composer_sha256=composer_hash, patch_source_sha256=source_hash,
                  data_sha256=data_hashes, complete=not missing, missing=missing, runs=rows)
    save(HERE / ('partial_audit.json' if allow_partial else 'final_audit.json'), result)
    require(allow_partial or not missing, f'Incomplete training: {missing}')
    return result


def metrics(spec, directory):
    if spec['method'] == 'latent':
        q = read(directory / 'metrics.json')
        require(q['split'] == 'test' and q['protected_test_cases_opened'] == 64, 'Wrong composer test split')
        rows = [r for r in q['rows'] if r['method'] == 'latentddm']
    else:
        q = read(directory / 'summary.json')
        require(q['groups'] == ['test'] and q['protected_test_cases_opened'] == 64, 'Wrong global test split')
        with (directory / 'iteration_metrics.csv').open() as handle:
            rows = list(csv.DictReader(handle))
        require(all(r['group'] == 'test' and int(r['sweep']) == 1 for r in rows), 'Wrong test rows')
    require(len(rows) == 64 and sorted(int(r['case_index']) for r in rows) == list(range(64)), 'Test case mismatch')
    require(all(math.isfinite(float(r['relative_l2'])) for r in rows), 'Nonfinite test metric')
    return sorted(rows, key=lambda r: int(r['case_index']))


def evaluate(scale):
    import torch
    require(torch.cuda.is_available(), 'Allocated CUDA GPU required')
    audit_result = audit()
    choices = read(OUTPUT / 'selection.json')['choices']
    code_hashes = {p.name: sha(p) for p in (HERE/'final_evaluation.py', HERE/'analyze_final.py', HERE/'ANALYSIS_PLAN.md')}
    freeze = read(HERE / 'analysis_sha256.json')
    require(code_hashes == freeze, 'Analysis changed after freezing')
    for spec in run_specs(scale, choices):
        target = FINAL / spec['key']
        checkpoint_hash = sha(spec['checkpoint'])
        identity = dict(spec=spec, checkpoint_sha256=checkpoint_hash,
                        local_checkpoint_sha256=audit_result['local_sha256'],
                        data_sha256=audit_result['data_sha256'][str(scale)],
                        selection_sha256=audit_result['selection_sha256'], analysis_sha256=code_hashes)
        if (target / 'COMPLETE.json').exists():
            old = read(target / 'COMPLETE.json')
            require(old['identity'] == identity, 'Existing evaluation provenance differs')
            metrics(spec, target)
            continue
        require(not target.exists(), f'Partial evaluation requires inspection: {target}')
        target.mkdir(parents=True)
        batch = 2 if scale == 8 else 1
        common = ['--data', str(dataset(scale)), '--batch-size', str(batch), '--workers', '2', '--max-cases', '64', '--allow-protected-test']
        if spec['method'] == 'latent':
            module = 'codiffuse_Schwarz_Darcy.evaluate_input_correction_latentddm'
            args = common + ['--local-checkpoint', str(LOCAL), '--composer-checkpoint', spec['checkpoint'],
                             '--output', str(target/'metrics.json'), '--split', 'test', '--sni-sweeps', '0']
        else:
            module = 'codiffuse_Schwarz_Darcy.evaluate_global'
            args = common + ['--checkpoint', spec['checkpoint'], '--output-dir', str(target), '--groups', 'test', '--sweeps', '1', '--initialization', 'zero']
        command = [sys.executable, '-m', module] + args
        record = dict(identity=identity, command=command, started=time.time(),
                      slurm_job_id=os.environ.get('SLURM_JOB_ID'), gpu=torch.cuda.get_device_name(),
                      hostname=os.uname().nodename, torch_version=torch.__version__)
        save(target/'command.json', record)
        print('EVALUATE', spec['key'], flush=True)
        with (target/'process.log').open('w') as handle:
            result = subprocess.run(command, cwd=PROJECT, stdout=handle, stderr=subprocess.STDOUT)
        record.update(ended=time.time(), exit_code=result.returncode)
        save(target/'command.json', record)
        require(result.returncode == 0, f'Evaluation failed: {target}')
        metrics(spec, target)
        require(sha(spec['checkpoint']) == checkpoint_hash, 'Checkpoint changed during evaluation')
        record['result_sha256'] = {p.name: sha(p) for p in target.iterdir() if p.name in ('metrics.json', 'summary.json', 'iteration_metrics.csv')}
        save(target/'COMPLETE.json', record)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['audit', 'evaluate'])
    parser.add_argument('--allow-partial', action='store_true')
    parser.add_argument('--scale', type=int, choices=[8, 16], default=8)
    args = parser.parse_args()
    if args.phase == 'audit':
        result = audit(args.allow_partial)
        print(json.dumps(dict(complete=result['complete'], completed=len(result['runs']), missing=result['missing'])))
    else:
        FINAL.mkdir(parents=True, exist_ok=True)
        with (FINAL/f'.scale{args.scale}.lock').open('w') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            evaluate(args.scale)


if __name__ == '__main__':
    main()

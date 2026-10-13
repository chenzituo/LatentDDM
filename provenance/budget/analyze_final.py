#!/usr/bin/env python3
"""Paired analysis of the complete frozen Darcy boost test matrix."""
import csv
import json
from pathlib import Path
import subprocess
import time

import numpy as np

from final_evaluation import (HERE, OUTPUT, FINAL, SEEDS, METHODS, read, sha,
                              require, run_specs, metrics, save)

N_BOOT = 20000
BOOT_SEED = 20260919
LABELS = dict(latent='LatentDDM', full='Spatial Patch FT', lora='Spatial Patch LoRA')
COLORS = dict(latent='#146b83', full='#bd4a2e', lora='#8064a2')


def bootstrap_draws(n_seeds=3, n_cases=64, draws=N_BOOT, seed=BOOT_SEED):
    rng = np.random.default_rng(seed)
    return rng.integers(n_seeds, size=(draws, n_seeds)), rng.integers(n_cases, size=(draws, n_cases))


def summarize(values, seeds, cases):
    values = np.asarray(values, dtype=float)
    require(values.ndim == 2 and np.isfinite(values).all(), 'Bad bootstrap input')
    if values.shape[0] == 1:
        means = values[0, cases].mean(axis=1)
    else:
        means = values[seeds[:, :, None], cases[:, None, :]].mean(axis=(1, 2))
    return dict(mean=float(values.mean()), ci95=np.quantile(means, [.025, .975]).tolist(),
                seed_means=values.mean(axis=1).tolist())


def write_csv(path, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def compute_ledger():
    jobs = read(HERE/'jobs.json')
    if (HERE/'final_jobs.json').exists():
        jobs.update(read(HERE/'final_jobs.json'))
    ids = ','.join(str(v['job_id']) for v in jobs.values())
    result = subprocess.run(['sacct', '-j', ids, '--parsable2', '--noheader',
                             '--format=JobIDRaw,JobName,State,ElapsedRaw,AllocTRES,NodeList,ExitCode'],
                            capture_output=True, text=True)
    commands = []
    for path in sorted((OUTPUT/'driver_logs').glob('*.command.json')):
        q = read(path)
        commands.append(dict(path=str(path), job=q.get('slurm_job_id'), phase=q.get('phase'),
                             exit_code=q.get('exit_code'),
                             process_seconds=(q['ended']-q['started']) if 'ended' in q else None))
    interrupted = []
    for path in sorted((OUTPUT/'interrupted').rglob('*.command.json')):
        interrupted.append(dict(path=str(path), record=read(path)))
    hardware = {p.name: read(p) for p in sorted((OUTPUT/'hardware').glob('*.json'))}
    return dict(scheduler_command=result.args, scheduler_returncode=result.returncode,
                scheduler_output=result.stdout, scheduler_stderr=result.stderr,
                commands=commands, interrupted_commands=interrupted, hardware=hardware,
                timeout_recovery=read(HERE/'timeout_recovery.json'),
                note='Allocation includes idle/startup and original timeout. Do not add allocation to process time. Final job may still be running at this snapshot.')


def main():
    freeze = read(HERE/'analysis_sha256.json')
    require(all(sha(HERE/name) == digest for name, digest in freeze.items()), 'Analysis code changed')
    choices = read(OUTPUT/'selection.json')['choices']
    audit = read(HERE/'final_audit.json')
    require(audit['complete'] and len(audit['runs']) == 54, 'Final audit incomplete')
    arrays, per_case, completions = {}, [], {}
    for scale in (8, 16):
        for spec in run_specs(scale, choices):
            folder = FINAL/spec['key']
            q = read(folder/'COMPLETE.json')
            require(q['exit_code'] == 0, 'Failed evaluation')
            identity = q['identity']
            require(identity['spec'] == spec and identity['analysis_sha256'] == freeze, 'Evaluation identity differs')
            require(identity['selection_sha256'] == sha(OUTPUT/'selection.json'), 'Selection changed')
            require(identity['checkpoint_sha256'] == sha(spec['checkpoint']), 'Checkpoint changed')
            for name, digest in q['result_sha256'].items():
                require(sha(folder/name) == digest, 'Result changed')
            rows = metrics(spec, folder)
            key = (scale, spec['budget'], spec['method'])
            arrays.setdefault(key, []).append([float(r['relative_l2']) for r in rows])
            completions[spec['key']] = sha(folder/'COMPLETE.json')
            per_case.extend(dict(scale=scale, budget=spec['budget'], method=spec['method'],
                                 seed=spec['seed'], case_index=int(r['case_index']),
                                 relative_l2=float(r['relative_l2'])) for r in rows)
    arrays = {k: np.array(v) for k, v in arrays.items()}
    seed_draws, case_draws = bootstrap_draws()
    estimates, comparisons = [], []
    for (scale, budget, method), values in arrays.items():
        require(values.shape == (1 if budget == 0 else 3, 64), 'Incomplete metric matrix')
        estimates.append(dict(scale=scale, budget=budget, method=method,
                              **summarize(values, seed_draws, case_draws)))
    for scale in (8, 16):
        for budget in (4, 16, 64):
            for method in ('full', 'lora'):
                delta = arrays[scale, budget, method] - arrays[scale, budget, 'latent']
                comparisons.append(dict(scale=scale, budget=budget, comparator=method,
                                        **summarize(delta, seed_draws, case_draws)))
    out = FINAL/'analysis'
    out.mkdir(exist_ok=True)
    save(out/'results.json', dict(created=time.time(), bootstrap_draws=N_BOOT, bootstrap_seed=BOOT_SEED,
                                plan_sha256=sha(HERE/'ANALYSIS_PLAN.md'), completion_sha256=completions,
                                estimates=estimates, paired_global_minus_latent=comparisons))
    write_csv(out/'per_case.csv', per_case)
    timing = [dict(scale=r['scale'], budget=r['budget'], method=r['method'], seed=r['seed'],
                   process_seconds=r['process_seconds'], trainer_seconds=r['trainer_seconds'],
                   slurm_job_id=r['slurm_job_id']) for r in audit['runs']]
    write_csv(out/'adaptation_times.csv', timing)
    save(out/'compute_ledger.json', compute_ledger())
    plot(out, estimates, timing)
    lines = ['# Darcy final-test analysis', '',
             '64 reused test domains per scale; three paired adaptation seeds. Fixed source checkpoints.',
             'Intervals are pointwise paired seed/domain bootstrap 95% intervals (20,000 draws).',
             '64 target-validation simulations per scale are additional overhead.', '',
             '| Scale | Training cases | Method | Mean relative L2 | 95% interval |',
             '|---|---:|---|---:|---|']
    for r in estimates:
        lines.append(f'| {r["scale"]}×{r["scale"]} | {r["budget"]} | {LABELS[r["method"]]} | {r["mean"]:.6f} | [{r["ci95"][0]:.6f}, {r["ci95"][1]:.6f}] |')
    lines += ['', 'm0 Patch is shared by FT and LoRA and evaluated once per scale. Its interval resamples domains only.', '',
              '| Scale | Training cases | Comparator minus LatentDDM | Mean difference | 95% interval |',
              '|---|---:|---|---:|---|']
    for r in comparisons:
        lines.append(f'| {r["scale"]}×{r["scale"]} | {r["budget"]} | {LABELS[r["comparator"]]} | {r["mean"]:.6f} | [{r["ci95"][0]:.6f}, {r["ci95"][1]:.6f}] |')
    positive = all(r['ci95'][0] > 0 for r in comparisons if r['budget'] in (4, 16))
    lines += ['', 'The predeclared low-training-budget criterion is ' + ('met.' if positive else 'not met across all settings.'), '',
              'Three seeds provide limited uncertainty resolution; source-pretraining variability is unmeasured.',
              'The comparison is between complete source recipes. The test sets were historically inspected.',
              'Process timing includes startup, training, validation, and checkpoint I/O; see the hardware ledger.',
              'No source-cost matching, total-label-budget, controlled hardware speedup, or interpolated label-savings claim is made.',
              '', 'Manuscript review proposals and visual inspection of generated figures remain separate tasks.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print('Analysis complete:', out, flush=True)


def plot(out, estimates, timing):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42})
    lookup = {(r['scale'], r['budget'], r['method']): r for r in estimates}
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.4), sharey=True)
    for ax, scale in zip(axes, (8, 16)):
        for method in METHODS:
            rows = [lookup[scale, b, 'full' if b == 0 and method == 'lora' else method] for b in (0, 4, 16, 64)]
            means = np.array([r['mean'] for r in rows])
            bounds = np.array([r['ci95'] for r in rows]).T
            ax.plot(range(4), means, marker='o', color=COLORS[method], label=LABELS[method],
                    linestyle='--' if method == 'lora' else '-')
            ax.vlines(range(4), bounds[0], bounds[1], color=COLORS[method], alpha=.8)
            for x, r in enumerate(rows):
                if x:
                    ax.scatter([x]*3, r['seed_means'], color=COLORS[method], s=10, alpha=.4)
        ax.set(title=f'{scale}×{scale} tiles', xlabel='Target training simulations',
               xticks=range(4), xticklabels=['0', '4', '16', '64'], yscale='log')
        ax.grid(axis='y', alpha=.15)
    axes[0].set_ylabel('Mean relative L2 error')
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(out/f'darcy_budget.{ext}', dpi=220)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.4), sharey=True)
    for ax, scale in zip(axes, (8, 16)):
        for method in METHODS:
            groups = [[r['process_seconds']/60 for r in timing if r['scale'] == scale and r['budget'] == b and r['method'] == method] for b in (4, 16, 64)]
            ax.plot(range(3), [np.mean(g) for g in groups], 'o-', color=COLORS[method], label=LABELS[method])
            for x, g in enumerate(groups):
                ax.scatter([x]*3, g, color=COLORS[method], s=12, alpha=.4)
        ax.set(title=f'{scale}×{scale} tiles', xlabel='Target training simulations',
               xticks=range(3), xticklabels=['4', '16', '64'], yscale='log')
        ax.grid(axis='y', alpha=.15)
    axes[0].set_ylabel('Adaptation process time (minutes)')
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(out/f'darcy_adaptation_time.{ext}', dpi=220)
    plt.close(fig)


if __name__ == '__main__':
    main()

"""Controlled Darcy adaptation-location experiment; isolated from published trainers."""
from __future__ import annotations
import argparse
import csv
import fcntl
import hashlib
import json
import os
import time
import traceback
from types import MethodType
from pathlib import Path

import h5py
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader

from codiffuse_Schwarz_Airfoil.code_transfer.cover import CartesianCover
from codiffuse_Schwarz_Darcy.adapter import H5DarcyDataset, tiled_sni_face_conditions
from codiffuse_Schwarz_Darcy.budget_protocol import shared_field_loss
from codiffuse_Schwarz_Darcy.evaluate import move_batch, relative_l2
from codiffuse_Schwarz_Darcy.input_correction_latentddm import (
    DarcyInputCorrectionLatentDDM, boundary_input_context, corrected_local_conditions,
)
from codiffuse_Schwarz_Darcy.train import seed_everything
from codiffuse_Schwarz_Darcy.train_residual_port_chart import load_frozen_local

torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DARCY = ROOT / 'codiffuse_Schwarz_Darcy'
OUT = DARCY / 'outputs/darcy_adaptation_location_20260921'
PACKAGE = HERE / 'ablation'
BUDGET = ROOT / 'iclr2027/experiments/darcy_boost_20260918'
LOCAL = DARCY / 'outputs/sni_face_local_seed20260806/checkpoint.pt'
SOURCE = DARCY / 'outputs/input_correction_modes4_spatial4_trace0p5_depth4_warm_seed20260811/checkpoint_best.pt'
SEEDS = [20260918, 20260919, 20260920]
ARMS = ['composition_only', 'local_adapter_only']
LRS = [3e-5, 1e-4, 3e-4]
RUN_COLUMNS = ['run_id','arm','scale','seed','status','learning_rate','trainable_parameters','selected_epoch','checkpoint_sha256','adaptation_seconds','peak_memory_mb']
TEST_COLUMNS = ['run_id','domain_id','relative_l2','error_l2','reference_l2']
HISTORY_COLUMNS = ['run_id','epoch','mean_relative_l2']


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def save(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,sort_keys=True,default=str)+'\n');temp.replace(path)


def read(path):return json.loads(Path(path).read_text())


def write_csv(path, rows, columns):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    with temp.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=columns,extrasaction='ignore');w.writeheader();w.writerows(rows)
    temp.replace(path)


def data(scale):return DARCY/f'data/darcy_domain_extension_fixed_dx_{scale}x{scale}_v1.h5'
def manifest(scale,seed):return BUDGET/f'manifests/darcy_domain_extension_fixed_dx_{scale}x{scale}_v1_seed{seed}.json'
def domain_id(scale,seed):return f'{scale}x{scale}:geometry_seed_{int(seed)}'


class ConvLoRA(nn.Module):
    """Delta W = B A on the flattened convolution kernel; alpha/rank = 1."""
    def __init__(self, base, rank):
        super().__init__()
        if base.groups != 1 or base.padding_mode != 'zeros':
            raise ValueError('Only ordinary ungrouped zero-padded convolutions supported')
        self.base=base.requires_grad_(False)
        self.rank=min(rank,base.out_channels,base.weight[0].numel())
        self.lora_a=nn.Parameter(base.weight.new_empty(self.rank,base.weight[0].numel()))
        self.lora_b=nn.Parameter(base.weight.new_zeros(base.out_channels,self.rank))
        nn.init.kaiming_uniform_(self.lora_a,a=5**.5)

    def forward(self,x):
        # Form delta in fp32 even under autocast, then convolve with the same kernel layout.
        with torch.autocast(device_type=x.device.type,enabled=False):
            weight=self.base.weight+(self.lora_b @ self.lora_a).reshape_as(self.base.weight)
        return F.conv2d(x,weight,self.base.bias,self.base.stride,self.base.padding,
                        self.base.dilation,self.base.groups)


def install(local,rank=24):
    paths={}
    for name,module in list(local.named_modules()):
        if isinstance(module,nn.Conv2d):
            parent,_,key=name.rpartition('.')
            adapted=ConvLoRA(module,rank)
            setattr(local.get_submodule(parent),key,adapted)
            paths[name]=adapted.rank
    return paths


class FiniteSqrt(torch.autograd.Function):
    """Exact sqrt forward; zero subgradient for a zero RMS feature."""
    @staticmethod
    def forward(ctx, x):
        y=x.sqrt();ctx.save_for_backward(y);return y

    @staticmethod
    def backward(ctx, grad):
        (y,)=ctx.saved_tensors
        positive=y>0
        return grad * positive / (2*y.masked_fill(~positive,1))


def differentiable_encoder_forward(self,local,physical,scalar,valid,grid_shape):
    # Same source features and order. RMS of an empty/all-zero face uses the
    # zero subgradient, avoiding sqrt(0)'s infinity times a zero field derivative.
    field=torch.cat((physical,local),dim=1)
    denominator=valid.sum((2,3)).clamp_min(1.)
    mean=(field*valid).sum((2,3))/denominator
    rms=FiniteSqrt.apply((field.square()*valid).sum((2,3))/denominator)
    faces=[]
    for axis in (2,3):
        for index in (0,-1):
            face=field.select(axis,index);weight=valid.select(axis,index)
            den=weight.sum(2).clamp_min(1.)
            faces.extend([(face*weight).sum(2)/den,
                          FiniteSqrt.apply((face.square()*weight).sum(2)/den)])
    features=torch.cat((mean,rms,torch.cat(faces,dim=1),scalar),dim=1)
    return self.encoder(features).reshape(len(local)//(grid_shape[0]*grid_shape[1]),-1,self.width)


def build(arm,device):
    local,normalizer,base=load_frozen_local(LOCAL,device)
    source=torch.load(SOURCE,map_location='cpu',weights_only=False)
    assert source['local_checkpoint_sha256']==sha(LOCAL)
    assert source['statistics']==base['statistics']
    a=source['args']
    composer=DarcyInputCorrectionLatentDDM(29,4,token_width=a['token_width'],heads=a['heads'],
        communication_depth=a['communication_depth'],trace_modes=a['trace_modes'],
        spatial_token_bins=a['spatial_token_bins']).to(device)
    composer.load_state_dict(source['composer'],strict=True)
    composer.encoder.forward=MethodType(differentiable_encoder_forward,composer.encoder)
    composer.requires_grad_(arm=='composition_only')
    paths=install(local) if arm=='local_adapter_only' else {}
    # Both arms keep identical deterministic module modes. No dropout/batchnorm in source.
    local.eval();composer.eval()
    return local,composer,normalizer,paths


def parameters(local,composer):
    return [p for m in [local,composer] for p in m.parameters() if p.requires_grad]


def local_call(local,physical,scalar,cover,chunk=64):
    # Source local family has neither cross-tile attention nor batch statistics.
    # Identical tile microbatching in both arms; checkpoint recomputation only in backward.
    output=[]
    for begin in range(0,len(physical),chunk):
        p=physical[begin:begin+chunk];s=scalar[begin:begin+chunk]
        def call(p,s):return local(p,s,cover.tile_count,tile_grid_shape=cover.grid_shape)
        needs_grad=torch.is_grad_enabled() and (p.requires_grad or any(q.requires_grad for q in local.parameters()))
        output.append(checkpoint(call,p,s,use_reentrant=False) if needs_grad else call(p,s))
    return torch.cat(output)


def predict(local,composer,normalizer,batch,chunk=64,return_stages=False):
    cover=CartesianCover(*batch['solution'].shape[-2:],32,4)
    cond=tiled_sni_face_conditions(normalizer,batch,torch.zeros_like(batch['solution']),cover)
    first=local_call(local,cond.physical,cond.scalar,cover,chunk)*cond.mask*cond.valid
    ctx=boundary_input_context(normalizer,batch,cond,cover,include_target=False,trace_modes=4)
    code=composer(first,ctx.physical,ctx.scalar,ctx.domain_mask*ctx.valid,cover.grid_shape).correction_code
    physical,_=corrected_local_conditions(normalizer,ctx,code)
    second=local_call(local,physical,ctx.scalar,cover,chunk)*ctx.domain_mask*ctx.valid
    result=cover.assemble_weighted(second,len(batch['solution']))*batch['mask']
    return (result,first,second) if return_stages else result


def loss_for(prediction,batch,normalizer):
    target=batch['solution']/float(normalizer.statistics.solution_std)*batch['mask']
    return shared_field_loss(prediction,target,batch['mask'],batch['sdf'])[0]


def train_dataset(scale,seed):
    ds=H5DarcyDataset(data(scale),'train')
    ds.case_indices=read(manifest(scale,seed))['permutation'][:16];ds.length=16
    return ds


def batch_from(ds,indices,device):
    return move_batch(torch.utils.data.default_collate([ds[i] for i in indices]),device)


@torch.no_grad()
def evaluate(local,composer,normalizer,scale,split,device,run_id='',limit=None):
    assert split!='test' or (PACKAGE/'test_freeze.json').exists(), 'Test requires frozen selections'
    ds=H5DarcyDataset(data(scale),split)
    if limit:ds.length=limit
    rows=[]
    try:
        for host in DataLoader(ds,batch_size=2 if scale==8 else 1,shuffle=False,num_workers=0):
            batch=move_batch(host,device)
            pred=predict(local,composer,normalizer,batch)*float(normalizer.statistics.solution_std)
            num=((pred-batch['solution']).square()*batch['mask']).flatten(1).sum(1)
            den=(batch['solution'].square()*batch['mask']).flatten(1).sum(1)
            rel=(num/den.clamp_min(1e-12)).sqrt()
            assert torch.isfinite(rel).all()
            for k in range(len(rel)):
                rows.append(dict(run_id=run_id,domain_id=domain_id(scale,batch['seed'][k]),
                    relative_l2=float(rel[k]),error_l2=float(num[k].sqrt()),reference_l2=float(den[k].sqrt())))
    finally:ds.close()
    return rows


def prepare():
    if (PACKAGE/'protocol.json').exists():raise FileExistsError('Protocol already prepared')
    PACKAGE.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
    local,composer,norm,paths=build('local_adapter_only',torch.device('cpu'))
    count=sum(p.numel() for p in parameters(local,composer))
    assert abs(count/796400-1)<=.05
    stats=torch.load(LOCAL,map_location='cpu',weights_only=False)['statistics']
    save(PACKAGE/'normalizer.json',stats)
    source=torch.load(SOURCE,map_location='cpu',weights_only=False)
    ids=[];datasets={};orders={}
    provenance=read(BUDGET/'input_provenance.json')
    for scale in [8,16]:
        path=data(scale);digest=sha(path)
        assert digest==provenance[str(path.relative_to(ROOT))]['sha256']
        with h5py.File(path) as f:
            split_seeds={split:f[group]['seed'][:].tolist() for split,group in [('train','train'),('validation','val'),('test','test')]}
            assert all(len(v)==len(set(v))==64 for v in split_seeds.values())
            assert len(set(sum(split_seeds.values(),[])))==192
        datasets[f'{scale}x{scale}']={'path':str(path.resolve()),'sha256':digest,'geometry_seeds':split_seeds}
        for seed in SEEDS:
            m=read(manifest(scale,seed));assert m['dataset_sha256']==digest
            for split,values in split_seeds.items():
                indices=m['permutation'][:16] if split=='train' else range(64)
                for index in indices:ids.append(dict(scale=f'{scale}x{scale}',seed=seed,split=split,domain_id=domain_id(scale,values[index])))
            # Dedicated generator isolates sample order from module/adapter initialization.
            generator=torch.Generator().manual_seed(seed)
            order=torch.randint(16,(24,256),generator=generator).tolist()
            save(PACKAGE/f'orders/{scale}x{scale}_{seed}.json',order)
            orders[f'{scale}x{scale}_{seed}']={'sha256':sha(PACKAGE/f'orders/{scale}x{scale}_{seed}.json'),
                'manifest_sha256':sha(manifest(scale,seed)),'case_indices':m['permutation'][:16]}
    protocol=dict(experiment='darcy_adaptation_location',source_checkpoint=str(SOURCE.resolve()),
        source_checkpoint_sha256=sha(SOURCE),local_checkpoint=str(LOCAL.resolve()),local_checkpoint_sha256=sha(LOCAL),
        normalizer_sha256=sha(PACKAGE/'normalizer.json'),training_budget=16,scales=['8x8','16x16'],seeds=SEEDS,
        epochs=24,samples_per_epoch=256,sample_exposures=6144,batch_sizes={'8x8':2,'16x16':1},
        loss_weights=dict(field=1.,gradient=.2,near_boundary=.25,trace=0.,code_penalty=0.),
        adapter=dict(type='flattened_kernel_conv2d_lora',layers=list(paths),ranks=paths,alpha_over_rank=1.,
            initialization='Kaiming uniform A; zero B',trainable_parameters=count,relative_parameter_difference=count/796400-1),
        composition_trainable_parameters=796400,source_composer_configuration=source['args'],
        learning_rates=LRS,lr_selection='per arm minimum validation error over epochs; 8x8 m16 seed20260918; tie -> smaller LR',
        checkpoint_selection='minimum mean validation relative L2; first epoch wins ties',
        optimizer=dict(type='AdamW',weight_decay=1e-4,betas=[.9,.999],eps=1e-8,gradient_clip_norm=1.),
        schedule='CosineAnnealingLR T_max=24, eta_min=0.01*initial_lr; step after epoch',
        precision='bf16 autocast training with fp32 master weights and GradScaler; fp32 evaluation',
        forward='two shared local evaluations, same composer and normalized overlap assembly; core32 halo4 trace_modes4',
        backward='full gradient through both local calls and frozen composer in local_adapter_only; no detached preliminary response',
        local_tile_microbatch=64,activation_checkpointing='nonreentrant per local tile microbatch in both arms',
        relative_l2_definition='sqrt(sum(mask*(prediction_physical-target_physical)^2) / max(sum(mask*target_physical^2),1e-12)), per domain; mask=(stored mask>0.5); raw uncentered pressure; solution normalization divides by frozen source solution_std',
        norm_columns='error_l2 and reference_l2 are square roots of the unstabilized masked sums; denominator norm floor 1e-6',
        datasets=datasets,orders=orders,test_history='same reused test domains as budget study; previously inspected in that study, no test-driven selection in this ablation',
        inference_reference='common unadapted complete source checkpoint; stored separately from 768 adapted test rows',
        uncertainty='20000 paired draws, independently resample three seed rows and 64 domain columns, use same selections for both arms; pointwise 95% percentile CI; RNG 20260921',
        primary_effect='mean(local_adapter_only relative_l2 - composition_only relative_l2), separately at each scale',
        interpretation='CI entirely positive favors composition over this control; CI entirely negative favors local adapters; otherwise inconclusive; not a test of every local adaptation strategy',
        failed_run_policy='retain failure record and traceback; no dropping failed seeds; no inferential summary until all 12 complete',
        timing='CUDA-synchronized elapsed model initialization + training + validation + checkpoint writes, excludes queue and test; peak_memory_mb=max CUDA allocated bytes / 2^20',
        code_files={str(p.relative_to(ROOT)):sha(p) for folder in [DARCY,ROOT/'codiffuse_Schwarz',ROOT/'codiffuse_Schwarz_Airfoil/code_transfer'] for p in folder.glob('*.py')},
        prepared_at=time.time())
    save(PACKAGE/'protocol.json',protocol)
    write_csv(PACKAGE/'splits.csv',ids,['scale','seed','split','domain_id'])
    write_csv(PACKAGE/'runs.csv',[],RUN_COLUMNS);write_csv(PACKAGE/'test_errors.csv',[],TEST_COLUMNS)
    write_csv(PACKAGE/'validation_history.csv',[],HISTORY_COLUMNS)
    print(json.dumps({'trainable_parameters':count,'relative_difference':count/796400-1,'split_rows':len(ids)}))


def verify_protocol():
    p=read(PACKAGE/'protocol.json')
    assert sha(LOCAL)==p['local_checkpoint_sha256'] and sha(SOURCE)==p['source_checkpoint_sha256']
    assert sha(PACKAGE/'normalizer.json')==p['normalizer_sha256']
    for path,digest in p['code_files'].items():assert sha(ROOT/path)==digest,f'Code changed: {path}'
    for key,row in p['orders'].items():
        assert sha(PACKAGE/f'orders/{key}.json')==row['sha256']
        scale,seed=key.split('_')
        assert sha(manifest(int(scale.split('x')[0]),int(seed)))==row['manifest_sha256']
    freeze=PACKAGE/'implementation_freeze.json'
    if freeze.exists():
        assert read(freeze)['protocol_sha256']==sha(PACKAGE/'protocol.json')
        for path,digest in read(freeze)['files'].items():assert sha(ROOT/path)==digest,f'Experiment changed: {path}'
    return p


def preflight():
    p=verify_protocol();device=torch.device('cuda');results=[]
    torch.set_num_threads(2)
    for scale in [8,16]:
        ds=train_dataset(scale,SEEDS[0]);batch=batch_from(ds,list(range(2 if scale==8 else 1)),device);ds.close()
        source_local,source_composer,norm,_=build('zero_shot',device)
        from codiffuse_Schwarz_Darcy.field_token_composer import DarcyFieldTokenEncoder
        source_composer.encoder.forward=MethodType(DarcyFieldTokenEncoder.forward,source_composer.encoder)
        with torch.no_grad():
            expected=predict(source_local,source_composer,norm,batch)
            # Full source implementation, unchunked, no altered forward graph.
            from codiffuse_Schwarz_Darcy.train_input_correction_latentddm import prepared_context
            from codiffuse_Schwarz_Darcy.input_correction_latentddm import solve_corrected_boundary_input
            cover=CartesianCover(*batch['solution'].shape[-2:],32,4)
            first,ctx=prepared_context(source_local,norm,batch,cover,include_target=False,trace_modes=4)
            code=source_composer(first,ctx.physical,ctx.scalar,ctx.domain_mask*ctx.valid,cover.grid_shape).correction_code
            second,_=solve_corrected_boundary_input(source_local,norm,ctx,code)
            original=cover.assemble_weighted(second,len(batch['solution']))*batch['mask']
            torch.testing.assert_close(expected,original,rtol=1e-5,atol=2e-5)
        del source_local,source_composer,first,second,ctx,code
        for arm in ARMS:
            seed_everything(SEEDS[0]);local,composer,norm,paths=build(arm,device)
            before={f'{group}.{n}':q.detach().cpu().clone() for group,m in [('local',local),('composer',composer)] for n,q in m.named_parameters()}
            with torch.no_grad():
                initial=predict(local,composer,norm,batch)
                torch.testing.assert_close(initial,expected,rtol=1e-5,atol=2e-5)
            torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();t=time.perf_counter()
            with torch.autocast('cuda',dtype=torch.bfloat16):
                pred,first,second=predict(local,composer,norm,batch,return_stages=True)
                if arm=='local_adapter_only':
                    assert first.requires_grad and second.requires_grad
                    first.retain_grad();second.retain_grad()
                objective=loss_for(pred,batch,norm)
            objective.backward()
            if arm=='local_adapter_only':
                assert first.grad is not None and first.grad.abs().sum()>0
                assert second.grad is not None and second.grad.abs().sum()>0
            trainable=parameters(local,composer)
            assert all(q.grad is None or torch.isfinite(q.grad).all() for q in trainable)
            assert sum(float(q.grad.abs().sum()) for q in trainable if q.grad is not None)>0
            opt=torch.optim.AdamW(trainable,lr=1e-4,weight_decay=1e-4);opt.step()
            changed=[]
            for group,m in [('local',local),('composer',composer)]:
                for n,q in m.named_parameters():
                    key=f'{group}.{n}';allowed=(group=='composer') if arm=='composition_only' else (group=='local' and '.lora_' in n)
                    assert q.requires_grad==allowed,(arm,key)
                    if not torch.equal(q.detach().cpu(),before[key]):
                        assert allowed,key;changed.append(key)
                    if not allowed:assert q.grad is None
            assert changed
            results.append(dict(scale=f'{scale}x{scale}',arm=arm,initial_max_abs=float((initial-expected).abs().max()),
                original_forward_max_abs=float((expected-original).abs().max()),trainable_parameters=sum(q.numel() for q in trainable),
                changed_parameter_names=changed,first_and_second_gradient_checked=arm=='local_adapter_only',
                peak_memory_mb=torch.cuda.max_memory_allocated()/2**20,step_seconds=time.perf_counter()-t,loss=float(objective.detach())))
            print(json.dumps(results[-1]),flush=True)
            del local,composer,trainable,opt,before,pred,first,second,objective,initial
        del batch,expected,original
    save(PACKAGE/'preflight.json',dict(passed=True,checks=results,gpu=torch.cuda.get_device_name(),test_fields_scored=False,time=time.time()))


def train(run_id,arm,scale,seed,lr,stage):
    protocol=verify_protocol();assert read(PACKAGE/'preflight.json')['passed']
    target=OUT/stage/run_id
    if target.exists():raise FileExistsError(target)
    target.mkdir(parents=True)
    record=dict(run_id=run_id,arm=arm,scale=f'{scale}x{scale}',seed=seed,status='running',learning_rate=lr,
        trainable_parameters=None,selected_epoch=None,checkpoint_sha256=None,adaptation_seconds=None,peak_memory_mb=None,
        stage=stage,slurm_job_id=os.environ.get('SLURM_JOB_ID'),protocol_sha256=sha(PACKAGE/'protocol.json'))
    save(target/'run.json',record)
    history=[];ds=None
    try:
        device=torch.device('cuda');torch.set_num_threads(2);seed_everything(seed)
        torch.cuda.synchronize();start=time.perf_counter();torch.cuda.reset_peak_memory_stats()
        local,composer,norm,_=build(arm,device);params=parameters(local,composer)
        record.update(trainable_parameters=sum(q.numel() for q in params),gpu=torch.cuda.get_device_name())
        expected_count=protocol['composition_trainable_parameters'] if arm=='composition_only' else protocol['adapter']['trainable_parameters']
        assert record['trainable_parameters']==expected_count
        opt=torch.optim.AdamW(params,lr=lr,weight_decay=1e-4)
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,24,eta_min=lr*.01)
        scaler=torch.amp.GradScaler('cuda')
        ds=train_dataset(scale,seed);order=read(PACKAGE/f'orders/{scale}x{scale}_{seed}.json');batch_size=2 if scale==8 else 1
        best=float('inf');observed=[]
        for epoch,sequence in enumerate(order,1):
            losses=[]
            for begin in range(0,len(sequence),batch_size):
                batch=batch_from(ds,sequence[begin:begin+batch_size],device)
                observed.extend(batch['case_index'].cpu().tolist())
                opt.zero_grad(set_to_none=True)
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    pred=predict(local,composer,norm,batch);loss=loss_for(pred,batch,norm)
                if not torch.isfinite(loss):raise FloatingPointError('nonfinite training loss')
                scaler.scale(loss).backward();scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)
                scaler.step(opt);scaler.update();losses.append(float(loss.detach()))
            scheduler.step()
            rows=evaluate(local,composer,norm,scale,'validation',device)
            score=float(np.mean([r['relative_l2'] for r in rows]));assert len(rows)==64
            history.append(dict(run_id=run_id,epoch=epoch,mean_relative_l2=score,mean_training_loss=float(np.mean(losses))))
            save(target/'history.json',history)
            if score<best:
                best=score;record['selected_epoch']=epoch
                torch.save(dict(composer=composer.state_dict(),local=local.state_dict(),arm=arm,epoch=epoch,
                    score=score,protocol_sha256=record['protocol_sha256']),target/'checkpoint_best.pt')
            print(json.dumps(history[-1]),flush=True)
        expected=[ds.case_indices[k] for sequence in order for k in sequence]
        assert observed==expected
        save(target/'observed_sample_order.json',observed)
        record['sample_order_sha256']=sha(target/'observed_sample_order.json')
        torch.cuda.synchronize()
        record.update(status='completed',adaptation_seconds=time.perf_counter()-start,
            peak_memory_mb=torch.cuda.max_memory_allocated()/2**20,best_validation_relative_l2=best,
            checkpoint_sha256=sha(target/'checkpoint_best.pt'))
        save(target/'run.json',record)
    except BaseException:
        record['status']='failed';record['error_log']=str(target/'error.log')
        (target/'error.log').write_text(traceback.format_exc());save(target/'run.json',record);raise
    finally:
        if ds is not None:ds.close()


def screen(index):
    arm=ARMS[index//3];lr=LRS[index%3]
    train(f'screen_{arm}_lr{lr:g}',arm,8,SEEDS[0],lr,'screen')


def select():
    verify_protocol();rows=[read(OUT/'screen'/f'screen_{arm}_lr{lr:g}'/'run.json') for arm in ARMS for lr in LRS]
    assert all(r['status']=='completed' for r in rows)
    choices={arm:min([r for r in rows if r['arm']==arm],key=lambda r:(r['best_validation_relative_l2'],r['learning_rate'])) for arm in ARMS}
    assert not (PACKAGE/'selection.json').exists()
    save(PACKAGE/'selection.json',dict(choices=choices,trials=rows,test_fields_scored=False,time=time.time()))


def final_spec(index):
    arm=ARMS[index//6];scale=[8,16][(index%6)//3];seed=SEEDS[index%3]
    return f'{arm}_{scale}x{scale}_seed{seed}',arm,scale,seed


def final_train(index):
    run_id,arm,scale,seed=final_spec(index);lr=read(PACKAGE/'selection.json')['choices'][arm]['learning_rate']
    train(run_id,arm,scale,seed,lr,'final')


def freeze_test():
    verify_protocol();records=[]
    for index in range(12):
        run_id,arm,scale,seed=final_spec(index);folder=OUT/'final'/run_id;r=read(folder/'run.json')
        assert r['status']=='completed' and r['checkpoint_sha256']==sha(folder/'checkpoint_best.pt')
        assert r['learning_rate']==read(PACKAGE/'selection.json')['choices'][arm]['learning_rate']
        assert len(read(folder/'history.json'))==24
        records.append(r)
    for scale in [8,16]:
        for seed in SEEDS:
            paired=[r for r in records if r['scale']==f'{scale}x{scale}' and r['seed']==seed]
            assert len({r['sample_order_sha256'] for r in paired})==1
    assert not (PACKAGE/'test_freeze.json').exists()
    save(PACKAGE/'test_freeze.json',dict(runs=records,protocol_sha256=sha(PACKAGE/'protocol.json'),
        selection_sha256=sha(PACKAGE/'selection.json'),time=time.time()))


def test_run(index):
    verify_protocol();freeze=read(PACKAGE/'test_freeze.json');assert freeze['protocol_sha256']==sha(PACKAGE/'protocol.json')
    if index<12:
        run_id,arm,scale,seed=final_spec(index);folder=OUT/'final'/run_id
    else:
        scale=[8,16][index-12];arm='zero_shot';run_id=f'zero_shot_{scale}x{scale}';folder=OUT/'reference'/run_id
        folder.mkdir(parents=True,exist_ok=False)
    device=torch.device('cuda');torch.set_num_threads(2);local,composer,norm,_=build(arm,device)
    if index<12:
        r=next(r for r in freeze['runs'] if r['run_id']==run_id)
        assert sha(folder/'checkpoint_best.pt')==r['checkpoint_sha256']
        saved=torch.load(folder/'checkpoint_best.pt',map_location=device,weights_only=False)
        local.load_state_dict(saved['local'],strict=True);composer.load_state_dict(saved['composer'],strict=True)
    assert not (folder/'test_errors.csv').exists()
    rows=evaluate(local,composer,norm,scale,'test',device,run_id)
    assert len(rows)==64
    write_csv(folder/'test_errors.csv',rows,TEST_COLUMNS)
    print(run_id,float(np.mean([r['relative_l2'] for r in rows])),flush=True)


def collect():
    records=[];history=[];errors=[];references=[];screen_records=[];screen_history=[]
    for stage,rec,hist in [('final',records,history),('screen',screen_records,screen_history)]:
        for f in sorted((OUT/stage).glob('*/run.json')):
            r=read(f);rec.append(r)
            if (f.parent/'history.json').exists():hist.extend(read(f.parent/'history.json'))
            if stage=='final' and (f.parent/'test_errors.csv').exists():
                with (f.parent/'test_errors.csv').open() as stream:errors.extend(csv.DictReader(stream))
    for f in sorted((OUT/'reference').glob('*/test_errors.csv')):
        with f.open() as stream:references.extend(csv.DictReader(stream))
    write_csv(PACKAGE/'runs.csv',records,RUN_COLUMNS);write_csv(PACKAGE/'validation_history.csv',history,HISTORY_COLUMNS)
    write_csv(PACKAGE/'test_errors.csv',errors,TEST_COLUMNS);write_csv(PACKAGE/'zero_shot_errors.csv',references,TEST_COLUMNS)
    write_csv(PACKAGE/'screen_runs.csv',screen_records,RUN_COLUMNS);write_csv(PACKAGE/'screen_validation_history.csv',screen_history,HISTORY_COLUMNS)
    failed=[r for r in records+screen_records if r['status']=='failed'];save(PACKAGE/'failed_runs.json',failed)
    if len(errors)!=768 or len(references)!=128 or len(records)!=12 or any(r['status']!='completed' for r in records):
        save(PACKAGE/'status.json',dict(complete=False,final_runs=len(records),test_rows=len(errors),zero_shot_rows=len(references)));return
    result={};rng=np.random.default_rng(20260921)
    for scale in [8,16]:
        matrices=[];seed_means=[];expected_ids=None
        for arm in ARMS:
            matrix=[]
            for seed in SEEDS:
                run_id=f'{arm}_{scale}x{scale}_seed{seed}'
                rows=sorted([r for r in errors if r['run_id']==run_id],key=lambda r:r['domain_id'])
                assert len(rows)==64 and len({r['domain_id'] for r in rows})==64
                ids=[r['domain_id'] for r in rows]
                if expected_ids is None:expected_ids=ids
                assert ids==expected_ids
                for row in rows:
                    assert np.isclose(float(row['relative_l2']),float(row['error_l2'])/max(float(row['reference_l2']),1e-6),rtol=1e-5)
                matrix.append([float(r['relative_l2']) for r in rows])
                seed_means.append(dict(arm=arm,seed=seed,mean_relative_l2=float(np.mean(matrix[-1]))))
            matrices.append(np.array(matrix))
        diff=matrices[1]-matrices[0];draws=[]
        for _ in range(20000):
            si=rng.integers(0,3,3);di=rng.integers(0,64,64)
            draws.append(diff[np.ix_(si,di)].mean())
        low,high=np.quantile(draws,[.025,.975])
        result[f'{scale}x{scale}']=dict(seed_means=seed_means,seed_differences=diff.mean(1).tolist(),
            composition_mean=float(matrices[0].mean()),local_adapter_mean=float(matrices[1].mean()),
            delta=float(diff.mean()),ci95=[float(low),float(high)],
            interpretation='favors_composition' if low>0 else 'favors_local_adapter' if high<0 else 'inconclusive',
            zero_shot_mean=float(np.mean([float(r['relative_l2']) for r in references if r['run_id']==f'zero_shot_{scale}x{scale}'])))
    save(PACKAGE/'analysis.json',result);save(PACKAGE/'status.json',dict(complete=True,final_runs=12,test_rows=768,zero_shot_rows=128))
    print(json.dumps(result,indent=2))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['prepare','preflight','screen','select','train','freeze-test','test','collect']);parser.add_argument('--index',type=int,default=0)
    args=parser.parse_args()
    if args.phase in ['screen','train','test']:globals()[{'screen':'screen','train':'final_train','test':'test_run'}[args.phase]](args.index)
    elif args.phase=='collect':
        with (PACKAGE/'collection.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX);collect()
    else:globals()[args.phase.replace('-','_')]()

if __name__=='__main__':main()

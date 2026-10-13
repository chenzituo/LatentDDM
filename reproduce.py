#!/usr/bin/env python3
"""Small dispatcher: saved-result analysis, explicit reruns, and controlled ablation."""
from pathlib import Path
import argparse, csv, hashlib, json, os, runpy, shutil, subprocess, sys, time
P=Path(__file__).resolve().parent

def read(p): return json.loads(Path(p).read_text())
def save(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2)+'\n')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def rows(p):
 with Path(p).open(newline='') as f:return list(csv.DictReader(f))
def writecsv(p,rs):
 with Path(p).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
def map_path(value,assets):
 if isinstance(value,str) and value.startswith('/'):
  for tag in ['codiffuse_Schwarz_Darcy/','codiffuse_Schwarz_Airfoil/']:
   if tag in value:return str(assets/value[value.index(tag):])
 return value

def summarize(out):
 import numpy as np
 out.mkdir(parents=True,exist_ok=True)
 raw=rows(P/'results/budget/per_case.csv');result=[]
 original=read(P/'results/budget/results.json')
 rng=np.random.default_rng(20260919)
 si=rng.integers(3,size=(20000,3));di=rng.integers(64,size=(20000,64))
 for expected in original['estimates']:
  scale,budget,method=(expected[k] for k in ['scale','budget','method'])
  subset=[r for r in raw if int(r['scale'])==scale and int(r['budget'])==budget and r['method']==method]
  seeds=sorted({r['seed'] for r in subset});matrix=[]
  for seed in seeds:
   samples=sorted([r for r in subset if r['seed']==seed],key=lambda r:int(r['case_index']))
   assert [int(r['case_index']) for r in samples]==list(range(64))
   matrix.append([float(r['relative_l2']) for r in samples])
  x=np.array(matrix);means=x[0,di].mean(1) if len(x)==1 else x[si[:,:,None],di[:,None,:]].mean((1,2))
  ci=np.quantile(means,[.025,.975]);assert np.isclose(x.mean(),expected['mean'],rtol=0,atol=1e-12)
  np.testing.assert_allclose(ci,expected['ci95'],rtol=0,atol=1e-12)
  result.append(dict(scale=scale,budget=budget,method=method,mean=float(x.mean()),ci95_low=ci[0],ci95_high=ci[1]))
 writecsv(out/'darcy_budget.csv',result)
 errors=rows(P/'results/ablation/test_errors.csv');runs=rows(P/'results/ablation/runs.csv');assert len(errors)==768 and len(runs)==12
 lookup={r['run_id']:r for r in runs};original=read(P/'results/ablation/analysis.json');rng=np.random.default_rng(20260921);ab=[]
 for scale in ['8x8','16x16']:
  matrices=[];ids_ref=None
  for arm in ['composition_only','local_adapter_only']:
   matrix=[]
   for seed in ['20260918','20260919','20260920']:
    group=sorted([r for r in errors if lookup[r['run_id']]['scale']==scale and lookup[r['run_id']]['arm']==arm and lookup[r['run_id']]['seed']==seed],key=lambda r:r['domain_id'])
    ids=[r['domain_id'] for r in group];assert len(ids)==len(set(ids))==64
    if ids_ref is None:ids_ref=ids
    assert ids==ids_ref
    for r in group:assert np.isclose(float(r['relative_l2']),float(r['error_l2'])/max(float(r['reference_l2']),1e-6),rtol=1e-5)
    matrix.append([float(r['relative_l2']) for r in group])
   matrices.append(np.array(matrix))
  delta=matrices[1]-matrices[0];draws=[]
  for _ in range(20000):
   s=rng.integers(0,3,3);d=rng.integers(0,64,64);draws.append(delta[np.ix_(s,d)].mean())
  ci=np.quantile(draws,[.025,.975]);np.testing.assert_allclose(ci,original[scale]['ci95'],rtol=0,atol=1e-12)
  ab.append(dict(scale=scale,composition_mean=matrices[0].mean(),local_adapter_mean=matrices[1].mean(),delta=delta.mean(),ci95_low=ci[0],ci95_high=ci[1]))
 writecsv(out/'ablation.csv',ab)
 air=read(P/'results/airfoil/aggregate_zero_shot_test.json');table=[]
 for split,methods in air['rollout'].items():
  for method,steps in methods.items():
   for step in [1,5,10,20]:
    q=steps[f'step_{step:02d}']['field'];table.append(dict(split=split,method=method,horizon=step,mean=q['mean'],ci95_low=q['ci95_low'],ci95_high=q['ci95_high']))
 writecsv(out/'airfoil_zero_shot.csv',table)
 domain=read(P/'results/domain_extension/protected_test_summary.json');table=[]
 for scale,v in domain['scales'].items():
  for method,q in v['methods'].items():
   for state in ['zero_shot_error','adapted_error']:
    if state in q:table.append(dict(scale=scale,method=method,state=state,**q[state]))
 writecsv(out/'darcy_domain_extension.csv',table)
 save(out/'verification.json',dict(budget_means_and_intervals_recomputed=True,ablation_intervals_recomputed=True,ablation_norms_and_pairing_verified=True,airfoil_and_domain_tables='extracted from archived aggregate; not a new field evaluation',airfoil_calibration='excluded by author request'))
 print('Wrote four tables and verification.json to',out)

def launch(spec,assets,code):
 """Remap only historical path metadata in memory; never rewrite checkpoints."""
 sys.path.insert(0,str(code))
 import torch
 load=torch.load
 def remap(x):
  if isinstance(x,dict):return {k:remap(v) for k,v in x.items()}
  if isinstance(x,list):return [remap(v) for v in x]
  if isinstance(x,tuple):return tuple(remap(v) for v in x)
  if isinstance(x,Path):return Path(map_path(str(x),assets))
  return map_path(x,assets)
 def relocated_load(*a,**kw):return remap(load(*a,**kw))
 torch.load=relocated_load
 sys.argv=[spec['module']]+spec['args']
 runpy.run_module(spec['module'],run_name='__main__')

def main():
 parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
 p=sub.add_parser('list');p.add_argument('--filter',default='')
 p=sub.add_parser('summarize');p.add_argument('--output',type=Path,default=P/'build/tables')
 p=sub.add_parser('check-assets');p.add_argument('--assets',type=Path,required=True);p.add_argument('--sha256',action='store_true');p.add_argument('--recipe')
 p=sub.add_parser('run');p.add_argument('recipe');p.add_argument('--assets',type=Path,required=True);p.add_argument('--airfoil-data',type=Path);p.add_argument('--output',type=Path,required=True);p.add_argument('--execute',action='store_true');p.add_argument('--smoke',action='store_true');p.add_argument('--checkpoint',type=Path,help='Evaluate a newly trained checkpoint instead of the archived one')
 p=sub.add_parser('ablation');p.add_argument('phase',choices=['init','prepare','preflight','screen','select','train','freeze-test','test','collect']);p.add_argument('--assets',type=Path,required=True);p.add_argument('--work-dir',type=Path,required=True);p.add_argument('--index',type=int,default=0)
 p=sub.add_parser('_launch');p.add_argument('spec',type=Path);p.add_argument('--assets',type=Path,required=True);p.add_argument('--code',type=Path,default=P/'code')
 args=parser.parse_args()
 if args.command=='_launch':return launch(read(args.spec),args.assets.resolve(),args.code.resolve())
 if args.command=='summarize':return summarize(args.output.resolve())
 recipes=read(P/'recipes.json')
 if args.command=='list':
  for k in recipes:
   if args.filter in k:print(k)
  return
 assets=args.assets.resolve()
 if args.command=='check-assets':
  manifest=read(P/'assets.json');required=set(manifest)
  if args.recipe:required={s[len('{assets}/'):] for s in recipes[args.recipe]['args'] if s.startswith('{assets}/')}
  bad=[]
  for rel in sorted(required):
   q=manifest[rel];p=assets/rel
   if not p.is_file():bad.append((rel,'missing'))
   elif p.stat().st_size!=q['bytes']:bad.append((rel,'size mismatch'))
   elif args.sha256 and sha(p)!=q['sha256']:bad.append((rel,'hash mismatch'))
  print(json.dumps(dict(checked=len(required),failures=bad),indent=2));raise SystemExit(bool(bad))
 if args.command=='ablation':
  w=args.work_dir.resolve();root=w/'workspace';exp=root/'iclr2027/experiments/darcy_adaptation_location_20260921'
  if args.phase=='init':
   if root.exists():raise FileExistsError(root)
   shutil.copytree(P/'code',root)
   darcy=root/'codiffuse_Schwarz_Darcy';(darcy/'data').symlink_to(assets/'codiffuse_Schwarz_Darcy/data',target_is_directory=True)
   (darcy/'outputs').mkdir(exist_ok=True)
   for name in ['sni_face_local_seed20260806','input_correction_modes4_spatial4_trace0p5_depth4_warm_seed20260811']:
    (darcy/'outputs'/name).symlink_to(assets/'codiffuse_Schwarz_Darcy/outputs'/name,target_is_directory=True)
   budget=root/'iclr2027/experiments/darcy_boost_20260918';budget.mkdir(parents=True,exist_ok=True)
   shutil.copy2(P/'provenance/budget/input_provenance.json',budget/'input_provenance.json')
   for p in (P/'inputs/manifests').glob('*.json'):
    q=read(p);q['dataset']=map_path(q['dataset'],assets);save(budget/'manifests'/p.name,q)
   print('Initialized',root);return
  if not root.exists():raise FileNotFoundError('Run ablation init first')
  spec=dict(module='iclr2027.experiments.darcy_adaptation_location_20260921.experiment',args=[args.phase,'--index',str(args.index)])
  path=w/f'{args.phase}_{args.index}.command.json';save(path,spec)
  subprocess.run([sys.executable,str(P/'reproduce.py'),'_launch',str(path),'--assets',str(assets),'--code',str(root)],cwd=root,check=True);return
 rec=recipes[args.recipe];out=args.output.resolve()
 if out.exists() and args.execute:raise FileExistsError('Choose a fresh --output directory: '+str(out))
 mapping=dict(assets=str(assets),output=str(out),package=str(P),manifests=str(out/'manifests'),airfoil_data=str(args.airfoil_data.resolve()) if args.airfoil_data else 'AIRFOIL_DATA_REQUIRED')
 argv=[v.format(**mapping) for v in rec['args']]
 if args.checkpoint:
  if '/eval/' not in args.recipe:raise ValueError('--checkpoint is for evaluation recipes only')
  flag='--composer-checkpoint' if '--composer-checkpoint' in argv else '--checkpoint';argv[argv.index(flag)+1]=str(args.checkpoint.resolve())
 if args.smoke:
  if not args.recipe.startswith(('domain/eval/','budget/eval/')):raise ValueError('--smoke supports Darcy evaluation only; it is not a reported run')
  argv[argv.index('--max-cases')+1]='1';argv[argv.index('--workers')+1]='0';argv[argv.index('--batch-size')+1]='1'
 spec=dict(module=rec['module'],args=argv,recipe=args.recipe,smoke=args.smoke,provenance=rec['provenance'])
 print(json.dumps(spec,indent=2),flush=True)
 if not args.execute:return
 if args.recipe.startswith('airfoil/') and not args.airfoil_data:raise ValueError('--airfoil-data must name the directory containing paired L5 and L6 HDF5 files')
 out.mkdir(parents=True)
 for p in (P/'inputs/manifests').glob('*.json'):
  q=read(p);q['dataset']=map_path(q['dataset'],assets);save(out/'manifests'/p.name,q)
 save(out/'command.json',spec)
 env=os.environ.copy();env.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONUNBUFFERED='1')
 start=time.time()
 completed=subprocess.run([sys.executable,str(P/'reproduce.py'),'_launch',str(out/'command.json'),'--assets',str(assets)],cwd=out,env=env)
 save(out/'status.json',dict(exit_code=completed.returncode,wall_seconds=time.time()-start,smoke=args.smoke))
 raise SystemExit(completed.returncode)
if __name__=='__main__':main()

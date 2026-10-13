"""Check paired inference against an analytically known constant treatment effect."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
import experiment as e


def test_collection_preserves_domain_and_seed_pairing(tmp_path,monkeypatch):
    monkeypatch.setattr(e,'PACKAGE',tmp_path/'package')
    monkeypatch.setattr(e,'OUT',tmp_path/'out')
    for index in range(12):
        run_id,arm,scale,seed=e.final_spec(index)
        directory=e.OUT/'final'/run_id
        e.save(directory/'run.json',dict(run_id=run_id,arm=arm,scale=f'{scale}x{scale}',seed=seed,status='completed'))
        e.save(directory/'history.json',[dict(run_id=run_id,epoch=epoch,mean_relative_l2=.1) for epoch in range(1,25)])
        # Large shared seed/domain variation cancels only if pairing is preserved.
        rows=[]
        for domain in range(64):
            value=.1+(seed-e.SEEDS[0])*.2+domain*.01+(.05 if arm=='local_adapter_only' else 0)
            rows.append(dict(run_id=run_id,domain_id=f'{scale}:{domain:03d}',relative_l2=value,error_l2=value,reference_l2=1.))
        # Deliberately reverse one arm's file order.
        if arm=='local_adapter_only':rows.reverse()
        e.write_csv(directory/'test_errors.csv',rows,e.TEST_COLUMNS)
    for scale in [8,16]:
        run_id=f'zero_shot_{scale}x{scale}'
        e.write_csv(e.OUT/'reference'/run_id/'test_errors.csv',[
            dict(run_id=run_id,domain_id=f'{scale}:{i:03d}',relative_l2=.9,error_l2=.9,reference_l2=1.) for i in range(64)],e.TEST_COLUMNS)
    e.collect()
    assert e.read(e.PACKAGE/'status.json')['test_rows']==768
    result=e.read(e.PACKAGE/'analysis.json')
    for row in result.values():
        assert np.isclose(row['delta'],.05)
        assert np.allclose(row['ci95'],[.05,.05])
        assert row['interpretation']=='favors_composition'
        assert np.allclose(row['seed_differences'],[.05]*3)

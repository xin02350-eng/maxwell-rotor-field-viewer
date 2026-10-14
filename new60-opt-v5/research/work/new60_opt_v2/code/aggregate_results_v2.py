"""Summarize frozen-model independent QA without selecting or training models."""
import json, csv
from pathlib import Path
from datetime import datetime, timezone
ROOT=Path(__file__).resolve().parent.parent
def main():
    choices=json.loads((ROOT/'both_versions_choices_frozen.json').read_text(encoding='utf8'))
    methods={m['id']:m for m in choices['models']}
    reports=[json.loads((ROOT/f'qa/{b}_replay/replay_report.json').read_text(encoding='utf8')) for b in ['prediction','fitting']]
    assert all(r['all_checks_pass'] for r in reports)
    summary=dict(version='NEW60_OPT_V2',created_utc=datetime.now(timezone.utc).isoformat(),
      choices_sha256=reports[0]['choices_sha256'],source_sha256=reports[0]['source_sha256'],
      independent_numerical_checks=sum(len(r['checks']) for r in reports),all_numerical_checks_pass=True,
      cli_replay=json.loads((ROOT/'qa/cli_v2_check.json').read_text(encoding='utf8')),
      prediction_scope='42 training / 9 validation / 9 reused test; V1 test already seen; no fresh blind test',
      fitting_scope='All60 training; scores are in-sample fitting errors only',
      physical_quantity='Rotor/stator sqrt(Br^2+Bt^2), planar, missing Bz; airgap signed Br',
      metric_weighting='Equal original point/component weights, not FEM area weights',
      metric_definition=dict(RMSE='sqrt(mean((prediction-truth)^2)) / T',MAE='mean(abs(prediction-truth)) / T',
        max_abs='max(abs(prediction-truth)) / T',relative_L2_pct='100*sqrt(sum((prediction-truth)^2)/sum(truth^2))'),
      limitations=['Known fixed60 measured rotor positions and original sample points only.',
        'No claim of arbitrary new angles, new spatial points, other speeds or range-external accuracy.',
        'Large local errors remain in prediction; inspect maximum error and worst-case plots.',
        'Fitting error cannot be used as prediction error on untrained conditions.',
        '3D plots are field-height maps, not volumetric magnetic-field reconstructions.',
        'Prediction choices were frozen before V2 reused-test evaluation and were not changed using its results.'],
      models=[])
    rows=[]
    for report in reports:
      for result in report['results']:
        frozen=methods[result['id']]
        item={k:result[k] for k in ['id','branch','region','quantity','model_sha256']}
        item.update(role=frozen['role'],method=frozen['method'],metrics={})
        for partition, detail in result['metrics'].items():
          metric=detail['global_metrics'].copy(); pc=detail['per_case']; i=max(range(len(pc['case_ids'])),key=lambda i:pc['rmse_T'][i])
          metric.update(worst_case_id=pc['case_ids'][i],worst_case_rmse_T=pc['rmse_T'][i],worst_case_relative_L2_pct=pc['relative_L2_pct'][i])
          item['metrics'][partition]=metric
          rows.append(dict(model_id=item['id'],branch=item['branch'],region=item['region'],quantity=item['quantity'],partition=partition,
            **{k:metric[k] for k in ['rmse_T','mae_T','max_abs_T','relative_L2_pct','worst_case_id','worst_case_rmse_T','worst_case_relative_L2_pct']}))
        summary['models'].append(item)
    baseline=ROOT/'figures/v1_v2_same_quantity_comparison.json'
    if baseline.exists():summary['same_quantity_v1_v2']=json.loads(baseline.read_text(encoding='utf8'))
    summary['additional_audits']={}
    for name,rel in [('saved_scientific_arrays','qa/scientific_npz_spotchecks.json'),('optimization_inventory','qa/optimization_inventory.json'),('website_payloads','qa/website_payload_check.json'),('browser_interaction','qa/browser_v2/browser_v2_check.json')]:
      path=ROOT/rel
      if path.exists():
        q=json.loads(path.read_text(encoding='utf8'))
        passed=any(q.get(k) is True for k in ['all_pass','all_checks_pass','passed','pass','verified']) or str(q.get('status','')).upper()=='PASS'
        assert passed,(name,'audit not passed')
        summary['additional_audits'][name]=dict(path=rel,passed=passed,check_count=q.get('check_count',len(q.get('checks',[]))))
        if name=='optimization_inventory':summary['optimization_inventory']=q
    summary['freeze_timing_note']='Neural prediction route frozen before all60 fitting; all60 fitting independent and not fed back into prediction choice. Final global choices frozen before V2 old-test re-evaluation.'
    (ROOT/'final_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    with (ROOT/'final_summary.csv').open('w',encoding='utf-8-sig',newline='') as f:
      w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(json.dumps(dict(models=len(rows),numerical_checks=summary['independent_numerical_checks'],passed=True)))
if __name__=='__main__':main()

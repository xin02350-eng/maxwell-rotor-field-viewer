"""Independent all-payload numeric audit of V2 website; no model inference."""
from pathlib import Path
from collections import Counter
import base64
import gzip
import hashlib
import json
import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]


def read(p):
    return json.loads(p.read_text(encoding="utf-8-sig"))


def sha(p):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(8*1024*1024), b""):
            h.update(b)
    return h.hexdigest()


def metrics(t, p, axes=None):
    e = p-t; sse = np.sum(e*e, axis=axes); energy = np.sum(t*t, axis=axes)
    return dict(rmse_T=np.sqrt(np.mean(e*e, axis=axes)), mae_T=np.mean(np.abs(e), axis=axes),
                max_abs_T=np.max(np.abs(e), axis=axes), relative_L2_pct=100*np.sqrt(sse/energy))


def views(t, p, quantity):
    if quantity == "Br,Bt":
        return dict(Br=(t[..., 0::2],p[..., 0::2]), Bt=(t[..., 1::2],p[..., 1::2]),
                    magnitude=(np.hypot(t[...,0::2],t[...,1::2]),np.hypot(p[...,0::2],p[...,1::2])))
    return {"Br" if quantity == "Br" else "magnitude": (t, p)}


def main():
    out = ROOT / "qa/website_payload_check.json"
    if out.exists():
        raise FileExistsError(out)
    web = ROOT / "website"; manifest = read(web / "manifest.json"); precision = read(web / "display_precision.json")
    results = read(web / "results.json"); freeze = read(ROOT / "both_versions_choices_frozen.json")
    frozen = {r["id"]: r for r in freeze["models"]}; expected_sha = sha(ROOT / "both_versions_choices_frozen.json")
    exact_reports = [ROOT / "qa/prediction_replay/replay_report.json", ROOT / "qa/fitting_replay/replay_report.json"]
    exact = {r["id"]: r for p in exact_reports for r in read(p)["results"]}
    precision_map = {p["path"]:p for p in precision["payloads"]}; results_map = {r["id"]:r for r in results["models"]}
    check_counts = Counter(); failures = []; audited = []; seen = set()
    def check(name, yes, detail=None):
        check_counts[name] += 1
        if not yes:
            failures.append(dict(check=name, detail=detail))
    def metric_match(expected, recorded):
        return all(np.allclose(expected[k], recorded[k], atol=2e-12, rtol=2e-12) for k in expected)
    check("global freeze hash", manifest["choices_sha256"] == results["choices_sha256"] == expected_sha)
    check("explicit evaluation limitations", manifest["previous_test_already_seen"] is True and manifest["test_is_fresh_blind"] is False)
    cases = {r["case_id"]:r for r in manifest["cases"]}
    check("exact sixty unique source cases", len(cases) == 60 and set(cases) == set(range(60)))
    with threadpool_limits(limits=2):
      for mode, section in manifest["modes"].items():
       for region, r in section["regions"].items():
        geometry = read(web/r["geometry"])
        for model in r["models"]:
            mid = model["key"]; row = frozen[mid]; quantity = model["quantity"]
            check("model identity", model["model_sha256"] == row["model_sha256"] and mode == row["branch"] and region == row["region"], mid)
            check("all60 manifest references", set(r["cases"]) == {str(i) for i in range(60)} and all(mid in c for c in r["cases"].values()), mid)
            count = 0; maximum_error = {"truth":0., "prediction":0.}; maximum_step = {"truth":0., "prediction":0.}
            value_ranges = {}
            for partition, source in exact[mid]["scientific_artifacts"].items():
                path = Path(source["path"])
                check("scientific source SHA", sha(path) == source["sha256"], mid+"/"+partition)
                with np.load(path,allow_pickle=False) as z:
                    truth=z["truth"]; prediction=z["prediction"]; ids=z["case_ids"]
                    check("full precision geometry", np.array_equal(z["point_id"],geometry["point_id"]) and np.array_equal(z["xy"],geometry["xy"]), mid)
                    check("full precision frame coordinates", np.array_equal(z["theta"],manifest["frames"]["theta_absolute_deg"]) and np.array_equal(z["time_s"],manifest["frames"]["time_s"]), mid)
                    check("case native inputs", all(cases[int(cid)]["I_rms_A"] == float(z["conditions"][i,0]) and cases[int(cid)]["Thet_deg"] == float(z["conditions"][i,1]) for i,cid in enumerate(ids)),mid)
                    # Whole partition numbers on results.json are independent of all q16 arrays.
                    partition_views = views(truth,prediction,quantity)
                    if quantity == "Br,Bt":
                        partition_views["signed_vector"]=(truth,prediction)
                    for key,(t,p) in partition_views.items():
                        check("unquantized partition metric", metric_match(metrics(t,p),results_map[mid]["metrics"][partition][key]), mid+"/"+partition+"/"+key)
                    for i,cid in enumerate(ids):
                        cid=int(cid); relative=r["cases"][str(cid)][mid]; payload_path=(web/relative).resolve()
                        check("payload path contained in website", payload_path.is_relative_to(web.resolve()), relative)
                        check("unique payload reference", relative not in seen, relative); seen.add(relative)
                        raw=payload_path.read_bytes(); payload=json.loads(gzip.decompress(raw)); prec=precision_map[relative]
                        check("gzip SHA and length", hashlib.sha256(raw).hexdigest()==prec["sha256"] and len(raw)==prec["bytes"],relative)
                        check("payload identity and physical layout", payload["case_id"]==cid and payload["model_key"]==mid and payload["mode"]==mode and payload["region"]==region
                            and payload["model_sha256"]==row["model_sha256"] and payload["quantity"]==quantity and payload["units"]=="T"
                            and payload["shape"]==list(truth[i].shape) and payload["layout"]==("interleaved" if quantity=="Br,Bt" else "scalar")
                            and payload["point_id"]==geometry["point_id"] and payload["original_split"]==cases[cid]["split"],relative)
                        for name, field in (("truth",truth[i]),("prediction",prediction[i])):
                            encoded=base64.b64decode(payload[name+"_q16_b64"],validate=True)
                            q=np.frombuffer(encoded,dtype="<u2")
                            check("q16 exact scalar count",q.size==field.size,relative+"/"+name)
                            q=q.reshape(field.shape);lo=float(payload["offset"][name]);step=float(payload["scale"][name]);back=q.astype(float)*step+lo
                            observed=float(np.max(np.abs(back-field))); bound=step/2+2e-14
                            check("q16 absolute reconstruction bound",observed<=bound,dict(path=relative,array=name,max_abs_T=observed,bound_T=bound))
                            expected_step=(float(field.max())-float(field.min()))/65535 if field.max()>field.min() else 1e-12
                            expected_q=np.rint((field-lo)/step).clip(0,65535).astype("<u2")
                            check("q16 offset step and nearest rounding",lo==float(field.min()) and step==expected_step and np.array_equal(q,expected_q),relative+"/"+name)
                            check("precision record matches observed",abs(observed-prec[name+"_max_abs_T"])<2e-14 and abs(observed-payload["quantization_max_abs_T"][name])<2e-14
                                  and step==prec[name+"_step_T"],relative+"/"+name)
                            maximum_error[name]=max(maximum_error[name],observed);maximum_step[name]=max(maximum_step[name],step)
                        component_views=views(truth[i],prediction[i],quantity)
                        check("metric component set",set(component_views)==set(payload["metrics"]),relative)
                        for key,(t,p) in component_views.items():
                            m=payload["metrics"][key]
                            check("unquantized case metrics",metric_match(metrics(t,p),m["aggregate"]),relative+"/"+key)
                            computed=metrics(t,p,axes=1);recorded={k:np.asarray([x[k] for x in m["frames"]]) for k in computed}
                            check("unquantized sixty frame metrics",len(m["frames"])==60 and metric_match(computed,recorded),relative+"/"+key)
                            lo=min(float(t.min()),float(p.min()),0.);hi=max(float(t.max()),float(p.max()));err=float(np.abs(p-t).max())
                            previous=value_ranges.setdefault(key,[lo,hi,err]);previous[:]=[min(previous[0],lo),max(previous[1],hi),max(previous[2],err)]
                        count+=1
            check("one model sixty cases",count==60,mid)
            for key,(lo,hi,error) in value_ranges.items():
                if key in {"Br","Bt"}:
                    hi=max(abs(lo),abs(hi),1e-12);lo=-hi
                check("fixed displayed color ranges cover original fields",np.array_equal(model["scales"][key]["field"],[lo,max(hi,lo+1e-12)]) and np.array_equal(model["scales"][key]["error"],[0,max(error,1e-12)]),mid+"/"+key)
            audited.append(dict(id=mid,cases=count,quantity=quantity,maximum_quantization_error_T=maximum_error,maximum_quantization_step_T=maximum_step))
    check("eleven models and660 packs",len(audited)==11 and len(seen)==660 and precision["payload_count"]==660 and len(precision_map)==660)
    check("payload files equal manifest references",{str(p.relative_to(web)).replace('\\','/') for p in (web/'data').rglob('*.json.gz')}==seen)
    result=dict(all_pass=not failures,check_count=sum(check_counts.values()),check_categories=dict(check_counts),failures=failures,
                models=audited,payload_count=len(seen),choices_sha256=expected_sha,checker_sha256=sha(Path(__file__)),
                website_files_sha256={name:sha(web/name) for name in ('manifest.json','results.json','display_precision.json')},
                no_inference_training_or_selection=True,metrics_from_unquantized_sources=True,
                limitations="All data packs checked numerically. This audit does not claim native browser rendering/interaction testing; separate component/browser reports apply.")
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    lines=['# V2 网页数据独立数值核验','',('PASS' if result['all_pass'] else 'FAIL'),'',
           f"逐包核验 11 模型 × 60 工况 = 660 个 gzip JSON，累计 {result['check_count']:,} 项检查。",
           '', '核对冻结模型/工况/物理量/BrBt排列/point ID/几何/原生输入/时间角度；uint16小端还原与原float64阵列逐元素比较，验证最近舍入和半步长误差上限。',
           '', '全部总体、逐工况、60帧指标重新从科学NPZ的未量化值计算，与网页数据逐项比较。网页颜色范围也对原精度真值/预测核验。',
           '', '仅本报告不代表浏览器交互或截图实测；另有组件和浏览器检查。没有重新推理、训练、选型或修改数据。']
    if failures:lines+=['',json.dumps(failures,ensure_ascii=False,indent=2)]
    out.with_suffix('.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(all_pass=result['all_pass'],checks=result['check_count'],payload_count=len(seen),failures=len(failures))))
    return 0 if result['all_pass'] else 2


if __name__=='__main__':
    raise SystemExit(main())

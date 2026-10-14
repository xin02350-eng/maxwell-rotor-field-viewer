"""Independent streaming ZIP, report, and relocated inference acceptance.

This file deliberately does not import the delivery builder or training code.
Run only after the final ZIP is complete. It does not extract the full study,
train, select models, or materialize magnetic-field labels during CLI checks.
"""
from __future__ import annotations

import argparse
import csv
from decimal import Decimal
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import subprocess
import sys
import time
import traceback
import xml.etree.ElementTree as ET
import zipfile
import zlib

import numpy as np

HERE = Path(__file__).resolve().parent
V2 = HERE.parent
WORKSPACE = V2.parents[1]
DEFAULT_ZIP = WORKSPACE / "outputs/沈夏鑫_新60双版本优化_20261014.zip"
CHOICES_SHA = "c26dcdb4213e5afe09d78cf301e08e34234be2c1bf78a7aa32eab5673c4174c0"
PREDICTION_SHA = "2e48445d085f07df328c678e4fe7c309213c19d7b67a003a0ed981bf0e6208bb"
V1_NAME, V2_NAME = "new60_20261014", "new60_opt_v2"
CHUNK = 4 * 1024 * 1024
INPUT_KEYS = ("conditions", "case_ids", "point_id", "xy", "theta", "time_s", "solver_index", "theta_m_relative_deg")
FORBIDDEN_DIRS = {".git", ".venv", "venv", "site-packages", "node_modules", "__pycache__", ".pytest_cache", "adaptive_pilot", "portable_delivery_check"}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(CHUNK), b""):
            h.update(b)
    return h.hexdigest()


class Audit:
    def __init__(self):
        self.checks = []

    def check(self, name, passed, **detail):
        self.checks.append(dict(name=name, passed=bool(passed), **detail))
        if not passed:
            print("FAIL:", name, json.dumps(detail, ensure_ascii=False)[:800], flush=True)
        return bool(passed)

    def close(self, name, actual, expected, atol=2e-12, rtol=2e-12):
        a, b = np.asarray(actual), np.asarray(expected)
        same = a.shape == b.shape
        delta = float(np.max(np.abs(a - b))) if same and a.size else (0.0 if same else None)
        return self.check(name, same and np.all(np.isfinite(a)) and np.allclose(a, b, atol=atol, rtol=rtol),
                          shape=list(a.shape), expected_shape=list(b.shape), max_abs_difference=delta, atol=atol, rtol=rtol)


def safe_relative(name):
    p = PurePosixPath(name)
    if not name or "\\" in name or p.is_absolute() or ":" in name or any(x in ("", ".", "..") for x in name.rstrip("/").split("/")):
        raise ValueError("Unsafe member or relative path: " + name)
    return p


def load_json(z, name):
    return json.loads(z.read(name).decode("utf-8-sig"))


def stream_archive(z, archive, audit, report):
    infos = z.infolist()
    names = [i.filename for i in infos]
    if not audit.check("unique_member_names", len(names) == len(set(names)) and len(names) == len({s.casefold() for s in names}), count=len(names)):
        raise ValueError("Duplicate or case-insensitive duplicate ZIP members")
    roots = set()
    for i in infos:
        p = safe_relative(i.filename)
        roots.add(p.parts[0])
        if stat.S_ISLNK(i.external_attr >> 16):
            raise ValueError("Symlink in archive: " + i.filename)
    if not audit.check("one_named_archive_root", roots == {archive.stem}, roots=sorted(roots)):
        raise ValueError("Archive root layout is not the specified named directory")
    top = roots.pop()
    manifest_name = top + "/delivery_manifest.json"
    manifest = load_json(z, manifest_name)
    audit.check("manifest_self_excluded", manifest.get("manifest_self_excluded") is True)
    audit.check("manifest_archive_root", manifest.get("archive_root") == top)
    records = manifest["records"]
    expected = {}
    for r in records:
        rel = str(safe_relative(r.get("path", r.get("relative_path"))))
        if rel in expected:
            raise ValueError("Duplicate manifest row: " + rel)
        expected[rel] = r
    actual_names = {i.filename[len(top) + 1:] for i in infos if not i.is_dir()}
    audit.check("manifest_exact_file_set", actual_names == set(expected) | {"delivery_manifest.json"},
                missing=sorted(set(expected) - actual_names), unlisted=sorted(actual_names - set(expected) - {"delivery_manifest.json"}))
    audit.check("sibling_study_layout", {PurePosixPath(n).parts[0] for n in actual_names if "/" in n} == {V1_NAME, V2_NAME})
    forbidden = [n for n in actual_names if any(p.lower() in FORBIDDEN_DIRS or p.lower().startswith(("synthetic_fixture", "toy_fixture", "failedrun", "failed_run")) for p in PurePosixPath(n).parts)]
    audit.check("no_runtime_or_incomplete_pilot", not forbidden, matches=forbidden)
    incomplete = []
    for n in actual_names:
        pp = PurePosixPath(n).parts
        if pp[0] == V2_NAME and "runs" in pp:
            idx = pp.index("runs")
            if len(pp) > idx + 2:
                completion = "/".join(pp[:idx + 2]) + "/result.json"
                if completion not in actual_names:
                    incomplete.append(completion)
    audit.check("all_packaged_v2_neural_runs_completed", not incomplete, missing_result_files=sorted(set(incomplete)))
    digests = {}
    start = time.time()
    for number, info in enumerate(infos, 1):
        if info.is_dir():
            continue
        h, crc, n = hashlib.sha256(), 0, 0
        with z.open(info, "r") as f:
            for block in iter(lambda: f.read(CHUNK), b""):
                h.update(block)
                crc = zlib.crc32(block, crc)
                n += len(block)
        rel = info.filename[len(top) + 1:]
        d = dict(path=rel, bytes=n, sha256=h.hexdigest(), crc32=f"{crc & 0xffffffff:08x}")
        digests[rel] = d
        target = expected.get(rel)
        ok = n == info.file_size and crc & 0xffffffff == info.CRC
        if target:
            ok = ok and n == target.get("bytes", target.get("size_bytes")) and d["sha256"] == target["sha256"]
        audit.check("member:" + rel, ok, **d)
        if number % 250 == 0:
            print(f"Streamed {number}/{len(infos)} members; {time.time() - start:.1f}s", flush=True)
    byte_total = sum(digests[n]["bytes"] for n in expected)
    audit.check("manifest_totals", manifest.get("file_count") == len(expected) and manifest.get("bytes") == byte_total,
                actual_files=len(expected), actual_bytes=byte_total)
    audit.check("manifest_frozen_identity", manifest.get("choices_sha256") == CHOICES_SHA)
    v1_path = V1_NAME + "/delivery_manifest.json"
    old = load_json(z, top + "/" + v1_path)
    old_rows = old["records"]
    old_names = {str(safe_relative(r.get("path", r.get("relative_path")))) for r in old_rows} | {"delivery_manifest.json"}
    actual_v1 = {n[len(V1_NAME) + 1:] for n in actual_names if n.startswith(V1_NAME + "/")}
    audit.check("v1_original_exact_file_set", old_names == actual_v1, count=len(actual_v1))
    old_mismatch = []
    for r in old_rows:
        k = V1_NAME + "/" + r.get("path", r.get("relative_path"))
        d = digests.get(k, {})
        if d.get("sha256") != r["sha256"] or d.get("bytes") != r.get("bytes", r.get("size_bytes")):
            old_mismatch.append(k)
    audit.check("v1_all_original_hashes_preserved", not old_mismatch, mismatch=old_mismatch)
    v1info = manifest.get("v1", {})
    audit.check("v1_manifest_identity", digests[v1_path]["sha256"] == v1info.get("manifest_sha256"))
    report["archive_members"] = dict(file_count=len(digests), uncompressed_bytes=sum(r["bytes"] for r in digests.values()),
                                      manifest_sha256=digests["delivery_manifest.json"]["sha256"], v1_preserved_files=len(old_names), stream_seconds=time.time() - start)
    return top, manifest, digests


def verify_frozen(z, top, digests, audit):
    prefix = V2_NAME + "/"
    choices = load_json(z, top + "/" + prefix + "both_versions_choices_frozen.json")
    for name, expected in (("both_versions_choices_frozen.json", CHOICES_SHA), ("prediction_choices_frozen.json", PREDICTION_SHA)):
        audit.check("frozen_identity:" + name, digests[prefix + name]["sha256"] == expected)
        sidecar = z.read(top + "/" + prefix + str(PurePosixPath(name).with_suffix(".sha256"))).decode().strip().split()[0]
        audit.check("frozen_sidecar:" + name, sidecar == expected)
    audit.check("protocol_frozen_identity", digests[prefix + "protocol.json"]["sha256"] == choices["protocol_sha256"])
    audit.check("separate_scientific_scopes", choices.get("previous_test_already_seen") is True and choices.get("full60_fitting_scores_used_in_prediction_choice") is False and choices.get("full60_fitting_authorized") is True)
    ids = [r["id"] for r in choices["models"]]
    audit.check("eleven_unique_frozen_models", len(ids) == 11 and len(ids) == len(set(ids)))
    for row in choices["models"]:
        for pkey, hkey in (("model_relative_path", "model_sha256"), ("inference_module_relative_path", "inference_module_sha256"), ("model_source_module_relative_path", "model_source_module_sha256")):
            if pkey in row:
                rel = prefix + row[pkey]
                audit.check("frozen_artifact:" + row["id"] + ":" + pkey, digests.get(rel, {}).get("sha256") == row[hkey], path=rel)
        for rec in row.get("provenance_files", []):
            rel = prefix + rec["relative_path"]
            audit.check("frozen_provenance:" + row["id"] + ":" + rel, digests.get(rel, {}).get("sha256") == rec["sha256"])
        cache = V1_NAME + "/cache_v1/" + row["region"] + ".npz"
        audit.check("frozen_cache:" + row["id"], digests[cache]["sha256"] == row["cache_sha256"])
        audit.check("frozen_split:" + row["id"], digests[V1_NAME + "/cache_v1/split.json"]["sha256"] == row["split_sha256"])
    return choices


def document_check(z, top, audit, report):
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    with zipfile.ZipFile(io.BytesIO(z.read(top + "/两版本说明.docx"))) as word:
        document = ET.fromstring(word.read("word/document.xml"))
    tables = []
    for table in document.findall(".//w:tbl", ns):
        rows = [["".join(c.itertext()) for c in tr.findall("w:tc", ns)] for tr in table.findall("w:tr", ns)]
        tables.append(rows)
    # itertext is suitable for the builder's simple cells; also remove XML indentation.
    text = "\n".join(t.text or "" for t in document.findall(".//w:t", ns))
    material_member = top + "/" + V2_NAME + "/figures/optim报告材料.md"
    has_appendix = material_member in z.namelist()
    audit.check("word_exact_table_count", len(tables) == (7 if has_appendix else 5), count=len(tables))
    expected_widths, expected_heights = [4, 5, 4, 5, 5, 7, 5], [4, 4, 4, 9, 9, 7, 6]
    for i, table in enumerate(tables):
        audit.check(f"word_table_{i}_dimensions", i < 7 and len(table) == expected_heights[i] and all(len(r) == expected_widths[i] for r in table),
                    row_count=len(table), row_widths=[len(r) for r in table])
    results = load_json(z, top + "/" + V2_NAME + "/website/results.json")
    summary = load_json(z, top + "/" + V2_NAME + "/final_summary.json")
    raw = load_json(z, top + "/" + V2_NAME + "/neural_route/fitting_full60/fitting_summary.json")
    audit.check("document_result_freeze", results["choices_sha256"] == summary["choices_sha256"] == CHOICES_SHA)
    summary_models = {m["id"]: m for m in summary["models"]}
    for ti, branch, part in ((1, "fitting_full60", "training_fit"), (3, "prediction", "validation"), (4, "prediction", "reused_test")):
        expected_rows = []
        for model in results["models"]:
            if model["branch"] != branch or part not in model["metrics"]:
                continue
            key = "signed_vector" if model["quantity"] == "Br,Bt" else "Br" if model["quantity"] == "Br" else "magnitude"
            m = model["metrics"][part][key]
            expected_rows.append([model["id"] + " / " + model["quantity"], model["region"], f"{m['rmse_T']:.9g}", f"{m['relative_L2_pct']:.8g}", f"{m['max_abs_T']:.9g}"])
            sm = summary_models[model["id"]]["metrics"][part]
            gm = sm.get("global_metrics", sm)
            for metric in ("rmse_T", "relative_L2_pct", "max_abs_T"):
                audit.close("word_source_matches_scientific_summary:" + model["id"] + ":" + part + ":" + metric, m[metric], gm[metric], atol=1e-14, rtol=1e-13)
        audit.check(f"word_metric_table_{ti}_exact_values", ti < len(tables) and tables[ti][1:] == expected_rows,
                    expected_rows=expected_rows, actual_rows=tables[ti][1:] if ti < len(tables) else [])
    raw_expected = [[r["region"], f"{r['rmse_T']:.9g}", f"{r['relative_L2_pct']:.8g}", str(r["raw_negative_count"])] for r in raw["rows"]]
    audit.check("word_raw_fit_table_exact_values", len(tables) > 2 and tables[2][1:] == raw_expected, expected=raw_expected)
    # These distinctive magnitudes catch percent/fraction or training/test swaps.
    for phrase in ("0.044774598%", "0.000540592T", "3.158007363%", "0.035582226T", "3.315663347%", "复用测试", "训练内", "不是新盲测", "104", "Bz"):
        audit.check("word_scope_and_scale:" + phrase, phrase in text)
    audit.check("word_planar_norm_cells_not_split", sum("|B|" in c for tab in tables for row in tab for c in row) >= 10)
    if has_appendix and len(tables) == 7:
        def rounded_value(label, printed, expected):
            parsed = Decimal(printed.rstrip("%"))
            atol = float(Decimal("0.50001") * Decimal(10) ** parsed.as_tuple().exponent)
            audit.close(label, float(parsed), expected, atol=atol, rtol=1e-13)
        primary_ids = ["prediction_rotor_magnitude", "prediction_stator_magnitude", "prediction_airgap_Br", "fitting_rotor", "fitting_stator", "fitting_airgap"]
        for table_row, mid in zip(tables[5][1:], primary_ids):
            model = summary_models[mid]
            fit_branch = model["branch"] == "fitting_full60"
            part = "training_fit" if fit_branch else "reused_test"
            m = model["metrics"][part]
            audit.check("word_appendix_scope:" + mid, table_row[0] == ("全60训练内拟合" if fit_branch else "复用测试9工况") and table_row[1] == model["region"] and table_row[2] == ("Br" if model["region"] == "airgap" else "二维范数"))
            for col, metric in ((3, "rmse_T"), (4, "mae_T"), (5, "max_abs_T"), (6, "relative_L2_pct")):
                rounded_value("word_appendix_metric:" + mid + ":" + metric, table_row[col], m[metric])
        comparison = load_json(z, top + "/" + V2_NAME + "/figures/v1_v2_same_quantity_comparison.json")
        for table_row, original in zip(tables[6][1:], comparison["rows"]):
            title = original["region"] + " " + ("二维范数" if original["quantity"] == "|B|" else original["quantity"])
            audit.check("word_comparison_same_quantity:" + original["model_id"], table_row[0] == title)
            for col, metric in ((1, "v1_relative_L2_pct"), (2, "v2_relative_L2_pct"), (3, "v1_max_abs_T"), (4, "v2_max_abs_T")):
                rounded_value("word_v1_v2_comparison:" + original["model_id"] + ":" + metric, table_row[col], original[metric])
            for metric in ("rmse_T", "mae_T", "relative_L2_pct", "max_abs_T"):
                audit.close("word_v2_comparison_source:" + original["model_id"] + ":" + metric, original["v2_" + metric], summary_models[original["model_id"]]["metrics"]["reused_test"][metric])
    report["word"] = dict(document="两版本说明.docx", tables=[dict(rows=len(t), widths=[len(r) for r in t]) for t in tables],
                           source_metrics="new60_opt_v2/final_summary.json and website/results.json", scope="All60 fitting and reused-test prediction separately verified; raw and deployed fit remain distinct.")


RUNNER = r'''import builtins,json,os,runpy,sys,traceback
from pathlib import Path
import numpy as np
root=Path(sys.argv[1]).resolve();cli=Path(sys.argv[2]).resolve();tracepath=Path(sys.argv[3]);argv=sys.argv[4:]
events=[];loads={};oldload=np.load;oldget=np.lib.npyio.NpzFile.__getitem__;oldimport=builtins.__import__
allowed={'conditions','case_ids','point_id','xy','theta','time_s','solver_index','theta_m_relative_deg'}
forbidden={'field','coeff','truth','target','targets','test_truth','test_labels','labels','Bmag','BrBt'}
def guarded_load(file,*a,**kw):
    p=Path(file).resolve()
    if not p.is_relative_to(root):raise RuntimeError('NPZ load escaped relocated tree: '+str(p))
    obj=oldload(file,*a,**kw);loads[id(obj)]=str(p);events.append({'action':'load','path':str(p)});return obj
def guarded_get(obj,key):
    path=loads.get(id(obj),'UNKNOWN')
    if path=='UNKNOWN':raise RuntimeError('Untracked NPZ object')
    events.append({'action':'key','path':path,'key':str(key)})
    if str(key) in forbidden:raise RuntimeError('Field/target label or training coefficient access forbidden: '+str(key))
    if '/cache_v1/' in Path(path).as_posix() and key not in allowed:raise RuntimeError('Cache key outside input whitelist: '+str(key))
    return oldget(obj,key)
def guarded_import(name,*a,**kw):
    if name.split('.')[0] in {'torch','tensorflow','jax','sklearn'}:raise RuntimeError('Non-NumPy inference dependency: '+name)
    return oldimport(name,*a,**kw)
np.load=guarded_load;np.lib.npyio.NpzFile.__getitem__=guarded_get;builtins.__import__=guarded_import
ok=False;error=None
try:
    sys.path.insert(0,str(cli.parent));sys.argv=[str(cli)]+argv;runpy.run_path(str(cli),run_name='__main__');ok=True
except BaseException:
    error=traceback.format_exc();raise
finally:
    tracepath.write_text(json.dumps({'passed':ok,'error':error,'events':events,'field_labels_materialized':False if ok else None},ensure_ascii=False,indent=2),encoding='utf8')
'''


def minimal_extract(z, top, choices, digests, destination, audit):
    if destination.exists():
        raise FileExistsError("Refusing to overwrite portable test directory: " + str(destination))
    v2prefix = V2_NAME + "/"
    selected = {"fitting_rotor", "prediction_rotor_magnitude", "prediction_rotor_vector", "prediction_rotor_magnitude_nn", "prediction_airgap_Br", "fitting_airgap"}
    files = {v2prefix + s for s in ("code/predict_v2.py", "code/inference_numpy_v2.py", "both_versions_choices_frozen.json", "both_versions_choices_frozen.sha256", "prediction_choices_frozen.json", "prediction_choices_frozen.sha256", "protocol.json")}
    files |= {V1_NAME + "/cache_v1/" + s for s in ("rotor.npz", "stator.npz", "airgap.npz", "split.json")}
    for row in choices["models"]:
        for key in ("inference_module_relative_path", "model_source_module_relative_path"):
            if key in row:
                files.add(v2prefix + row[key])
        if row["id"] not in selected:
            continue
        rel = v2prefix + row["model_relative_path"]
        files.add(rel)
        sidecar = str(PurePosixPath(rel).with_suffix(".sha256"))
        if sidecar in digests:
            files.add(sidecar)
        # Only metadata references, never cache fields or coefficient labels.
        with z.open(top + "/" + rel) as f, np.load(io.BytesIO(f.read()), allow_pickle=False) as model:
            if "representation_relative_path" in model.files:
                reference = str(model["representation_relative_path"].item())
                unresolved = PurePosixPath(rel).parent / reference.replace("\\", "/")
                normalized = []
                for segment in unresolved.parts:
                    if segment == "..":
                        if not normalized:
                            raise ValueError("Representation outside archive")
                        normalized.pop()
                    elif segment != ".":
                        normalized.append(segment)
                dep = str(safe_relative("/".join(normalized)))
                if dep not in digests:
                    raise ValueError("Missing external representation " + dep)
                audit.check("portable_representation_identity:" + row["id"], digests[dep]["sha256"] == str(model["representation_sha256"].item()))
                files.add(dep)
    total = sum(digests[f]["bytes"] for f in files)
    free = shutil.disk_usage(destination.parent).free
    if total > 1024 ** 3 or free < total + 512 * 1024 ** 2:
        raise RuntimeError(f"Minimal extraction size/free-space safety condition failed: {total=} {free=}")
    destination.mkdir(parents=True)
    for rel in sorted(files):
        path = destination.joinpath(*PurePosixPath(rel).parts)
        if not path.resolve().is_relative_to(destination.resolve()):
            raise ValueError("Extraction escapes destination")
        path.parent.mkdir(parents=True, exist_ok=True)
        h, count = hashlib.sha256(), 0
        with z.open(top + "/" + rel) as src, path.open("xb") as dst:
            for block in iter(lambda: src.read(CHUNK), b""):
                h.update(block)
                count += len(block)
                dst.write(block)
        audit.check("portable_extract:" + rel, h.hexdigest() == digests[rel]["sha256"] and count == digests[rel]["bytes"])
    return dict(directory=str(destination), extracted_files=len(files), extracted_bytes=total, available_bytes_before=free, paths=sorted(files), full_study_extracted=False)


def saved_expected(z, top, row):
    part = "training_fit" if row["branch"] == "fitting_full60" else "reused_test"
    rel = V2_NAME + "/evaluation/" + row["id"] + "/" + part + "_predictions.npz"
    with np.load(io.BytesIO(z.read(top + "/" + rel)), allow_pickle=False) as f:
        case_index = np.flatnonzero(f["case_ids"] == 2)
        if len(case_index) != 1:
            raise ValueError("Case 02 missing/duplicated in saved evaluation: " + rel)
        # Truth exists in scientific NPZ, but is intentionally never materialized.
        out = {k: f[k] for k in ("point_id", "xy", "theta", "time_s", "solver_index")}
        out["prediction"] = f["prediction"][case_index[0]]
    return out, rel


def portable_cli(z, top, choices, destination, python, audit, report):
    rows = {r["id"]: r for r in choices["models"]}
    scenarios = [("fit_rotor_npz", "fitting_rotor", 44, "npz"),
                 ("prediction_rotor_norm_npz", "prediction_rotor_magnitude", 0, "npz"),
                 ("prediction_rotor_norm_csv", "prediction_rotor_magnitude", 59, "csv"),
                 ("pure_nn_rotor_vector_csv", "prediction_rotor_vector", 17, "csv"),
                 ("pure_nn_rotor_norm_npz", "prediction_rotor_magnitude_nn", 17, "npz"),
                 ("prediction_gap_all_frames_npz", "prediction_airgap_Br", None, "npz"),
                 ("fitting_gap_csv", "fitting_airgap", 59, "csv")]
    outdir = destination / "portable_outputs"
    outdir.mkdir()
    cli = destination / V2_NAME / "code/predict_v2.py"
    study = destination / V2_NAME
    env = dict(os.environ, PYTHONIOENCODING="utf8", PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1", PYTHONPATH="",
               OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2", NUMEXPR_NUM_THREADS="2")
    trials = []
    for name, model_id, frame, ext in scenarios:
        row = rows[model_id]
        output = outdir / (name + "." + ext)
        tracepath = outdir / (name + "_access_trace.json")
        args = ["--study", str(study), "--branch", row["branch"], "--region", row["region"], "--model-id", model_id,
                "--quantity", row["quantity"], "--case-id", "2", "--coordinates", "world", "--output", str(output)]
        if frame is not None:
            args += ["--frame", str(frame)]
        command = [str(python), "-c", RUNNER, str(destination), str(cli), str(tracepath), *args]
        start = time.time()
        proc = subprocess.run(command, cwd=destination, env=env, text=True, encoding="utf8", capture_output=True, timeout=180)
        audit.check("relocated_cli_exit:" + name, proc.returncode == 0, returncode=proc.returncode, stderr=proc.stderr[-4000:])
        if proc.returncode:
            raise RuntimeError("Relocated CLI failed: " + name + "\n" + proc.stderr)
        trace = json.loads(tracepath.read_text(encoding="utf8"))
        audit.check("runtime_field_access_guard:" + name, trace["passed"] is True and trace["field_labels_materialized"] is False,
                    data_load_count=sum(e["action"] == "load" for e in trace["events"]), field_key_access_count=sum(e["action"] == "key" for e in trace["events"]))
        cacheevents = [e for e in trace["events"] if e["action"] == "key" and "/cache_v1/" in Path(e["path"]).as_posix()]
        audit.check("cache_input_keys_exact:" + name, set(e["key"] for e in cacheevents) == set(INPUT_KEYS))
        meta = json.loads(output.with_suffix(output.suffix + ".json").read_text(encoding="utf8"))
        audit.check("output_identity:" + name, meta["model_id"] == model_id and meta["model_sha256"] == row["model_sha256"] and meta["choices_sha256"] == CHOICES_SHA and meta["field_labels_read"] is False and meta["original_case_id"] == 2 and meta["quantity"] == row["quantity"])
        scope = "training_fit, not heldout accuracy" if row["branch"] == "fitting_full60" else "reused_test, not fresh blind test"
        audit.check("output_scope:" + name, meta["score_scope"] == scope and meta["units"] == "T" and meta["xy_units"] == "mm")
        expected, source = saved_expected(z, top, row)
        indices = np.arange(60) if frame is None else np.array([frame])
        want = expected["prediction"][indices]
        reference = expected["xy"]
        angles = expected["theta"][indices]
        xy = np.broadcast_to(reference, (len(indices), *reference.shape)).copy()
        if row["region"] == "rotor":
            angle = np.deg2rad(angles)
            rotation = np.stack((np.cos(angle), -np.sin(angle), np.sin(angle), np.cos(angle)), -1).reshape(-1, 2, 2)
            xy = np.einsum("tij,pj->tpi", rotation, reference)
        if ext == "npz":
            with np.load(output, allow_pickle=False) as saved:
                audit.close("prediction_parity:" + name, saved["prediction"], want)
                audit.close("world_coordinates:" + name, saved["xy"], xy)
                audit.close("reference_coordinates:" + name, saved["xy_reference"], reference)
                audit.close("point_ids:" + name, saved["point_id"], expected["point_id"], atol=0, rtol=0)
                for key in ("theta", "time_s", "solver_index"):
                    audit.close(key + ":" + name, saved[key], expected[key][indices], atol=0, rtol=0)
                audit.check("npz_embedded_metadata:" + name, json.loads(saved["metadata_json"].item()) == meta)
        else:
            with output.open(encoding="utf8", newline="") as f:
                reader = csv.reader(f)
                header = next(reader)
                values = np.asarray([[float(v) for v in record] for record in reader])
            fixed = ["frame_index", "solver_index", "time_s", "theta_mechanical_absolute_deg", "point_id", "x_mm", "y_mm"]
            physical = ["Br_T", "Bt_T", "B_planar_magnitude_T"] if row["quantity"] == "Br,Bt" else ["Br_T" if row["quantity"] == "Br" else "B_planar_magnitude_T"]
            audit.check("csv_columns:" + name, header == fixed + physical)
            pcount = len(expected["point_id"])
            audit.check("csv_rows:" + name, len(values) == len(indices) * pcount)
            values = values.reshape(len(indices), pcount, -1)
            audit.close("world_coordinates:" + name, values[..., 5:7], xy)
            audit.close("point_ids:" + name, values[..., 4], np.broadcast_to(expected["point_id"], (len(indices), pcount)), atol=0, rtol=0)
            for column, wanttime in ((0, indices), (1, expected["solver_index"][indices]), (2, expected["time_s"][indices]), (3, angles)):
                audit.close(f"csv_column_{column}:" + name, values[..., column], np.broadcast_to(wanttime[:, None], (len(indices), pcount)), atol=0, rtol=0)
            wantcolumns = want.reshape(len(indices), pcount, 2) if row["quantity"] == "Br,Bt" else want[..., None]
            if row["quantity"] == "Br,Bt":
                wantcolumns = np.concatenate((wantcolumns, np.linalg.norm(wantcolumns, axis=-1, keepdims=True)), axis=-1)
            audit.close("prediction_parity:" + name, values[..., 7:], wantcolumns)
        trials.append(dict(name=name, model_id=model_id, frame_indices=indices.tolist(), format=ext, elapsed_seconds=time.time() - start,
                           prediction_reference=source, output=str(output), access_trace=str(tracepath), output_sha256=sha(output)))
        print("Portable scenario passed:", name, flush=True)
    report["portable_scenarios"] = trials


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ZIP)
    parser.add_argument("--report-root", type=Path, default=HERE)
    parser.add_argument("--portable-parent", type=Path, default=HERE / "portable_delivery_check")
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    args = parser.parse_args()
    args.archive = args.archive.resolve()
    args.report_root.mkdir(parents=True, exist_ok=True)
    paths = [args.report_root / "delivery_zip_audit.json", args.report_root / "delivery_zip_audit.md"]
    if any(p.exists() for p in paths):
        raise FileExistsError("Audit report already exists; choose a new --report-root")
    if not args.archive.is_file():
        raise FileNotFoundError("Wait for the completed final ZIP: " + str(args.archive))
    audit = Audit()
    start = time.time()
    report = dict(schema_version=1, audit_code_sha256=sha(__file__), archive=str(args.archive),
                  interpretation="Training-fit is all60 in-sample. Prediction uses the old42/9/9 split and an already seen test; no fresh-blind claim.",
                  operations=dict(training=False, model_selection=False, publication=False, complete_extraction=False, CLI_field_labels_materialized=False))
    error = None
    try:
        report["archive_sha256"] = sha(args.archive)
        report["archive_bytes"] = args.archive.stat().st_size
        side = Path(str(args.archive) + ".sha256")
        expected_archive_sha = side.read_text(encoding="utf8").strip().split()[0]
        audit.check("archive_sha256_sidecar", report["archive_sha256"] == expected_archive_sha)
        with zipfile.ZipFile(args.archive, "r") as z:
            top, manifest, digests = stream_archive(z, args.archive, audit, report)
            choices = verify_frozen(z, top, digests, audit)
            document_check(z, top, audit, report)
            if not all(c["passed"] for c in audit.checks):
                raise RuntimeError("Integrity/document preflight failed; no extraction or CLI execution performed")
            args.portable_parent.mkdir(parents=True, exist_ok=True)
            destination = (args.portable_parent / top).resolve()
            report["minimal_extraction"] = minimal_extract(z, top, choices, digests, destination, audit)
            portable_cli(z, top, choices, destination, args.python.resolve(), audit, report)
    except Exception:
        error = traceback.format_exc()
        audit.check("audit_completed_without_exception", False, error=error)
    report.update(elapsed_seconds=time.time() - start, check_count=len(audit.checks), all_checks_pass=all(c["passed"] for c in audit.checks), checks=audit.checks, error=error)
    paths[0].write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
    members = report.get("archive_members", {})
    extract = report.get("minimal_extraction", {})
    lines = ["# 双版本交付 ZIP 独立验收", "", f"结论：{'PASS' if report['all_checks_pass'] else 'FAIL'}；检查 {len(audit.checks)} 项。", "",
             f"- ZIP：`{args.archive.name}`", f"- ZIP SHA256：`{report.get('archive_sha256', '')}`",
             f"- 逐成员 CRC、字节数、SHA256：{members.get('file_count', 0)} 个文件；原 V1 保留 {members.get('v1_preserved_files', 0)} 个文件。",
             "- Word 正文的拟合、验证、复用测试数值分别对照原精度科学报告，检查 |B| 表格列数和原始/非负输出区分。",
             f"- 最小解压：{extract.get('extracted_files', 0)} 个文件，{extract.get('extracted_bytes', 0):,} 字节；未完整解压交付包。",
             f"- 重定位 CLI：{len(report.get('portable_scenarios', []))} 个实际子进程场景，比较已冻结保存预测、坐标、单位和身份；访问监控禁止加载真值及训练系数。",
             "- 全60拟合仅为训练内误差；旧9工况测试为复用评价，不能称为全新盲测。没有训练、重新选型或发布。", ""]
    failures = [c for c in audit.checks if not c["passed"]]
    if failures:
        lines += ["失败项目：", ""] + ["- " + c["name"] for c in failures]
    paths[1].write_text("\n".join(lines) + "\n", encoding="utf8")
    print(json.dumps(dict(all_checks_pass=report["all_checks_pass"], checks=len(audit.checks), report=str(paths[0])), ensure_ascii=False), flush=True)
    if error:
        print(error, file=sys.stderr)
    return 0 if report["all_checks_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

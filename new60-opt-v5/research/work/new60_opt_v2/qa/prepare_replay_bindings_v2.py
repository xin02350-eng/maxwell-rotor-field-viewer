"""Add immutable evidence links without modifying any frozen model choice."""
from pathlib import Path
import json
import replay_selected_v2 as qa

ROOT = Path(__file__).resolve().parents[1]


def main():
    freeze = ROOT / "both_versions_choices_frozen.json"
    choices = qa.read_json(freeze)
    expected = "c26dcdb4213e5afe09d78cf301e08e34234be2c1bf78a7aa32eab5673c4174c0"
    if qa.sha256(freeze) != expected:
        raise ValueError("Unexpected global frozen choices")
    supplement = dict(choices_sha256=expected, note="Evidence-only supplement; no choices, inference kwargs, weights, or validation selection metrics changed.", models=[])
    for choice in choices["models"]:
        if choice["branch"] != "prediction":
            continue
        relative = Path(choice["model_relative_path"])
        row = dict(id=choice["id"], provenance_files=[], source_bindings=[], saved_predictions=[])
        source = ROOT / relative
        transform = None
        key = "prediction"
        if relative.parts[0] == "kernel_route":
            provenance = relative.parent / "provenance.json"
            row["source_bindings"].append(dict(source_relative_path="kernel_route/kernel_model.py",
                record=provenance.as_posix(), pointer="/source_sha256/kernel_model.py"))
            routes = qa.read_json(ROOT / "kernel_route/route_choices.json")["rows"]
            found = [r for r in routes if (Path("kernel_route")/r["model_relative_path"]).as_posix() == relative.as_posix()
                     and qa.QUANTITIES[r["quantity"]] == qa.QUANTITIES[choice["quantity"]]]
            if len(found) != 1:
                raise ValueError("Ambiguous kernel validation prediction")
            saved = Path("kernel_route") / found[0]["prediction_relative_path"]
        elif relative.parts[0] == "neural_route":
            family = relative.parent
            selection_path = family / "selection_frozen.json"
            selection = qa.read_json(ROOT / selection_path)
            saved = family / "runs" / selection["selected"] / "validation_predictions.npz"
            for evidence in (selection_path, family / "prepared/representation_identity.json"):
                if (ROOT / evidence).exists():
                    row["provenance_files"].append(dict(relative_path=evidence.as_posix(), sha256=qa.sha256(ROOT/evidence)))
            if qa.QUANTITIES[choice.get("source_quantity", choice["quantity"])] == "vector" and qa.QUANTITIES[choice["quantity"]] == "planar_magnitude":
                transform = "hypot(Br,Bt)"
            elif choice["predict_kwargs"].get("nonnegative") is False:
                key = "raw_prediction"
        elif relative.parts[0] == "neural_kernel_route":
            saved = relative.parents[1] / "best_validation_predictions.npz"
            summary_path = relative.parents[1] / "summary.json"
            summary = qa.read_json(ROOT / summary_path)
            if summary["best"]["model_sha256"] != choice["model_sha256"]:
                raise ValueError("Best validation prediction is for another model")
            row["provenance_files"].append(dict(relative_path=summary_path.as_posix(), sha256=qa.sha256(ROOT/summary_path)))
        else:
            raise ValueError("Unknown frozen family")
        item = dict(partition="validation", relative_path=saved.as_posix(), sha256=qa.sha256(ROOT/saved), prediction_key=key, case_ids_key="case_ids")
        if transform:
            item["saved_transform"] = transform
        row["saved_predictions"].append(item)
        supplement["models"].append(row)
    path = Path(__file__).with_name("prediction_replay_bindings.json")
    with path.open("x", encoding="utf-8") as stream:
        json.dump(supplement, stream, ensure_ascii=False, indent=2)
    path.with_suffix(".json.sha256").write_text(qa.sha256(path)+"\n", encoding="ascii")
    print(path)


if __name__ == "__main__":
    main()

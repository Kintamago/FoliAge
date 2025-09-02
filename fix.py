import json

INPUT_PATH  = "detections.jsonl"
OUTPUT_PATH = "detections_fixed.jsonl"

def normalize_class(r):
    cid = r.get("class_id", None)
    if cid == 0:
        r["class_id"] = 0
        r["class_name"] = "tree"
    elif cid == 2:
        r["class_id"] = 2
        r["class_name"] = "grass"
    else:
        r["class_id"] = 1
        r["class_name"] = "bush"
    return r

def repair_jsonl(in_path, out_path):
    fixed = 0
    total = 0
    with open(in_path, "r") as fin, open(out_path, "w") as fout:
        for ln in fin:
            ln = ln.strip()
            if not ln:
                continue
            try:
                r = json.loads(ln)
                total += 1
                r = normalize_class(r)
                fout.write(json.dumps(r) + "\n")
                fixed += 1
            except json.JSONDecodeError as e:
                print(f"Skipping bad line: {ln[:50]}... ({e})")
    print(f"Processed {total} records → wrote {fixed} fixed records to {out_path}")

if __name__ == "__main__":
    repair_jsonl(INPUT_PATH, OUTPUT_PATH)

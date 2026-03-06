import re, json

def decode_token_number(token_str):
    pattern = r"(-?)\[NUM\]\[INT\](\d{3})\[SEP\]\[DEC\](\d{3})\[ENDNUM\]"
    m = re.search(pattern, token_str)
    if not m:
        raise ValueError(f"Invalid: {token_str!r}")
    sign = -1 if m.group(1) == "-" else 1
    return round(sign * (int(m.group(2)) + int(m.group(3)) / 1000), 4)

def parse_predicted_deltas(llm_output):
    marker = "Next motion deltas:"
    start = llm_output.find(marker)
    if start != -1:
        llm_output = llm_output[start + len(marker):]
    frame_pat = re.compile(
        r"\((-?\[NUM\]\[INT\]\d{3}\[SEP\]\[DEC\]\d{3}\[ENDNUM\])"
        r",(-?\[NUM\]\[INT\]\d{3}\[SEP\]\[DEC\]\d{3}\[ENDNUM\])\)"
    )
    joint_pat = re.compile(r"([A-Z][A-Z_]*):")
    result = {}
    for seg in llm_output.split(" | "):
        jm = joint_pat.match(seg.strip())
        if not jm:
            continue
        joint = jm.group(1)
        frames = [(decode_token_number(x), decode_token_number(y)) for x, y in frame_pat.findall(seg)]
        if frames:
            result[joint] = frames
    return result

with open("BaseLine/finetune_dataset_delta_test.jsonl", encoding="utf-8") as f:
    sample = json.loads(f.readline())

gt = parse_predicted_deltas(sample["completion"])
print("Joints parsed:", list(gt.keys()))
print("Frames per joint:", {j: len(v) for j, v in gt.items()})
print("First frame of NOSE:", gt.get("NOSE", [None])[0])
print("Parser: OK")

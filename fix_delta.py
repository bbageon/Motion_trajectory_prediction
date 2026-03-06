"""One-time fix script for delta/01_test2_delta.py parser functions."""
import re

path = "delta/01_test2_delta.py"
with open(path, encoding="utf-8") as f:
    content = f.read()

# --- Fix decode_token_number ---
old_decode = (
    'def decode_token_number(token_str: str):\n'
    '    # 토큰 문자열에서 수치를 추출한다.\n'
    '    # 기대 포맷: [NUM]-?[DEC]iii[SEP][DEC]ddd[ENDNUM]\n'
    '    pattern = r"\\[NUM\\](-?)(?:\\[DEC\\])?(\\d{3})\\[SEP\\]\\[DEC\\](\\d{3})\\[ENDNUM\\]"\n'
    '    m = re.search(pattern, token_str)\n'
    '    if not m:\n'
    '        raise ValueError(f"Invalid token format: {token_str}")\n'
    '    sign = -1 if m.group(1) == "-" else 1\n'
    '    int_part = int(m.group(2))\n'
    '    dec_part = int(m.group(3))\n'
    '    v = round(sign * (int_part + dec_part / 1000), 4)\n'
    '    return v'
)
new_decode = (
    'def decode_token_number(token_str: str):\n'
    '    # 학습 포맷: {sign}[NUM][INT]{int}[SEP][DEC]{dec}[ENDNUM]\n'
    '    # 부호가 [NUM] 앞에 위치, 정수부는 [INT] 토큰 사용\n'
    r'    pattern = r"(-?)\[NUM\]\[INT\](\d{3})\[SEP\]\[DEC\](\d{3})\[ENDNUM\]"' + '\n'
    '    m = re.search(pattern, token_str)\n'
    '    if not m:\n'
    '        raise ValueError(f"Invalid token format: {token_str}")\n'
    '    sign = -1 if m.group(1) == "-" else 1\n'
    '    int_part = int(m.group(2))\n'
    '    dec_part = int(m.group(3))\n'
    '    return round(sign * (int_part + dec_part / 1000), 4)'
)

# --- Fix parse_predicted_deltas ---
old_parse = (
    'def parse_predicted_deltas(llm_output: str):\n'
    '    # LLM 출력 텍스트에서\n'
    '    # JOINT:(x_token,y_token) 패턴을 모두 수집한다.\n'
    '    pattern = r"([A-Z_]+):\\(\\s*(\\[NUM\\].*?\\[ENDNUM\\])\\s*,\\s*(\\[NUM\\].*?\\[ENDNUM\\])\\s*\\)"\n'
    '    matches = re.findall(pattern, llm_output)\n'
    '\n'
    '    result = {}\n'
    '    for joint, x_tok, y_tok in matches:\n'
    '        dx = decode_token_number(x_tok)\n'
    '        dy = decode_token_number(y_tok)\n'
    '\n'
    '        if joint not in result:\n'
    '            result[joint] = []\n'
    '        result[joint].append((dx, dy))\n'
    '    return result'
)
new_parse = (
    'def parse_predicted_deltas(llm_output: str):\n'
    '    # 학습 완료 포맷: "Next motion deltas: JOINT:(tok,tok),(tok,tok),... | JOINT:..."\n'
    '    # "Next motion deltas:" 이후 텍스트만 사용\n'
    '    marker = "Next motion deltas:"\n'
    '    start = llm_output.find(marker)\n'
    '    if start != -1:\n'
    '        llm_output = llm_output[start + len(marker):]\n'
    '\n'
    r'    frame_pat = re.compile(' + '\n'
    r'        r"\((-?\[NUM\]\[INT\]\d{3}\[SEP\]\[DEC\]\d{3}\[ENDNUM\])"' + '\n'
    r'        r",(-?\[NUM\]\[INT\]\d{3}\[SEP\]\[DEC\]\d{3}\[ENDNUM\])\)"' + '\n'
    '    )\n'
    r'    joint_pat = re.compile(r"([A-Z][A-Z_]*):")' + '\n'
    '\n'
    '    result = {}\n'
    '    for seg in llm_output.split(" | "):\n'
    '        jm = joint_pat.match(seg.strip())\n'
    '        if not jm:\n'
    '            continue\n'
    '        joint = jm.group(1)\n'
    '        frames = []\n'
    '        for x_tok, y_tok in frame_pat.findall(seg):\n'
    '            dx = decode_token_number(x_tok)\n'
    '            dy = decode_token_number(y_tok)\n'
    '            frames.append((dx, dy))\n'
    '        if frames:\n'
    '            result[joint] = frames\n'
    '    return result'
)

found_decode = old_decode in content
found_parse  = old_parse  in content
print(f"found decode: {found_decode}")
print(f"found parse:  {found_parse}")

if found_decode:
    content = content.replace(old_decode, new_decode)
if found_parse:
    content = content.replace(old_parse, new_parse)

if found_decode or found_parse:
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print("Saved.")
else:
    # fallback: print surrounding context for debugging
    idx = content.find("def decode_token_number")
    print("Context around decode_token_number:")
    print(repr(content[idx:idx+300]))

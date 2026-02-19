import glob
import json
import os

# 입력 JSON 파일 폴더
INPUT_DIR = "./Custom_Data"
# 변환 결과(JSONL) 저장 경로
OUTPUT_PATH = "./finetune_dataset_delta.jsonl"


def num_to_tokens(val: float) -> str:
    """
    Convert a float to the structured numeric token format.
    Example: -0.012 -> -[NUM][INT]000[SEP][DEC]012[ENDNUM]
    """
    # 삼항 연산자 문법: 조건이 참이면 "-" 아니면 ""
    sign = "-" if val < 0 else ""
    # abs(): 절대값
    val = abs(val)
    # int(): 정수부 추출
    int_part = int(val)
    # round(x, 3): 소수점 셋째 자리까지 반올림
    dec_part = round((val - int_part) * 1000)

    # f-string 문법으로 문자열 내부에 변수/표현식 삽입
    # zfill(3): 문자열 길이를 3자리로 0-padding
    return (
        f"{sign}[NUM][INT]{str(int_part).zfill(3)}"
        f"[SEP][DEC]{str(dec_part).zfill(3)}[ENDNUM]"
    )


def format_pose_sequence_deltas(seq, max_frames: int = 10) -> str:
    """
    Encode pose sequence as (dx, dy) deltas using numeric structure tokens.
    """
    # 예외 케이스 처리: 프레임이 2개 미만이면 delta 계산 불가
    if len(seq) < 2:
        return ""

    # 리스트 누적 패턴
    formatted = []
    # dict.keys(): 첫 프레임에 있는 관절 이름 목록
    joints_list = seq[0].keys()

    # for-in 문법: 각 관절(joint)별로 반복
    for joint in joints_list:
        deltas = []
        # range(1, len(seq)): 이전 프레임(i-1)과 현재 프레임(i) 비교
        for i in range(1, len(seq)):
            # in 연산자로 key 존재 여부 확인
            if joint in seq[i] and joint in seq[i - 1]:
                # 튜플 언패킹 문법
                x1, y1 = seq[i - 1][joint]
                x2, y2 = seq[i][joint]
                dx = round(x2 - x1, 3)
                dy = round(y2 - y1, 3)
                dx_tok = num_to_tokens(dx)
                dy_tok = num_to_tokens(dy)
                deltas.append(f"({dx_tok},{dy_tok})")

        if deltas:
            # 슬라이싱 문법 [:max_frames]으로 최대 프레임 제한
            # ",".join(...)으로 리스트를 CSV 형태 문자열로 결합
            formatted.append(f"{joint}:{','.join(deltas[:max_frames])}")

    # " | ".join(...)으로 관절별 결과를 하나의 문자열로 결합
    return " | ".join(formatted)


def convert_all_to_delta() -> None:
    """
    Read [INPUT_DIR]/*.json and build JSONL finetuning samples.
    """
    # with 문법: 파일을 열고 블록 종료 시 자동 close
    # "w" 모드: 파일 새로 쓰기(기존 내용 덮어씀)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as fout:
        # glob + os.path.join: INPUT_DIR의 *.json 파일 순회
        for path in glob.glob(os.path.join(INPUT_DIR, "*.json")):
            with open(path, encoding="utf-8") as f:
                # json.load: 파일(JSON) -> 파이썬 dict
                data = json.load(f)
                # dict.get(key, default): 키 없으면 기본값([]) 반환
                seq = data.get("pose_sequence", [])
                if len(seq) < 3:
                    # continue: 현재 루프 건너뛰고 다음 파일로 진행
                    continue

                # // 는 정수 나눗셈(몫)
                mid = len(seq) // 2
                # 슬라이싱 문법: 앞 절반/뒤 절반 분리
                obs = seq[:mid]
                pred = seq[mid:]

                obs_deltas = format_pose_sequence_deltas(obs)
                pred_deltas = format_pose_sequence_deltas(pred)

                prompt = f"Observed motion deltas: {obs_deltas}"
                completion = f"Next motion deltas: {pred_deltas}"

                fout.write(
                    # json.dumps: dict -> JSON 문자열
                    # ensure_ascii=False: 한글 등 유니코드 원문 유지
                    json.dumps(
                        {
                            "prompt": prompt,
                            "completion": completion,
                            "task": "trajectory_delta",
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    # print로 저장 완료 로그 출력
    print(f"Saved finetuning data to {OUTPUT_PATH}")


# 파이썬 엔트리포인트 문법:
# 이 파일을 직접 실행했을 때만 main 로직을 수행
if __name__ == "__main__":
    convert_all_to_delta()

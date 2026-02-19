import json
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.lines import Line2D

# Delta 결과와 Absolute 결과를 같은 좌표계에 겹쳐 그려
# 프레임별 골격/궤적 차이를 비교하는 시각화 스크립트.

def visualize_comparison_step_arrows(delta_path, abs_path, title="Trajectory Comparison: Delta vs Absolute"):
    """
    두 JSON(pose_sequence)을 읽어 비교 플롯을 저장/표시한다.
    - delta_path: delta 방식 결과 JSON 경로
    - abs_path: absolute 방식 결과 JSON 경로
    """
    # 1. 파일 로드
    if not os.path.exists(delta_path) or not os.path.exists(abs_path):
        print(f"❌ 파일을 찾을 수 없습니다.\nDelta: {delta_path}\nAbs: {abs_path}")
        return

    # 저장 경로 설정 (Delta 파일 위치 기준)
    output_dir = os.path.dirname(delta_path)
    output_name = os.path.join(output_dir, "motion_comparison_arrows.png")

    with open(delta_path, "r", encoding="utf-8") as f:
        delta_data = json.load(f)
    with open(abs_path, "r", encoding="utf-8") as f:
        abs_data = json.load(f)
    
    # 2. 그래프 설정 (기존 구조 유지)
    plt.figure(figsize=(10, 10))
    ax = plt.gca()
    
    # 공통 설정
    JOINTS = ["RIGHT_SHOULDER", "RIGHT_ELBOW", "RIGHT_WRIST", "LEFT_SHOULDER", "LEFT_ELBOW", "LEFT_WRIST"]
    CONNECTIONS = [
        ("RIGHT_SHOULDER", "RIGHT_ELBOW"), ("RIGHT_ELBOW", "RIGHT_WRIST"),
        ("LEFT_SHOULDER", "LEFT_ELBOW"), ("LEFT_ELBOW", "LEFT_WRIST"),
        ("RIGHT_SHOULDER", "LEFT_SHOULDER")
    ]
    base_colors = plt.cm.tab10(np.linspace(0, 1, 10)) # 시간 흐름에 따른 뼈대 색상

    all_x, all_y = [], [] # 축 범위를 위한 좌표 모음
    legend_handles = []

    # =========================================================
    # 내부 함수: 단일 시퀀스 그리기 (Skeleton + Arrow)
    # =========================================================
    def draw_sequence(pose_seq, arrow_color, label_prefix, is_main=True):
        # 최대 10프레임만 렌더링하여 비교 가독성을 유지
        num_frames = min(10, len(pose_seq))
        traj_x, traj_y = [], []

        for f_idx in range(num_frames):
            frame = pose_seq[f_idx]
            
            # 뼈대 색상: 메인(Delta)은 진하게, 비교군(Absolute)은 연하게
            skel_alpha = 0.5 if is_main else 0.2
            color = base_colors[f_idx]

            # 손목 좌표 수집
            if "RIGHT_WRIST" in frame:
                wx, wy = frame["RIGHT_WRIST"]
                traj_x.append(wx)
                traj_y.append(wy)
            
            # 관절 점 (Scatter)
            for joint in JOINTS:
                if joint in frame:
                    x, y = frame[joint]
                    ax.scatter(x, y, color=color, s=70 if is_main else 40, edgecolors='white', 
                               linewidth=0.8, alpha=skel_alpha, zorder=5)
                    all_x.append(x); all_y.append(y)

            # 뼈대 선 (Plot)
            first_line = True
            for start_j, end_j in CONNECTIONS:
                if start_j in frame and end_j in frame:
                    x_pts = [frame[start_j][0], frame[end_j][0]]
                    y_pts = [frame[start_j][1], frame[end_j][1]]
                    line, = ax.plot(x_pts, y_pts, color=color, linewidth=1.8 if is_main else 1.0, 
                                    alpha=skel_alpha, zorder=4)
                    
                    # 범례용 핸들 (메인 모델의 프레임만 추가)
                    if first_line and is_main:
                        line.set_label(f"Frame {f_idx}")
                        legend_handles.append(line)
                        first_line = False

        # 📌 [핵심] 화살표 그리기 (기존 annotate 구조 유지)
        if len(traj_x) > 1:
            for i in range(len(traj_x) - 1):
                start_x, start_y = traj_x[i], traj_y[i]
                end_x, end_y = traj_x[i+1], traj_y[i+1]
                
                ax.annotate('', 
                            xy=(end_x, end_y),          
                            xytext=(start_x, start_y),  
                            arrowprops=dict(arrowstyle='-|>', 
                                            color=arrow_color, # 파라미터로 받은 색상
                                            lw=2.3, 
                                            ls='-',    
                                            mutation_scale=20, 
                                            alpha=0.9), # 화살표는 잘 보이게
                            zorder=10)
    
    # =========================================================
    # 3. 두 모델 그리기 실행
    # =========================================================
    
    # (1) Absolute Model 그리기 (빨간색, 배경)
    abs_seq = abs_data.get("pose_sequence", [])
    draw_sequence(abs_seq, arrow_color='crimson', label_prefix="Abs", is_main=False)

    # (2) Delta Model 그리기 (파란색, 전경 - 메인)
    delta_seq = delta_data.get("pose_sequence", [])
    draw_sequence(delta_seq, arrow_color='blue', label_prefix="Delta", is_main=True)


    # 4. 줌 및 스타일 설정 (기존 코드 동일)
    if all_x and all_y:
        # 데이터 범위에 맞춰 축 자동 조정
        x_min, x_max = min(all_x), max(all_x)
        y_min, y_max = min(all_y), max(all_y)
        x_range = x_max - x_min if x_max != x_min else 0.1
        y_range = y_max - y_min if y_max != y_min else 0.1
        margin_rate = 0.05 
        ax.set_xlim(x_min - x_range * margin_rate, x_max + x_range * margin_rate)
        
        # 이미지 좌표계(MediaPipe) 특성상 Y축 반전이 필요하면 아래 주석 해제 (기존 코드엔 없어서 유지함)
        ax.set_ylim(y_max + y_range * margin_rate, y_min - y_range * margin_rate)

    # 눈금 설정
    ax.xaxis.set_major_locator(plt.MultipleLocator(0.1))
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.05))
    ax.grid(True, linestyle='-', color='gray', alpha=0.3, zorder=1)

    ax.set_xlabel("X Coordinate", fontsize=14, fontweight='bold')
    ax.set_ylabel("Y Coordinate", fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=18, fontweight='bold', pad=25)

    # 범례 설정 (Delta, Absolute 화살표 구분 추가)
    custom_legend = legend_handles + [
        Line2D([0], [0], color='blue', lw=2.5, linestyle='-', label='Delta'),
        Line2D([0], [0], color='crimson', lw=2.5, linestyle='-', label='Absolute')
    ]
    
    ax.legend(handles=custom_legend, loc='upper left', bbox_to_anchor=(1.02, 1), 
              title="Comparison", fontsize=11, title_fontsize=12, frameon=True)

    plt.tight_layout()
    plt.savefig(output_name, dpi=300, bbox_inches='tight')
    print(f"✅ 비교 분석 시각화 완료: {output_name}")
    plt.show()

if __name__ == "__main__":
    # 경로 설정
    delta_path = r"C:\Users\DSU\Desktop\Finetuning\For_Paper_Code\Delta_result\predicted_motion_result.json"
    abs_path = r"C:\Users\DSU\Desktop\Finetuning\For_Paper_Code\Absolute_result\predicted_motion_absolute_result.json"
    
    visualize_comparison_step_arrows(delta_path, abs_path)

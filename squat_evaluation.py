import cv2
import csv
import mediapipe as mp
from datetime import datetime
import pandas as pd
import math

class SquatEvaluation:
    def __init__(self, side_data_path, front_data_path):
        self.df_side = pd.read_csv(side_data_path)
        self.df_front = pd.read_csv(front_data_path)

        # 가장 아래로 내려갔을 때의 프레임 값 추출
        self.lowest_frame_side = self.find_lowest_frame(self.df_side)
        self.lowest_frame_front = self.find_lowest_frame(self.df_front)

        # 가장 위로 올라갔을 때의 프레임 값 추출
        self.highest_frame_side = self.find_highest_frame(self.df_side)
        self.highest_frame_front = self.find_highest_frame(self.df_front)

        # 측면 영상이 왼쪽인지 오른쪽인지 판별
        self.side = self.left_or_right(self.df_side, self.lowest_frame_side)

        # 축척 계산
        self.scale_side, self.scale_front = self.scale(self.df_side, self.df_front, self.highest_frame_side, self.highest_frame_front)

    def find_lowest_frame(self, df): # 가장 아래로 내려갔을 때의 프레임 값 추출
        max_value = df[['Left Hip y', 'Right Hip y']].max().max()
        max_row_index = df[(df['Left Hip y'] == max_value) | (df['Right Hip y'] == max_value)].index[0]
        return df.loc[max_row_index, 'Frame']

    def find_highest_frame(self, df): # 가장 위로 올라갔을 때의 프레임 값 추출
        min_value = df[['Left Hip y', 'Right Hip y']].min().min()
        min_row_index = df[(df['Left Hip y'] == min_value) | (df['Right Hip y'] == min_value)].index[0]
        return df.loc[min_row_index, 'Frame']

    def left_or_right(self, df, lowest_frame_side): # 측면 영상이 왼쪽인지 오른쪽인지 판별
        if df.loc[lowest_frame_side, 'Left Hip x'] > df.loc[lowest_frame_side, 'Left Knee x']:
            return 'Left'
        else:
            return 'Right'

    def calculate_angle(self, df, lowest_frame_side, side): # 어깨-골반이 지면과 이루는 각도 계산
        df_lowest_shoulder = df[df['Frame'] == lowest_frame_side][[f'{side} Shoulder x', f'{side} Shoulder y', f'{side} Shoulder z']]
        df_lowest_hip = df[df['Frame'] == lowest_frame_side][[f'{side} Hip x', f'{side} Hip y', f'{side} Hip z']]

        x1, y1, z1 = df_lowest_shoulder.iloc[0]
        x2, y2, z2 = df_lowest_hip.iloc[0]

        vx = x2 - x1
        vy = y2 - y1
        vz = z2 - z1

        v_magnitude = math.sqrt(vx**2 + vy**2 + vz**2)
        dot_product = vy # xz 평면의 법선 벡터와 내적

        cos_theta = dot_product / v_magnitude
        theta_radian = math.acos(cos_theta)
        theta_degree = abs(90 - math.degrees(theta_radian)) # 90도에서 뺀 값을 반환
        return theta_degree

    def calculate_knee_foot(self, df, lowest_frame_side, side): # 무릎과 발의 위치 비교
        df_lowest_knee = df[df['Frame'] == lowest_frame_side][[f'{side} Knee x', f'{side} Knee y', f'{side} Knee z']]
        df_lowest_foot = df[df['Frame'] == lowest_frame_side][[f'{side} Foot Index x', f'{side} Foot Index y', f'{side} Foot Index z']]

        knee_x = df_lowest_knee.iloc[0][f'{side} Knee x']
        foot_x = df_lowest_foot.iloc[0][f'{side} Foot Index x']

        knee_foot_diff = (knee_x - foot_x) if side == 'Left' else (foot_x - knee_x) # 무릎과 발의 좌표 차이 계산
        knee_compare_foot = '뒤' if knee_foot_diff > 0 else '앞' # 무릎이 발보다 앞에 있는지 뒤에 있는지 판별  

        return knee_compare_foot, abs(knee_foot_diff) / self.scale_side # 무릎과 발의 좌표 차이를 축척으로 나누어 반환

    def calculate_hip_knee(self, df, lowest_frame_side, side): # 골반과 무릎의 높이 비교
        df_hip_y = df[df['Frame'] == lowest_frame_side][[f'{side} Hip y']]
        df_knee_y = df[df['Frame'] == lowest_frame_side][[f'{side} Knee y']]

        hip_y = df_hip_y.iloc[0][f'{side} Hip y']
        knee_y = df_knee_y.iloc[0][f'{side} Knee y']

        hip_knee_diff = (hip_y - knee_y) if hip_y > knee_y else (knee_y - hip_y) # 골반과 무릎의 좌표 차이 계산
        hip_compare_knee = '아래' if hip_y > knee_y else '위' # 골반이 무릎보다 아래에 있는지 위에 있는지 판별  

        return hip_compare_knee, hip_knee_diff / self.scale_side # 골반과 무릎의 좌표 차이를 축척으로 나누어 반환

    def compare_knee_foot_distance(self, df, lowest_frame_front): # 무릎사이의 거리와 발사이의 거리 비교
        df_lowest_knee = df[df['Frame'] == lowest_frame_front][['Left Knee x', 'Right Knee x']]
        df_lowest_foot = df[df['Frame'] == lowest_frame_front][['Left Ankle x', 'Right Ankle x']]

        knee_distance = abs(df_lowest_knee.iloc[0]['Left Knee x'] - df_lowest_knee.iloc[0]['Right Knee x'])
        foot_distance = abs(df_lowest_foot.iloc[0]['Left Ankle x'] - df_lowest_foot.iloc[0]['Right Ankle x'])

        knee_foot_distance = '벌어' if knee_distance > foot_distance else '좁혀'
        return knee_foot_distance, (knee_distance - foot_distance) / self.scale_front

    def scale(self, df_side, df_front, highest_frame_side, highest_frame_front):
        shoulder_side_mean = df_side[df_side['Frame'] == highest_frame_side][['Left Shoulder y', 'Right Shoulder y']].mean(axis=1).iloc[0]
        ankle_side_mean = df_side[df_side['Frame'] == highest_frame_side][['Left Ankle y', 'Right Ankle y']].mean(axis=1).iloc[0]

        shoulder_front_mean = df_front[df_front['Frame'] == highest_frame_front][['Left Shoulder y', 'Right Shoulder y']].mean(axis=1).iloc[0]
        ankle_front_mean = df_front[df_front['Frame'] == highest_frame_front][['Left Ankle y', 'Right Ankle y']].mean(axis=1).iloc[0]

        scale_side = abs(shoulder_side_mean - ankle_side_mean)
        scale_front = abs(shoulder_front_mean - ankle_front_mean)

        return scale_side, scale_front

    def data_analyze(self):
        print("=== 데이터 분석 결과 ===")

        # 가장 아래로 내려간 프레임
        print(f'측면 영상 가장 아래 프레임: {self.lowest_frame_side}')
        print(f'정면 영상 가장 아래 프레임: {self.lowest_frame_front}')

        # 가장 위로 올라간 프레임
        print(f'측면 영상 가장 위 프레임: {self.highest_frame_side}')
        print(f'정면 영상 가장 위 프레임: {self.highest_frame_front}')

        # 방향 판별
        print(f'측면 영상 방향: {self.side}')

        # 축척 계산
        print(f'측면 영상 축척: {self.scale_side:.2f}')
        print(f'정면 영상 축척: {self.scale_front:.2f}')

    def feedback(self):
        print("=== 자세 피드백 ===")
        feedback = []

        angle = self.calculate_angle(self.df_side, self.lowest_frame_side, self.side)
        if angle < 55:
            feedback.append('허리가 너무 굽혀졌습니다.')
        elif angle > 70:
            feedback.append('허리가 너무 펴졌습니다.')

        _, knee_foot_diff = self.calculate_knee_foot(self.df_side, self.lowest_frame_side, self.side)
        if knee_foot_diff < -0.03:
            feedback.append('무릎이 너무 앞으로 나와 있어요')

        hip_compare_knee, hip_knee_diff = self.calculate_hip_knee(self.df_side, self.lowest_frame_side, self.side)
        if hip_knee_diff < -0.05 and hip_compare_knee == '위':
            feedback.append('더 앉아 주세요')

        _, distance_diff = self.compare_knee_foot_distance(self.df_front, self.lowest_frame_front)
        if distance_diff < -0.01:
            feedback.append('무릎을 너무 모았어요')

        if not feedback:
            feedback.append('정확한 자세로 스쿼트를 진행하셨습니다.')

        for message in feedback:
            print(message)

    def analyze(self):
        print("=== 분석 결과 ===")

        # 각도 계산
        angle = self.calculate_angle(self.df_side, self.lowest_frame_side, self.side)
        print(f'허리가 지면과 이루는 각도 : {angle:.2f}도')

        # 무릎과 발의 위치 비교
        knee_compare_foot, knee_foot_diff = self.calculate_knee_foot(self.df_side, self.lowest_frame_side, self.side)
        print(f'무릎이 발보다 {abs(knee_foot_diff):.4f}만큼 {knee_compare_foot}에 있습니다.')

        # 골반과 무릎의 높이 비교
        hip_compare_knee, hip_knee_diff = self.calculate_hip_knee(self.df_side, self.lowest_frame_side, self.side)
        print(f'골반이 무릎보다 {abs(hip_knee_diff):.4f}만큼 {hip_compare_knee}에 있습니다.')

        # 무릎과 발의 거리 비교
        knee_foot_distance, distance_diff = self.compare_knee_foot_distance(self.df_front, self.lowest_frame_front)
        print(f'무릎이 발보다 {abs(distance_diff):.4f}만큼 {knee_foot_distance}져 있습니다.')

    def evaluate(self):
        print("=== 점수 계산 ===")
        decent_score = 0

        angle = self.calculate_angle(self.df_side, self.lowest_frame_side, self.side)
        decent_score += abs(round(angle - 55, 4)) if angle < 55 else abs(round(angle - 70, 4)) if angle > 70 else 0

        _, knee_foot_diff = self.calculate_knee_foot(self.df_side, self.lowest_frame_side, self.side)
        decent_score += abs(knee_foot_diff + 0.03) * 100 if knee_foot_diff < -0.03 else 0

        hip_compare_knee, hip_knee_diff = self.calculate_hip_knee(self.df_side, self.lowest_frame_side, self.side)
        decent_score += abs(hip_knee_diff + 0.05) * 100 if hip_knee_diff < -0.05 and hip_compare_knee == '위' else 0

        _, distance_diff = self.compare_knee_foot_distance(self.df_front, self.lowest_frame_front)
        decent_score += abs(distance_diff + 0.01) * 100 if distance_diff < -0.01 else 0

        return round((100 - decent_score), 2)

# Mediapipe 초기화
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# 웹캠 열기
cap = cv2.VideoCapture(0)

# 영상 저장 설정
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = 30
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

# Mediapipe Pose 모델 초기화
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    recording = False
    frame_count = 0
    side_data = []
    front_data = []
    recording_count = 0
    side_video_filename = None
    front_video_filename = None
    out = None
    start_time = None

    print("측면 영상 촬영을 위해 준비해주세요.")
    print("Enter를 누르면 측면 영상 촬영이 시작됩니다.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("웹캠에서 영상을 가져올 수 없습니다.")
            break

        # Mediapipe Pose 처리
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        # 랜드마크 시각화 및 데이터 수집
        if results.pose_landmarks and recording:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
            # 데이터 수집
            landmarks = results.pose_landmarks.landmark
            row = [frame_count]

            # 각 관절의 x, y, z 좌표 수집
            for idx in [11, 12, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]:
                landmark = landmarks[idx]
                if landmark.visibility < 0.8:  # visibility가 낮은 경우
                    row.extend([None, None, None])  # 해당 관절 데이터만 None으로 처리
                else:
                    row.extend([landmark.x, landmark.y, landmark.z])

            # 현재 프레임 데이터 저장
            if recording_count == 0:
                side_data.append(row)
            else:
                front_data.append(row)
            frame_count += 1

        # 녹화 중일 때 처리
        if recording:
            current_time = datetime.now()
            elapsed_time = (current_time - start_time).total_seconds()
            
            out.write(frame)

            # 5초 경과 시 녹화 종료 및 CSV 파일 저장
            if elapsed_time >= 5.0:
                print("5초 녹화 완료")
                recording = False
                recording_count += 1
                
                # 비디오 파일 저장 완료
                if out is not None:
                    out.release()
                    out = None

                # CSV 파일 저장
                try:
                    csv_filename = (side_video_filename if recording_count == 1 else front_video_filename).replace('.mp4', '_pose_data.csv')
                    current_data = side_data if recording_count == 1 else front_data
                    
                    with open(csv_filename, mode='w', newline='') as csv_file:
                        csv_writer = csv.writer(csv_file)
                        header = ['Frame']
                        for joint in ['Left Shoulder', 'Right Shoulder', 'Left Hip', 'Right Hip',
                                    'Left Knee', 'Right Knee', 'Left Ankle', 'Right Ankle',
                                    'Left Heel', 'Right Heel', 'Left Foot Index', 'Right Foot Index']:
                            header.extend([f'{joint} x', f'{joint} y', f'{joint} z'])
                        csv_writer.writerow(header)
                        
                        for row in sorted(current_data, key=lambda x: x[0]):
                            csv_writer.writerow(row)
                    
                    if recording_count == 1:
                        side_filename = csv_filename
                        print("\n정면 영상 촬영을 위해 준비해주세요.")
                        print("Enter를 누르면 정면 영상 촬영이 시작됩니다.")
                    else:
                        front_filename = csv_filename
                        print("\n촬영이 모두 완료되었습니다.")
                        print("'q'를 누르면 분석을 시작합니다.")
                    
                    print(f"CSV 파일 저장 완료: {csv_filename}")
                
                except Exception as e:
                    print(f"CSV 파일 저장 중 오류 발생: {str(e)}")

                frame_count = 0

        # 화면에 현재 상태 표시
        if recording:
            elapsed = (datetime.now() - start_time).total_seconds()
            status_text = f"Recording... {5-elapsed:.1f}s"
        else:
            if recording_count == 0:
                status_text = "Side view recording standby..."
            elif recording_count == 1:
                status_text = "Front view recording standby..."
            else:
                status_text = "Recording complete. Press 'q' to analyze"

        cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('Pose Estimation', frame)

        # 키보드 입력 처리
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            if recording_count == 2:
                break
        elif key == 13 and not recording:  # Enter 키
            if recording_count < 2:
                print("녹화를 시작합니다...")
                recording = True
                frame_count = 0
                start_time = datetime.now()
                formatted_time = start_time.strftime("%Y-%m-%d_%H-%M-%S")
                output_filename = f"{formatted_time}_{'side' if recording_count == 0 else 'front'}.mp4"
                
                # 영상 파일명 저장
                if recording_count == 0:
                    side_video_filename = output_filename
                else:
                    front_video_filename = output_filename
                
                out = cv2.VideoWriter(output_filename, fourcc, fps, (frame_width, frame_height))

# 리소스 해제
cap.release()
if out is not None:
    out.release()
cv2.destroyAllWindows()

# 스쿼트 분석 실행
if side_filename and front_filename:
    try:
        # 데이터 유효성 검사
        df_side = pd.read_csv(side_filename)
        df_front = pd.read_csv(front_filename)
        
        if len(df_side) < 10 or len(df_front) < 10:  # 최소 프레임 수 체크
            print("\n정확한 자세가 아닙니다. 스쿼트 동작을 수행해 주세요.")
        else:
            print("\n=== 스쿼트 자세 분석 시작 ===")
            squat_evaluation = SquatEvaluation(side_filename, front_filename)
            squat_evaluation.data_analyze()
            squat_evaluation.feedback()
            squat_evaluation.analyze()
            print(f'점수는 {squat_evaluation.evaluate()} 점입니다.')
    except Exception as e:
        print("\n정확한 자세가 아닙니다. 스쿼트 동작을 수행해 주세요.")
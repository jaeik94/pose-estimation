# 필요한 모듈 import
# mediapipe, moviepy, yt_dlp, opencv-python  인스톨 필요
import yt_dlp
from moviepy import VideoFileClip
import cv2
import mediapipe as mp
import csv
import os

# 영상 URL 이미 저장된 영상을 사용할 경우 이 단계 뛰어넘음
url = "https://www.youtube.com/watch?v=OKhJCJ6zi6Q" #원하는 영상 주소 입력

# 다운로드 옵션
ydl_opts = {
    #'format': 'best',  # 최고 화질 다운로드
    'outtmpl': 'squat_female.mp4',  # 저장 파일명 설정
}

# 다운로드 실행
with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    ydl.download([url])

print("영상 다운로드 완료")

# 영상 파일 불러오기
video = VideoFileClip("squat_female.mp4")

# 자를 부분 설정 (초 단위) 
start_time = 282.2 # 시작 시간 (초) 영상마다 원하는 부분을 바꿀것
end_time = 285   # 종료 시간 (초)

# 영상 자르기
clip = video.subclipped(start_time, end_time)

# 자른 영상 저장
clip.write_videofile("squat_female_front.mp4", codec="libx264") #앞에서 찍은 화면

# 자를 부분 설정 (초 단위) 
start_time = 160 # 시작 시간 (초) 영상마다 원하는 부분을 바꿀것
end_time = 164   # 종료 시간 (초)

# 영상 자르기
clip = video.subclipped(start_time, end_time)

# 자른 영상 저장
clip.write_videofile("squat_female_side.mp4", codec="libx264") #앞에서 찍은 화면

print("영상 클립 저장 완료")

video_path = [
    "squat_female_front.mp4",
    "squat_female_side.mp4"]  # 영상 경로 설정

def process_video(input_video_path):
    # Mediapipe Pose 모델 초기화
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils

    # CSV 파일 이름 자동 생성 (영상 이름에서 확장자를 제거하고 '_data.csv' 추가)
    video_name = os.path.splitext(os.path.basename(input_video_path))[0]
    output_csv_path = f"{video_name}_data.csv"

    # CSV 파일 열기
    with open(output_csv_path, mode='w', newline='', encoding='utf-8') as csv_file:
        csv_writer = csv.writer(csv_file)

        # CSV 파일의 헤더 작성
        csv_writer.writerow([
            'Frame',  # 프레임 번호
            'Left Shoulder x', 'Left Shoulder y', 'Left Shoulder z',
            'Right Shoulder x', 'Right Shoulder y', 'Right Shoulder z',
            'Left Hip x', 'Left Hip y', 'Left Hip z',
            'Right Hip x', 'Right Hip y', 'Right Hip z',
            'Left Knee x', 'Left Knee y', 'Left Knee z',
            'Right Knee x', 'Right Knee y', 'Right Knee z',
            'Left Ankle x', 'Left Ankle y', 'Left Ankle z',
            'Right Ankle x', 'Right Ankle y', 'Right Ankle z',
            'Left Heel x', 'Left Heel y', 'Left Heel z',
            'Right Heel x', 'Right Heel y', 'Right Heel z',
            'Left Toe x', 'Left Toe y', 'Left Toe z',
            'Right Toe x', 'Right Toe y', 'Right Toe z'
        ])

        # 비디오 파일 열기
        cap = cv2.VideoCapture(input_video_path)
        frame_count = 0  # 프레임 카운팅

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1  # 프레임 번호 증가

            # 색상 변환 (OpenCV는 BGR을 사용하므로, Mediapipe는 RGB를 사용)
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Pose 추정
            results = pose.process(image_rgb)

            if results.pose_landmarks:
                # 추적할 관절 목록
                selected_landmarks = [11, 12, 17, 18, 19, 20, 21, 22, 25, 26, 27, 28]
                landmarks = results.pose_landmarks.landmark
                row = [frame_count]  # 첫 번째 열에 프레임 번호 추가

                # 원하는 관절 번호에 해당하는 x, y, z 좌표 추출
                for i in selected_landmarks:
                    landmark = landmarks[i]
                    row.extend([landmark.x, landmark.y, landmark.z])  # x, y, z 좌표 추가

                # CSV 파일에 데이터 쓰기
                csv_writer.writerow(row)

            # 영상에 추정된 관절을 그려줌 (옵션)
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # 영상 표시 (옵션)
            cv2.imshow('Pose Estimation', frame)

            # 'q' 키를 눌러 종료
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # 리소스 해제
        cap.release()
        cv2.destroyAllWindows()

for input_video_path in video_path:
    process_video(input_video_path)

# print(f"Processed {input_video_path} and saved data to {output_csv_path}")
import pandas as pd
import math

class SquatAnalysis:
    def __init__(self, side_data_path, front_data_path):
        self.df_side = pd.read_csv(side_data_path)
        self.df_front = pd.read_csv(front_data_path)

        # 가장 아래로 내려갔을 때의 프레임 값 추출
        self.lowest_frame_side = self.find_lowest_frame(self.df_side)
        self.lowest_frame_front = self.find_lowest_frame(self.df_front)
        print('측면 영상에서 가장 앉은 프레임 : ', self.lowest_frame_side)
        print('정면 영상에서 가장 앉은 프레임 : ', self.lowest_frame_front)

        # 가장 위로 올라갔을 때의 프레임 값 추출
        self.highest_frame_side = self.find_highest_frame(self.df_side)
        self.highest_frame_front = self.find_highest_frame(self.df_front)
        print('측면 영상에서 가장 일어선 프레임 : ', self.highest_frame_side)
        print('정면 영상에서 가장 일어선 프레임 : ', self.highest_frame_front)

        # 측면 영상이 왼쪽인지 오른쪽인지 판별
        self.side = self.left_or_right(self.df_side, self.lowest_frame_side)
        print(f'측면 영상은 {self.side} 방향에서 측정했습니다.')

        #축척
        self.scale_side, self.scale_front = self.scale(self.df_side, self.df_front, self.highest_frame_side, self.highest_frame_front)
        print(f'측면 영상 축척 : {self.scale_side:.2f}, 정면 영상 축척 : {self.scale_front:.2f}')

    def find_lowest_frame(self, df):
        max_value = df[['Left Hip y', 'Right Hip y']].max().max()
        max_row_index = df[(df['Left Hip y'] == max_value) | (df['Right Hip y'] == max_value)].index[0]
        return df.loc[max_row_index, 'Frame']

    def find_highest_frame(self, df):
        min_value = df[['Left Hip y', 'Right Hip y']].min().min()
        min_row_index = df[(df['Left Hip y'] == min_value) | (df['Right Hip y'] == min_value)].index[0]
        return df.loc[min_row_index, 'Frame']

    def left_or_right(self, df, lowest_frame_side):
        if df.loc[lowest_frame_side, 'Left Hip x'] > df.loc[lowest_frame_side, 'Left Knee x']:
            return 'Left'
        else:
            return 'Right'

    def calculate_angle(self, df, lowest_frame_side, side):
        df_lowest_shoulder = df[df['Frame'] == lowest_frame_side][[f'{side} Shoulder x', f'{side} Shoulder y', f'{side} Shoulder z']]
        df_lowest_hip = df[df['Frame'] == lowest_frame_side][[f'{side} Hip x', f'{side} Hip y', f'{side} Hip z']]
        
        x1, y1, z1 = df_lowest_shoulder.iloc[0]
        x2, y2, z2 = df_lowest_hip.iloc[0]

        vx = x2 - x1
        vy = y2 - y1
        vz = z2 - z1

        v_magnitude = math.sqrt(vx**2 + vy**2 + vz**2)
        dot_product = vy  # 내적은 y축 방향 성분

        cos_theta = dot_product / v_magnitude
        theta_radian = math.acos(cos_theta)
        theta_degree = abs(90 - math.degrees(theta_radian))
        return theta_degree

    def calculate_knee_foot(self, df, lowest_frame_side, side):
        df_lowest_knee = df[df['Frame'] == lowest_frame_side][[f'{side} Knee x', f'{side} Knee y', f'{side} Knee z']]
        df_lowest_foot = df[df['Frame'] == lowest_frame_side][[f'{side} Foot Index x', f'{side} Foot Index y', f'{side} Foot Index z']]

        knee_x = df_lowest_knee.iloc[0][f'{side} Knee x']
        foot_x = df_lowest_foot.iloc[0][f'{side} Foot Index x']

        if side == 'Left':
            if knee_x > foot_x:
                knee_compare_foot = '뒤'
            else:
                knee_compare_foot = '앞'
        else:
            if knee_x < foot_x:
                knee_compare_foot = '뒤'
            else:
                knee_compare_foot = '앞'

        return knee_compare_foot, abs(knee_x - foot_x)/self.scale_side

    def calculate_hip_knee(self, df, lowest_frame_side, side):
        df_hip_y = df[df['Frame'] == lowest_frame_side][[f'{side} Hip y']]
        df_knee_y = df[df['Frame'] == lowest_frame_side][[f'{side} Knee y']]

        hip_y = df_hip_y.iloc[0][f'{side} Hip y']
        knee_y = df_knee_y.iloc[0][f'{side} Knee y'] 

        if hip_y > knee_y:
            hip_compare_knee = '아래'
        else:
            hip_compare_knee = '위'

        return hip_compare_knee, abs(hip_y - knee_y)/self.scale_side

    def compare_knee_foot_distance(self, df, lowest_frame_front):
        df_lowest_knee = df[df['Frame'] == lowest_frame_front][['Left Knee x', 'Right Knee x']]
        df_lowest_foot = df[df['Frame'] == lowest_frame_front][['Left Ankle x', 'Right Ankle x']]

        knee_distance = abs(df_lowest_knee.iloc[0]['Left Knee x'] - df_lowest_knee.iloc[0]['Right Knee x'])
        foot_distance = abs(df_lowest_foot.iloc[0]['Left Ankle x'] - df_lowest_foot.iloc[0]['Right Ankle x'])

        if knee_distance > foot_distance:
            knee_foot_distance = '벌어'
        else:
            knee_foot_distance = '좁혀'

        return knee_foot_distance, abs(knee_distance - foot_distance)/self.scale_front

    def scale(self, df_side, df_front, highest_frame_side, highest_frame_front):
        # 측면 영상 어깨와 발목 평균 계산
        shoulder_side_mean = df_side[df_side['Frame'] == highest_frame_side][['Left Shoulder y', 'Right Shoulder y']].mean(axis=1).iloc[0]
        ankle_side_mean = df_side[df_side['Frame'] == highest_frame_side][['Left Ankle y', 'Right Ankle y']].mean(axis=1).iloc[0]

        # 정면 영상 어깨와 발목 평균 계산
        shoulder_front_mean = df_front[df_front['Frame'] == highest_frame_front][['Left Shoulder y', 'Right Shoulder y']].mean(axis=1).iloc[0]
        ankle_front_mean = df_front[df_front['Frame'] == highest_frame_front][['Left Ankle y', 'Right Ankle y']].mean(axis=1).iloc[0]

        # 어깨-발목 y 좌표 차이 계산
        scale_side = abs(shoulder_side_mean - ankle_side_mean)
        scale_front = abs(shoulder_front_mean - ankle_front_mean)

        return scale_side, scale_front
    
    def analyze(self):
        
        # 각도 계산
        angle = self.calculate_angle(self.df_side, self.lowest_frame_side, self.side)
        print(f'허리가 지면과 이루는 각도 : {angle:.2f}도')

        # 무릎과 발의 위치 비교
        knee_compare_foot, knee_foot_diff = self.calculate_knee_foot(self.df_side, self.lowest_frame_side, self.side)
        print(f'무릎이 발보다 {knee_foot_diff:.4f}만큼 {knee_compare_foot}에 있습니다.')

        # 골반과 무릎의 높이 비교
        hip_compare_knee, hip_knee_diff = self.calculate_hip_knee(self.df_side, self.lowest_frame_side, self.side)
        print(f'골반이 무릎보다 {hip_knee_diff:.4f}만큼 {hip_compare_knee}에 있습니다.')

        # 무릎과 발의 거리 비교
        knee_foot_distance, distance_diff = self.compare_knee_foot_distance(self.df_front, self.lowest_frame_front)
        print(f'무릎이 발보다 {distance_diff:.4f}만큼 {knee_foot_distance}져 있습니다.')


# 사용 예시
squat_analysis = SquatAnalysis('squat_female_side_data.csv', 'squat_female_front_data.csv')
squat_analysis.analyze()

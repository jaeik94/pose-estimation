from webcam_squat_evaluate4 import SquatDataRecorder, SquatEvaluation
import pandas as pd

def main():
    try:
        # 데이터 녹화
        print("\n=== 스쿼트 자세 분석 프로그램 ===")
        print("1. 측면 영상 촬영 (Enter로 시작)")
        print("2. 정면 영상 촬영 (Enter로 시작)")
        print("3. 녹화 완료 후 'a'를 눌러 분석 시작")
        print("* 언제든 'q'를 누르면 프로그램 종료")
        
        recorder = SquatDataRecorder()
        side_filename, front_filename = recorder.record_data()

        if not side_filename or not front_filename:
            print("\n녹화가 완료되지 않았습니다. 프로그램을 종료합니다.")
            return

        # 데이터 유효성 검사
        try:
            df_side = pd.read_csv(side_filename)
            df_front = pd.read_csv(front_filename)
            
            if len(df_side) < 10 or len(df_front) < 10:  # 최소 프레임 수 체크
                print("\n정확한 자세가 아닙니다. 스쿼트 동작을 수행해 주세요.")
                return
                
            # 데이터에 None 값이 너무 많은지 확인
            side_nulls = df_side.isnull().sum().sum() / (df_side.shape[0] * df_side.shape[1])
            front_nulls = df_front.isnull().sum().sum() / (df_front.shape[0] * df_front.shape[1])
            
            if side_nulls > 0.5 or front_nulls > 0.5:  # 50% 이상이 None인 경우
                print("\n정확한 자세가 아닙니다. 스쿼트 동작을 수행해 주세요.")
                return

        except Exception as e:
            print("\n정확한 자세가 아닙니다. 스쿼트 동작을 수행해 주세요.")
            print(f"오류 상세: {str(e)}")
            return

        # SquatEvaluation 초기화
        print("\n녹화된 데이터를 분석중입니다...")
        squat_evaluation = SquatEvaluation(side_filename, front_filename)

        while True:
            print("\n=== 실행 가능한 메서드 목록 ===")
            print("1. 데이터 분석 (data_analyze)")
            print("2. 자세 피드백 (feedback)")
            print("3. 자세 분석 결과 (analyze)")
            print("4. 점수 확인 (evaluate)")
            print("5. 종료")

            # 사용자 입력 받기
            choice = input("\n실행할 작업 번호를 입력하세요: ")

            if choice == "1":
                print("\n[데이터 분석 결과]")
                squat_evaluation.data_analyze()
            elif choice == "2":
                print("\n[자세 피드백]")
                squat_evaluation.feedback()
            elif choice == "3":
                print("\n[자세 분석 결과]")
                squat_evaluation.analyze()
            elif choice == "4":
                print("\n[점수 확인]")
                print(f'점수는 {squat_evaluation.evaluate()} 점입니다.')
            elif choice == "5":
                print("\n프로그램을 종료합니다.")
                break
            else:
                print("\n유효하지 않은 입력입니다. 다시 시도하세요.")

    except Exception as e:
        print(f"\n프로그램 실행 중 오류가 발생했습니다: {str(e)}")
        print("프로그램을 종료합니다.")

if __name__ == "__main__":
    main()
# [Step 1] 필요한 도구(라이브러리) 가져오기
import os        # 경로 합치기용
import glob      # 파일 찾기용
import pandas as pd  # 데이터 처리 끝판왕

# [Step 2] 어디를 뒤질지 지도(경로) 그리기
# mnt/d/는 WSL에서 바라본 윈도우의 D드라이브입니다.
data_folder = '/mnt/d/data'
# **은 모든 하위 폴더, *.xlsx는 모든 엑셀 파일을 의미합니다.
search_pattern = os.path.join(data_folder, '**', '*.xlsx')

# [Step 3] 실제로 파일들 주소 체포해 오기
# recursive=True를 켜야 하위 폴더 끝까지 샅샅이 뒤집니다.
excel_files = glob.glob(search_pattern, recursive=True)
all_data = [] # 데이터프레임들을 담을 빈 바구니

print(f"총 {len(excel_files)}개의 엑셀 파일을 찾았습니다. 합치기를 시작합니다...")

# [Step 4] 파일 하나씩 열어서 내용물 바구니에 담기 (반복문)
for file in excel_files:
    try:
        # 엑셀을 읽어 파이썬 표(df)로 만듭니다.
        df = pd.read_excel(file)
        # 나중에 어떤 파일에서 온 데이터인지 알 수 있게 파일 경로도 기록해두면 좋습니다.
        df['source_file'] = file 
        all_data.append(df)
        print(f"성공: {os.path.basename(file)}")
    except Exception as e:
        print(f"실패: {os.path.basename(file)} - 에러: {e}")

# [Step 5] 바구니에 담긴 데이터들 하나로 합치기
if all_data:
    # ignore_index=True는 줄 번호를 0부터 새로 예쁘게 매겨줍니다.
    combined_df = pd.concat(all_data, ignore_index=True)

    # [Step 6] 최종 결과물을 기계용 파일(Parquet)로 저장하기
    # engine='pyarrow'는 가장 빠르고 안정적인 저장 방식입니다.
    output_path = '/mnt/d/chatbot_data.parquet'
    combined_df.to_parquet(output_path, engine='pyarrow', index=False)
    
    print("\n" + "="*50)
    print(f"🎉 작업 완료! 총 {len(combined_df)}줄의 데이터가 저장되었습니다.")
    print(f"파일 위치: {output_path}")
    print("="*50)
else:
    print("합칠 데이터가 없습니다. 폴더 경로를 확인해주세요.")
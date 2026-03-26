import pandas as pd

# 1. 엑셀 파일 경로 설정
file_path = 'data/inventory_apparel_5000.xlsx'

try:
    # 2. 엑셀 파일 읽기 (특정 시트가 있다면 sheet_name='시트명' 추가)
    # 엔진으로 openpyxl을 사용하여 xlsx를 직접 읽습니다.
    df = pd.read_excel(file_path, engine='openpyxl')
    
    # 컬럼명 앞뒤에 혹시 모를 공백 제거
    df.columns = [col.strip() for col in df.columns]

    # 3. 문장형 변환 함수 (엑셀 컬럼명에 맞춰 최적화)
    def create_excel_chunk(row):
        # 엑셀의 숫자 데이터가 가끔 소수점으로 읽힐 수 있어 정수형으로 변환 처리
        qty = int(row['재고수량']) if pd.notnull(row['재고수량']) else 0
        price = int(row['단가(원)']) if pd.notnull(row['단가(원)']) else 0
        
        sentence = (
            f"재고 ID {row['재고ID']}에 해당하는 품목은 '{row['품목명']}'입니다. "
            f"카테고리는 {row['카테고리']}이며, 현재 {row['창고']}에 보관되어 있습니다. "
            f"보유 수량은 {qty:,}개이며, 단가는 {price:,}원입니다. "
            f"공급처는 {row['공급업체']}이며, 현재 재고 상태는 '{row['상태']}'로 파악됩니다."
        )
        return sentence

    # 4. 청킹 적용
    df['chunk'] = df.apply(create_excel_chunk, axis=1)
    df[['chunk']].to_csv('data/processed_chunks.csv', index=False, encoding='utf-8-sig')

    # 5. 결과 확인
    print(f"총 {len(df)}개의 행을 성공적으로 읽어왔습니다.")
    print("\n--- 엑셀 데이터 청킹 예시 ---")
    print(df['chunk'].iloc[0])

except Exception as e:
    print(f"파일을 읽는 중 오류가 발생했습니다: {e}")
    print("팁: 컬럼명이 '재고ID', '품목명' 등과 일치하는지 확인해주세요.")
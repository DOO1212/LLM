import logging
import os
import traceback
import warnings

from app import create_app
from app.routes.chatbot import load_learning_stats
from extract_data import run_extraction

warnings.filterwarnings("ignore")
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

app = create_app()


if __name__ == "__main__":
    try:
        port = int(os.environ.get("PORT", "8000"))
        debug = os.environ.get("FLASK_DEBUG", "1").lower() in ("1", "true", "yes")
        print("🔄 서버 시작: 교정 로그에서 학습 데이터를 갱신합니다...")
        run_extraction()
        learning_stats = load_learning_stats()
        print(f"✅ 학습 준비 완료: {learning_stats['learned_examples']}개의 예시 데이터 로드됨")
        print("🔄 데이터 폴더 스캔 중...")
        from data_reader import rescan
        rescan()
        print("✅ 데이터 조회 준비 완료")
        print(f"🌐 서버 바인딩: http://0.0.0.0:{port}")
        app.run(host="0.0.0.0", debug=debug, port=port, use_reloader=False)
    except Exception:
        error_text = traceback.format_exc()
        with open("server_start_error.log", "w", encoding="utf-8") as f:
            f.write(error_text)
        print("❌ 서버 시작 실패. server_start_error.log 파일을 확인하세요.")
        raise
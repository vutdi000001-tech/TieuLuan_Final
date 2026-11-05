import os
import sys

# Chuyển thư mục làm việc về webapp
webapp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'webapp')
os.chdir(webapp_dir)
sys.path.insert(0, webapp_dir)

from app import app

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"==================================================================")
    print(f"  CỔNG THỰC NGHIỆM AI - TIỂU LUẬN FINAL (ĐẦY ĐỦ 6 CHƯƠNG)")
    print(f"  Đang khởi chạy tại: http://localhost:{port}")
    print(f"==================================================================")
    app.run(host='0.0.0.0', port=port, debug=True)


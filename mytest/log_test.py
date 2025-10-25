# log_writer.py
import os
import time
import argparse
from datetime import datetime

def get_log_filename():
    """根据当前时间生成日志文件名"""
    now = datetime.now()
    return now.strftime("%Y%m%d_%H.log")

def write_log(message, log_dir):
    """将日志写入当前小时的日志文件"""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, get_log_filename())
    with open(log_file, "a") as f:
        f.write(message + "\n")

def main(duration, log_dir):
    """主逻辑：每分钟写入日志，每小时切换文件"""
    start_time = time.time()
    end_time = start_time + duration
    last_hour = datetime.now().hour

    while time.time() < end_time:
        now = datetime.now()
        current_hour = now.hour
        message = f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] Writing log..."
        write_log(message, log_dir)
        print(message)  # 控制台输出（便于在Kaggle观察）

        # 检查小时是否变化，若变化则切换日志文件（自动生成新文件名）
        if current_hour != last_hour:
            last_hour = current_hour
            print(f"🕒 新建日志文件：{get_log_filename()}")

        # 每分钟执行一次（60秒）
        time.sleep(60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="定时写日志，每小时新建文件")
    parser.add_argument("--duration", type=int, required=True, help="运行时间（秒）")
    parser.add_argument("--log-dir", type=str, default="/kaggle/working/logs", help="日志目录（默认：/kaggle/working/logs）")
    args = parser.parse_args()

    print(f"启动日志记录，总运行时间：{args.duration} 秒，日志目录：{args.log_dir}")
    main(args.duration, args.log_dir)
    print("✅ 运行结束")

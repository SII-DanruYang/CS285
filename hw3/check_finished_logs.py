import os
import re
import glob
from pathlib import Path

logs = sorted(glob.glob("*.log") + glob.glob("run_logs_sac/*.log"))

def read_text(path):
    with open(path, "rb") as f:
        b = f.read()
    return b.decode("utf-8", errors="ignore").replace("\r", "\n")

def last_match(pattern, text):
    ms = list(re.finditer(pattern, text))
    return ms[-1] if ms else None

print("=" * 110)
print(f"{'STATUS':<14} {'LOG':<38} {'PROGRESS':<20} {'DATA_DIR / INFO'}")
print("=" * 110)

for log in logs:
    text = read_text(log)

    has_error = bool(re.search(r"Traceback|RuntimeError|ImportError|Error:|Killed|Exception", text))

    # 找 logging outputs to 的目录
    data_match = last_match(r"logging outputs to\s+(.+)", text)
    data_dir = data_match.group(1).strip() if data_match else ""

    # 找所有 tqdm 进度，例如 100%|...| 300000/300000
    progress_matches = list(re.finditer(r"(\d+)%\|.*?(\d+)\s*/\s*(\d+)", text))
    progress = ""
    done = False

    if progress_matches:
        m = progress_matches[-1]
        pct, cur, total = m.group(1), m.group(2), m.group(3)
        progress = f"{pct}% {cur}/{total}"
        done = (pct == "100" and cur == total)
    else:
        # 有些日志最后只保留了 100% 行但格式略不一样
        done = bool(re.search(r"100%\|.*?(\d+)/\1", text))
        progress = "found 100%" if done else "-"

    if has_error:
        status = "ERROR"
    elif done:
        status = "DONE_100%"
    elif progress != "-":
        status = "PARTIAL"
    else:
        status = "NO_PROGRESS"

    print(f"{status:<14} {log:<38} {progress:<20} {data_dir[:80]}")

print("=" * 110)
print("Currently running run_hw3 processes:")
os.system("ps -ef | grep -E 'run_hw3_(sac|dqn).py' | grep -v grep || true")

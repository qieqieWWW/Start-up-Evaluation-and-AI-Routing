#!/usr/bin/env python
# 临时调试脚本：直接运行 m16.py 并捕获完整输出
import subprocess
import sys
import os

os.chdir(r"c:\Users\wulin\Desktop\Start-up-Evaluation-and-AI-Routing\scripts")

result = subprocess.run(
    [sys.executable, "m16.py"],
    capture_output=True,
    text=True,
    encoding='utf-8'
)

print("=== STDOUT ===")
print(result.stdout)
print("\n=== STDERR ===")
print(result.stderr)
print(f"\n=== RETURN CODE: {result.returncode} ===")

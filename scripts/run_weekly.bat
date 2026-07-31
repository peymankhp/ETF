@echo off
REM ETF Intel - weekly scheduled run.
REM Uses the venv's Python by absolute path so it does NOT depend on PATH, uv,
REM or the Windows Store python aliases (which fail under Task Scheduler).
REM Output is appended to data\weekly_run.log for debugging.

cd /d "D:\Github\ETF"
"D:\Github\ETF\.venv\Scripts\python.exe" scripts\run_pipeline.py >> "D:\Github\ETF\data\weekly_run.log" 2>&1

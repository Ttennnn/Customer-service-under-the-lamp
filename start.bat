@echo off
setlocal
chcp 65001 >nul

set "BASH="
for %%P in (
  "%ProgramFiles%\Git\bin\bash.exe"
  "%ProgramFiles(x86)%\Git\bin\bash.exe"
  "%LocalAppData%\Programs\Git\bin\bash.exe"
) do (
  if not defined BASH if exist "%%~P" set "BASH=%%~P"
)

if not defined BASH (
  for /f "delims=" %%I in ('where bash 2^>nul') do (
    if not defined BASH (
      "%%I" -c "uname -s 2>/dev/null" 2>nul | findstr /i /c:"MINGW" /c:"MSYS" >nul && set "BASH=%%I"
    )
  )
)

if not defined BASH (
  echo [start] 未找到 Git Bash，请先安装 Git for Windows。
  pause
  exit /b 1
)

pushd "%~dp0" >nul
"%BASH%" start.sh %*
set "CODE=%ERRORLEVEL%"
popd >nul
if not "%CODE%"=="0" pause
exit /b %CODE%

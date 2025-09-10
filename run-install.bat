@echo off
setlocal

chcp 65001 > nul
cd /d "%~dp0"

:: --- Argument Parsing ---
set "DEVICE="
set "SOURCE="

if "%~1"=="" goto :show_usage
if /i "%~1"=="CU128" set DEVICE=CU128
if /i "%~1"=="CU126" set DEVICE=CU126
if /i "%~1"=="CPU"   set DEVICE=CPU
if not defined DEVICE (
    echo [ERROR] Invalid Device specified: %1
    goto :show_usage
)

if "%~2"=="" goto :show_usage
if /i "%~2"=="HF"         set SOURCE=HF
if /i "%~2"=="HF-Mirror"  set SOURCE=HF-Mirror
if /i "%~2"=="ModelScope" set SOURCE=ModelScope
if not defined SOURCE (
    echo [ERROR] Invalid Source specified: %2
    goto :show_usage
)

:: ============================================================================
:: --- VIRTUAL ENVIRONMENT SETUP ---
:: ============================================================================
echo [INFO]: Checking for Python virtual environment...

if exist "runtime\Scripts\activate.bat" (
    echo [INFO]: Virtual environment 'env' already exists. Activating it.
) else (
    echo [INFO]: Virtual environment not found. Creating it now...
    python -m venv env
    call :check_error
    echo [SUCCESS]: Virtual environment 'runtime' created.
)

:: Activate the virtual environment for this script session
call env\Scripts\activate
call :check_error
echo [SUCCESS]: Virtual environment activated.
:: ============================================================================


echo [INFO]: Checking for FFmpeg and CMake...
if exist "tools\ffmpeg\bin\ffmpeg.exe" (
    echo [INFO]: FFmpeg already exists.
) else (
    echo [INFO]: FFmpeg not found. Downloading...
    curl -L "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip" -o ffmpeg.zip
    call :check_error
    echo [INFO]: Extracting FFmpeg...
    tar -xf ffmpeg.zip
    call :check_error
    mkdir "tools\ffmpeg"
    move "ffmpeg-master-latest-win64-gpl"/* "tools/ffmpeg"
    del ffmpeg.zip
)

if exist "tools\cmake\bin\cmake.exe" (
    echo [INFO]: CMake already exists.
) else (
    echo [INFO]: CMake not found. Downloading...
    curl -L "https://github.com/Kitware/CMake/releases/download/v3.28.1/cmake-3.28.1-windows-x86_64.zip" -o cmake.zip
    call :check_error
    echo [INFO]: Extracting CMake...
    tar -xf cmake.zip
    call :check_error
    mkdir "tools\cmake"
    move "cmake-3.28.1-windows-x86_64"/* "tools/cmake"
    del cmake.zip
)

set "PATH=%CD%\tools\ffmpeg\bin;%CD%\tools\cmake\bin;%PATH%"
echo [SUCCESS]: FFmpeg & CMake are ready.

:: --- URL Configuration ---
set "PretrainedURL="
set "G2PWURL="
set "NLTKURL="
set "OpenJTalkURL="

if /i "%SOURCE%"=="HF" (
    echo [INFO] Download Model From HuggingFace
    set "PretrainedURL=https://huggingface.co/XXXXRT/GPT-SoVITS-Pretrained/resolve/main/pretrained_models.zip"
    set "G2PWURL=https://huggingface.co/XXXXRT/GPT-SoVITS-Pretrained/resolve/main/G2PWModel.zip"
    set "NLTKURL=https://huggingface.co/XXXXRT/GPT-SoVITS-Pretrained/resolve/main/nltk_data.zip"
    set "OpenJTalkURL=https://huggingface.co/XXXXRT/GPT-SoVITS-Pretrained/resolve/main/open_jtalk_dic_utf_8-1.11.tar.gz"
)
if /i "%SOURCE%"=="HF-Mirror" (
    echo [INFO] Download Model From HuggingFace-Mirror
    set "PretrainedURL=https://hf-mirror.com/XXXXRT/GPT-SoVITS-Pretrained/resolve/main/pretrained_models.zip"
    set "G2PWURL=https://hf-mirror.com/XXXXRT/GPT-SoVITS-Pretrained/resolve/main/G2PWModel.zip"
    set "NLTKURL=https://hf-mirror.com/XXXXRT/GPT-SoVITS-Pretrained/resolve/main/nltk_data.zip"
    set "OpenJTalkURL=https://hf-mirror.com/XXXXRT/GPT-SoVITS-Pretrained/resolve/main/open_jtalk_dic_utf_8-1.11.tar.gz"
)
if /i "%SOURCE%"=="ModelScope" (
    echo [INFO] Download Model From ModelScope
    set "PretrainedURL=https://www.modelscope.cn/models/XXXXRT/GPT-SoVITS-Pretrained/resolve/master/pretrained_models.zip"
    set "G2PWURL=https://www.modelscope.cn/models/XXXXRT/GPT-SoVITS-Pretrained/resolve/master/G2PWModel.zip"
    set "NLTKURL=https://www.modelscope.cn/models/XXXXRT/GPT-SoVITS-Pretrained/resolve/master/nltk_data.zip"
    set "OpenJTalkURL=https://www.modelscope.cn/models/XXXXRT/GPT-SoVITS-Pretrained/resolve/master/open_jtalk_dic_utf_8-1.11.tar.gz"
)

:: --- Download Models ---
if exist "GPT_SoVITS\pretrained_models\sv" (
    echo [INFO] Pretrained Model Exists. Skip Downloading.
) else (
    echo [INFO] Downloading Pretrained Models...
    curl -L "%PretrainedURL%" -o pretrained_models.zip
    call :check_error
    tar -xf pretrained_models.zip -C GPT_SoVITS
    call :check_error
    del pretrained_models.zip
    echo [SUCCESS] Pretrained Models Downloaded.
)

if exist "GPT_SoVITS\text\G2PWModel" (
    echo [INFO] G2PWModel Exists. Skip Downloading.
) else (
    echo [INFO] Downloading G2PWModel...
    curl -L "%G2PWURL%" -o G2PWModel.zip
    call :check_error
    tar -xf G2PWModel.zip -C GPT_SoVITS/text
    call :check_error
    del G2PWModel.zip
    echo [SUCCESS] G2PWModel Downloaded.
)

:: --- Install Python Dependencies ---
if "%DEVICE%"=="CU128" (
    echo [INFO] Installing PyTorch For CUDA 12.8...
    pip install torch torchaudio --index-url "https://download.pytorch.org/whl/cu128"
)
if "%DEVICE%"=="CU126" (
    echo [INFO] Installing PyTorch For CUDA 12.6...
    pip install torch torchaudio --index-url "https://download.pytorch.org/whl/cu126"
)
if "%DEVICE%"=="CPU" (
    echo [INFO] Installing PyTorch For CPU...
    pip install torch torchaudio --index-url "https://download.pytorch.org/whl/cpu"
)
call :check_error
echo [SUCCESS] PyTorch Installed.

echo [INFO] Installing Python Dependencies From requirements.txt...
pip install -r extra-req.txt --no-deps
call :check_error
pip install -r requirements.txt
call :check_error
echo [SUCCESS] Python Dependencies Installed.

echo [INFO] Downloading NLTK Data...
curl -L "%NLTKURL%" -o nltk_data.zip
call :check_error
for /f "delims=" %%i in ('python -c "import sys; print(sys.prefix)"') do set "PYTHON_PREFIX=%%i"
tar -xf nltk_data.zip -C "%PYTHON_PREFIX%"
call :check_error
del nltk_data.zip
echo [SUCCESS] NLTK Data Downloaded.

echo [INFO] Downloading Open JTalk Dict...
curl -L "%OpenJTalkURL%" -o open_jtalk_dic_utf_8-1.11.tar.gz
call :check_error
for /f "delims=" %%i in ('python -c "import os, pyopenjtalk; print(os.path.dirname(pyopenjtalk.__file__))"') do set "TARGET_DIR=%%i"
tar -xzf open_jtalk_dic_utf_8-1.11.tar.gz -C "%TARGET_DIR%"
call :check_error
del open_jtalk_dic_utf_8-1.11.tar.gz
echo [SUCCESS] Open JTalk Dic Downloaded.

echo.
echo [SUCCESS] Installation Completed!

goto :eof

:check_error
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] The previous command failed with error code %errorlevel%.
    echo [ERROR] Aborting installation.
    exit /b %errorlevel%
)
goto :eof


:show_usage
echo.
echo Invalid arguments. Please follow the syntax below.
echo.
echo   SYNTAX: install.bat [Device] [Source]
echo.
echo   [Device]: CU128, CU126, CPU
echo   [Source]: HF, HF-Mirror, ModelScope
echo.
echo   Example: install.bat CU128 HF
echo.

exit /b 1

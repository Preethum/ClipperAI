@echo off
echo Starting Label Studio server on port 8080...
echo Ensure Label Studio is installed via: pip install label-studio

:: Enable serving local files from your drive
set LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true
set LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=%~dp0

label-studio start --port 8080
pause

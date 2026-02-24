@echo off 
echo Stopping AI-Powered Chatbot System... 
taskkill /F /IM python.exe 2>nul 
taskkill /F /FI "WINDOWTITLE eq npm*" 2>nul 
echo All services stopped 
pause 

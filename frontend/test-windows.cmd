@echo off  
echo Checking Node.js...  
node --version  
  
echo Checking npm...  
npm --version  
  
echo Checking .env file...  
if exist .env (echo [OK] .env exists) else (echo [FAIL] .env not found)  
  
echo Frontend ready for testing! 

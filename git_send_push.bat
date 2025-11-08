@echo off
echo ==========================================
echo        GIT PUSH - ENVIAR CODIGO
echo ==========================================
echo.

REM Verificar se está dentro de um repositório git
git rev-parse --is-inside-work-tree >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo ERRO: Este diretorio nao é um repositório Git!
    pause
    exit /b
)

echo Status atual:
git status
echo.

set /p commit_msg=Digite a mensagem do commit: 

echo Adicionando arquivos...
git add .

echo Criando commit...
git commit -m "%commit_msg%"

echo Enviando para o GitHub...
git push

echo.
echo ✅ Código enviado com sucesso!
echo ==========================================
pause

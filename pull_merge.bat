@echo off
echo ==========================================
echo      GIT PULL + MERGE AUTOMATICO
echo ==========================================
echo.

REM Verificar se está dentro de um repositório git
git rev-parse --is-inside-work-tree >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo ERRO: Este diretorio nao é um repositório Git!
    pause
    exit /b
)

echo Branch atual:
git branch --show-current
echo.

set /p merge_branch=Digite o nome do branch que deseja mergear para o atual: 

echo.
echo 🔄 Fazendo PULL do repositório remoto...
git pull origin %merge_branch% --allow-unrelated-histories

echo.
echo 🔀 Realizando merge do branch %merge_branch% no branch atual...
git merge %merge_branch%

echo.
echo ✅ Merge realizado! Agora enviando para o GitHub...
git push

echo.
echo ✅ Processo concluído com sucesso!
echo ==========================================
pause

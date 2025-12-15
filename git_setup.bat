@echo off
REM Git Setup und Push zu GitHub
REM Für Big_Data_Umweltbeh-rde Repository

echo ========================================
echo Git Setup für EPA Dashboard
echo ========================================
echo.

REM Prüfen ob Git installiert ist
git --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Git ist nicht installiert!
    echo Bitte installieren Sie Git von: https://git-scm.com/
    pause
    exit /b 1
)

echo [OK] Git gefunden!
git --version
echo.

REM Git Status anzeigen
echo [1/6] Aktueller Git-Status:
echo ----------------------------------------
git status
echo.

REM Dateien hinzufügen
echo [2/6] Füge Dateien zu Git hinzu...
git add .

if errorlevel 1 (
    echo [ERROR] Fehler beim Hinzufügen der Dateien!
    pause
    exit /b 1
)

echo [OK] Dateien hinzugefügt!
echo.

REM Commit erstellen
echo [3/6] Erstelle Commit...
set /p commit_msg="Commit-Nachricht (Enter für Standard): "
if "%commit_msg%"=="" set commit_msg=Initial deployment setup for EPA Dashboard v2.0

git commit -m "%commit_msg%"

if errorlevel 1 (
    echo [WARNUNG] Commit fehlgeschlagen oder keine Änderungen
)
echo.

REM Branch auf main setzen
echo [4/6] Setze Branch auf 'main'...
git branch -M main
echo [OK] Branch ist jetzt 'main'
echo.

REM Remote prüfen/setzen
echo [5/6] Prüfe Remote-Repository...
git remote -v | findstr "origin" >nul
if errorlevel 1 (
    echo Remote 'origin' wird hinzugefügt...
    git remote add origin https://github.com/SebastianKuehnrich/Big_Data_Umweltbeh-rde.git
    echo [OK] Remote hinzugefügt!
) else (
    echo Remote 'origin' existiert bereits:
    git remote -v
    echo.
    echo Möchten Sie die Remote-URL aktualisieren? (J/N)
    set /p update_remote=
    if /i "%update_remote%"=="J" (
        git remote set-url origin https://github.com/SebastianKuehnrich/Big_Data_Umweltbeh-rde.git
        echo [OK] Remote-URL aktualisiert!
    )
)
echo.

REM Push zu GitHub
echo [6/6] Push zu GitHub...
echo ----------------------------------------
echo Dies wird Ihr Dashboard zu GitHub hochladen.
echo Danach können Sie es auf Railway deployen.
echo.
echo Möchten Sie jetzt pushen? (J/N)
set /p do_push=

if /i "%do_push%"=="J" (
    echo.
    echo Pushing zu GitHub...
    git push -u origin main

    if errorlevel 1 (
        echo.
        echo [ERROR] Push fehlgeschlagen!
        echo.
        echo Mögliche Ursachen:
        echo - Repository existiert nicht auf GitHub
        echo - Keine Berechtigung
        echo - Authentifizierung erforderlich
        echo.
        echo Erstellen Sie zuerst das Repository auf GitHub:
        echo https://github.com/new
        echo Name: Big_Data_Umweltbeh-rde
        echo.
        pause
        exit /b 1
    )

    echo.
    echo ========================================
    echo Push erfolgreich abgeschlossen!
    echo ========================================
    echo.
    echo Ihr Code ist jetzt auf GitHub verfügbar:
    echo https://github.com/SebastianKuehnrich/Big_Data_Umweltbeh-rde
    echo.
    echo Nächste Schritte:
    echo 1. Gehen Sie zu: https://railway.app
    echo 2. Login mit GitHub
    echo 3. "New Project" - "Deploy from GitHub repo"
    echo 4. Wählen Sie: Big_Data_Umweltbeh-rde
    echo 5. Railway deployt automatisch!
    echo 6. "Generate Domain" für öffentliche URL
    echo.
    echo Das Dashboard wird in ca. 2-3 Minuten verfügbar sein.
    echo.
) else (
    echo.
    echo Push abgebrochen.
    echo.
    echo Um später zu pushen, führen Sie aus:
    echo   git push -u origin main
    echo.
)

echo ========================================
echo Git-Setup abgeschlossen!
echo ========================================
echo.
pause


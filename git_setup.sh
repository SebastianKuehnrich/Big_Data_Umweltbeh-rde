#!/bin/bash
# Git Setup und Push zu GitHub
# Für Big_Data_Umweltbeh-rde Repository

echo "========================================"
echo "Git Setup für EPA Dashboard"
echo "========================================"
echo ""

# Prüfen ob Git installiert ist
if ! command -v git &> /dev/null; then
    echo "[ERROR] Git ist nicht installiert!"
    echo "Bitte installieren Sie Git von: https://git-scm.com/"
    exit 1
fi

echo "[OK] Git gefunden!"
git --version
echo ""

# Git Status anzeigen
echo "[1/6] Aktueller Git-Status:"
echo "----------------------------------------"
git status
echo ""

# Dateien hinzufügen
echo "[2/6] Füge Dateien zu Git hinzu..."
git add .

if [ $? -ne 0 ]; then
    echo "[ERROR] Fehler beim Hinzufügen der Dateien!"
    exit 1
fi

echo "[OK] Dateien hinzugefügt!"
echo ""

# Commit erstellen
echo "[3/6] Erstelle Commit..."
read -p "Commit-Nachricht (Enter für Standard): " commit_msg
if [ -z "$commit_msg" ]; then
    commit_msg="Initial deployment setup for EPA Dashboard v2.0"
fi

git commit -m "$commit_msg"

if [ $? -ne 0 ]; then
    echo "[WARNUNG] Commit fehlgeschlagen oder keine Änderungen"
fi
echo ""

# Branch auf main setzen
echo "[4/6] Setze Branch auf 'main'..."
git branch -M main
echo "[OK] Branch ist jetzt 'main'"
echo ""

# Remote prüfen/setzen
echo "[5/6] Prüfe Remote-Repository..."
if git remote -v | grep -q "origin"; then
    echo "Remote 'origin' existiert bereits:"
    git remote -v
    echo ""
    read -p "Möchten Sie die Remote-URL aktualisieren? (j/n) " update_remote
    if [ "$update_remote" = "j" ] || [ "$update_remote" = "J" ]; then
        git remote set-url origin https://github.com/SebastianKuehnrich/Big_Data_Umweltbeh-rde.git
        echo "[OK] Remote-URL aktualisiert!"
    fi
else
    echo "Remote 'origin' wird hinzugefügt..."
    git remote add origin https://github.com/SebastianKuehnrich/Big_Data_Umweltbeh-rde.git
    echo "[OK] Remote hinzugefügt!"
fi
echo ""

# Push zu GitHub
echo "[6/6] Push zu GitHub..."
echo "----------------------------------------"
echo "Dies wird Ihr Dashboard zu GitHub hochladen."
echo "Danach können Sie es auf Railway deployen."
echo ""
read -p "Möchten Sie jetzt pushen? (j/n) " do_push

if [ "$do_push" = "j" ] || [ "$do_push" = "J" ]; then
    echo ""
    echo "Pushing zu GitHub..."
    git push -u origin main

    if [ $? -ne 0 ]; then
        echo ""
        echo "[ERROR] Push fehlgeschlagen!"
        echo ""
        echo "Mögliche Ursachen:"
        echo "- Repository existiert nicht auf GitHub"
        echo "- Keine Berechtigung"
        echo "- Authentifizierung erforderlich"
        echo ""
        echo "Erstellen Sie zuerst das Repository auf GitHub:"
        echo "https://github.com/new"
        echo "Name: Big_Data_Umweltbeh-rde"
        echo ""
        exit 1
    fi

    echo ""
    echo "========================================"
    echo "Push erfolgreich abgeschlossen!"
    echo "========================================"
    echo ""
    echo "Ihr Code ist jetzt auf GitHub verfügbar:"
    echo "https://github.com/SebastianKuehnrich/Big_Data_Umweltbeh-rde"
    echo ""
    echo "Nächste Schritte:"
    echo "1. Gehen Sie zu: https://railway.app"
    echo "2. Login mit GitHub"
    echo "3. 'New Project' - 'Deploy from GitHub repo'"
    echo "4. Wählen Sie: Big_Data_Umweltbeh-rde"
    echo "5. Railway deployt automatisch!"
    echo "6. 'Generate Domain' für öffentliche URL"
    echo ""
    echo "Das Dashboard wird in ca. 2-3 Minuten verfügbar sein."
    echo ""
else
    echo ""
    echo "Push abgebrochen."
    echo ""
    echo "Um später zu pushen, führen Sie aus:"
    echo "  git push -u origin main"
    echo ""
fi

echo "========================================"
echo "Git-Setup abgeschlossen!"
echo "========================================"
echo ""


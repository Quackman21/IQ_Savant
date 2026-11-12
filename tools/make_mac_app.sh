#!/bin/bash
# Builds "IQ Savant" on your Desktop with your logo. Run once (and again if you move the folder or change the logo):
#     bash tools/make_mac_app.sh                        # logo: ~/Dev/images/IQ RB Paisley Logo.png
#     bash tools/make_mac_app.sh "/path/to/other.png"
set -e
REPO="$(cd "$(dirname "$0")/.." && pwd)"
LOGO="${1:-$HOME/Dev/images/IQ RB Paisley Logo.png}"
APP="$HOME/Desktop/IQ Savant.app"
[ -f "$LOGO" ] || { echo "Logo not found: $LOGO"; exit 1; }
chmod +x "$REPO/tools/iq_app.sh"

# 1. icon (every size macOS wants) from the PNG
TMP="$(mktemp -d)"; SET="$TMP/IQ.iconset"; mkdir "$SET"
for s in 16 32 128 256 512; do
  sips -s format png -z $s $s "$LOGO" --out "$SET/icon_${s}x${s}.png" >/dev/null
  sips -s format png -z $((s * 2)) $((s * 2)) "$LOGO" --out "$SET/icon_${s}x${s}@2x.png" >/dev/null
done
iconutil -c icns "$SET" -o "$TMP/IQ.icns"

# 2. the app: asks what to do if the site is already running
cat > "$TMP/launcher.applescript" <<OSA
on run
  set sh to quoted form of "$REPO/tools/iq_app.sh"
  if (do shell script sh & " status") is "running" then
    set r to button returned of (display dialog "IQ Savant is running." buttons {"Stop", "Restart", "Open"} default button "Open" with title "IQ Savant" with icon (path to resource "applet.icns"))
    if r is "Open" then do shell script sh & " open"
    if r is "Stop" then do shell script sh & " stop"
    if r is "Restart" then do shell script sh & " restart > /dev/null 2>&1 &"
  else
    do shell script sh & " start > /dev/null 2>&1 &"
  end if
end run
OSA
rm -rf "$APP"
osacompile -o "$APP" "$TMP/launcher.applescript"

# 3. put the logo on it (newer macOS keeps the default icon in Assets.car -- remove it so ours shows)
cp "$TMP/IQ.icns" "$APP/Contents/Resources/applet.icns"
rm -f "$APP/Contents/Resources/Assets.car"
/usr/libexec/PlistBuddy -c "Delete :CFBundleIconName" "$APP/Contents/Info.plist" 2>/dev/null || true
/usr/libexec/PlistBuddy -c "Set :CFBundleIconFile applet" "$APP/Contents/Info.plist" 2>/dev/null || \
  /usr/libexec/PlistBuddy -c "Add :CFBundleIconFile string applet" "$APP/Contents/Info.plist"
codesign --force --deep -s - "$APP" >/dev/null 2>&1 || true
touch "$APP"
rm -rf "$TMP"
echo "Made: $APP"
echo "Double-click it to start the site. Drag it to your Dock to keep it there."

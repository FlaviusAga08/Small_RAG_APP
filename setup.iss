; Inno Setup script for Offline RAG — fully self-contained.
; Bundles: app + Ollama + both LLM models. Zero internet needed on her laptop.
;
; Before compiling, make sure these exist next to this file:
;   - dist\OfflineRAG\          (from: pyinstaller RagApp.spec)
;   - OllamaSetup.exe           (from: ollama.com/download/windows)
;   - ollama_models\            (from: xcopy "%USERPROFILE%\.ollama\models" "ollama_models\" /E /I)
;
; Output: Output\OfflineRAG-Setup.exe

#define MyAppName    "Offline RAG"
#define MyAppVersion "1.0"
#define MyAppExe     "OfflineRAG.exe"

[Setup]
AppName={#MyAppName}
AppVersion={#MyAppVersion}
DefaultDirName={localappdata}\Programs\OfflineRAG
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputBaseFilename=OfflineRAG-Setup
; lzma2/fast instead of /max — models are already compressed, max just wastes time
Compression=lzma2/fast
SolidCompression=no
ChangesEnvironment=yes
WizardStyle=modern

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Shortcuts:"

[Files]
; the app
Source: "dist\OfflineRAG\*"; DestDir: "{app}"; Flags: recursesubdirs ignoreversion

; Ollama installer — extracted to temp, deleted after install
Source: "OllamaSetup.exe"; DestDir: "{tmp}"; Flags: deleteafterinstall

; the two models — copied straight into her Ollama models folder
Source: "ollama_models\*"; DestDir: "{userdocs}\..\AppData\Roaming\Ollama\models"; Flags: recursesubdirs ignoreversion uninsneveruninstall

[Icons]
Name: "{group}\{#MyAppName}";    Filename: "{app}\{#MyAppExe}"
Name: "{userdesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExe}"; Tasks: desktopicon

[Registry]
; GPU + RAM tuning — per-user, no admin needed
Root: HKCU; Subkey: "Environment"; ValueType: string; ValueName: "OLLAMA_VULKAN";            ValueData: "1";    Flags: preservestringtype
Root: HKCU; Subkey: "Environment"; ValueType: string; ValueName: "OLLAMA_MAX_LOADED_MODELS"; ValueData: "1";    Flags: preservestringtype
Root: HKCU; Subkey: "Environment"; ValueType: string; ValueName: "OLLAMA_NUM_PARALLEL";      ValueData: "1";    Flags: preservestringtype
Root: HKCU; Subkey: "Environment"; ValueType: string; ValueName: "OLLAMA_KEEP_ALIVE";        ValueData: "5m";   Flags: preservestringtype
Root: HKCU; Subkey: "Environment"; ValueType: string; ValueName: "OLLAMA_FLASH_ATTENTION";   ValueData: "1";    Flags: preservestringtype
Root: HKCU; Subkey: "Environment"; ValueType: string; ValueName: "OLLAMA_KV_CACHE_TYPE";     ValueData: "q8_0"; Flags: preservestringtype

[Run]
; install Ollama silently
Filename: "{tmp}\OllamaSetup.exe"; Parameters: "/VERYSILENT /SUPPRESSMSGBOXES /NORESTART"; StatusMsg: "Installing Ollama…"; Flags: waituntilterminated
; offer to launch
Filename: "{app}\{#MyAppExe}"; Description: "Launch {#MyAppName}"; Flags: nowait postinstall skipifsilent

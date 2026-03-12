param(
    [switch]$Attach,
    [switch]$Once,
    [double]$RefreshSec = 2.0,
    [int]$MaxKeyframes = 8,
    [string]$Title = "AURA Memory Monitor",
    [string]$MemoryDbPath = "",
    [string]$KeyframeDir = "",
    [string]$PythonExe = ""
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoDir = [System.IO.Path]::GetFullPath((Join-Path $ScriptDir "..\.."))
$ResolvedScriptPath = [System.IO.Path]::GetFullPath($MyInvocation.MyCommand.Path)
$InvariantCulture = [System.Globalization.CultureInfo]::InvariantCulture

if ([string]::IsNullOrWhiteSpace($MemoryDbPath)) {
    $MemoryDbPath = Join-Path $RepoDir "state\memory\memory.sqlite"
}
if ([string]::IsNullOrWhiteSpace($KeyframeDir)) {
    $KeyframeDir = Join-Path $RepoDir "state\memory\keyframes"
}

function Resolve-PythonInvoker {
    param(
        [Parameter(Mandatory = $false)]
        [AllowEmptyString()]
        [string]$RequestedPythonExe
    )

    if (-not [string]::IsNullOrWhiteSpace($RequestedPythonExe)) {
        $resolvedRequested = Get-Command $RequestedPythonExe -ErrorAction SilentlyContinue
        return @{
            Exe = if ($null -ne $resolvedRequested) { $resolvedRequested.Source } else { $RequestedPythonExe }
            Args = @()
        }
    }

    if (-not [string]::IsNullOrWhiteSpace($env:PYTHON_EXE)) {
        $resolvedEnv = Get-Command $env:PYTHON_EXE -ErrorAction SilentlyContinue
        return @{
            Exe = if ($null -ne $resolvedEnv) { $resolvedEnv.Source } else { $env:PYTHON_EXE }
            Args = @()
        }
    }

    $pythonCommand = Get-Command "python.exe" -ErrorAction SilentlyContinue
    if ($null -ne $pythonCommand) {
        return @{
            Exe = $pythonCommand.Source
            Args = @()
        }
    }

    $pyCommand = Get-Command "py.exe" -ErrorAction SilentlyContinue
    if ($null -ne $pyCommand) {
        return @{
            Exe = $pyCommand.Source
            Args = @("-3")
        }
    }

    $pythonFallback = Get-Command "python" -ErrorAction SilentlyContinue
    if ($null -ne $pythonFallback) {
        return @{
            Exe = $pythonFallback.Source
            Args = @()
        }
    }

    throw "python executable not found. Set PYTHON_EXE or pass -PythonExe explicitly."
}

function Start-MemoryMonitorTerminal {
    param(
        [Parameter(Mandatory = $true)]
        [string]$ResolvedScriptPath,
        [Parameter(Mandatory = $true)]
        [string]$RepoDir,
        [Parameter(Mandatory = $true)]
        [string]$Title,
        [Parameter(Mandatory = $true)]
        [string]$MemoryDbPath,
        [Parameter(Mandatory = $true)]
        [string]$KeyframeDir,
        [Parameter(Mandatory = $true)]
        [AllowEmptyString()]
        [string]$PythonExe,
        [Parameter(Mandatory = $true)]
        [double]$RefreshSec,
        [Parameter(Mandatory = $true)]
        [int]$MaxKeyframes
    )

    $launchArgs = @(
        "-NoProfile",
        "-ExecutionPolicy", "Bypass",
        "-File", $ResolvedScriptPath,
        "-Attach",
        "-RefreshSec", $RefreshSec.ToString("0.###", $InvariantCulture),
        "-MaxKeyframes", $MaxKeyframes.ToString($InvariantCulture),
        "-Title", $Title,
        "-MemoryDbPath", $MemoryDbPath,
        "-KeyframeDir", $KeyframeDir
    )

    if (-not [string]::IsNullOrWhiteSpace($PythonExe)) {
        $launchArgs += @("-PythonExe", $PythonExe)
    }

    $wtCommand = Get-Command "wt.exe" -ErrorAction SilentlyContinue
    if ($null -ne $wtCommand) {
        & $wtCommand.Source "new-tab" "--title" $Title "powershell.exe" @launchArgs
        if (($null -ne $LASTEXITCODE) -and ($LASTEXITCODE -ne 0)) {
            throw "wt.exe failed with exit code $LASTEXITCODE"
        }
        return
    }

    Start-Process -FilePath "powershell.exe" -ArgumentList $launchArgs -WorkingDirectory $RepoDir | Out-Null
}

$monitorCode = @'
import argparse
import json
import sqlite3
import time
from datetime import datetime
from pathlib import Path


def fmt_timestamp(value):
    if value in (None, "", 0, 0.0):
        return "-"
    try:
        return datetime.fromtimestamp(float(value)).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return str(value)


def load_snapshot(db_path):
    if not db_path.exists():
        return None, "missing"
    try:
        with sqlite3.connect(db_path) as conn:
            row = conn.execute(
                "SELECT snapshot_id, created_at, payload_json FROM memory_snapshots "
                "ORDER BY snapshot_id DESC LIMIT 1"
            ).fetchone()
    except sqlite3.OperationalError as exc:
        return None, f"sqlite error: {type(exc).__name__}: {exc}"
    except Exception as exc:
        return None, f"query failed: {type(exc).__name__}: {exc}"
    if row is None:
        return None, "none"
    try:
        payload = json.loads(str(row[2]))
    except Exception as exc:
        return None, f"payload decode failed: {type(exc).__name__}: {exc}"
    return {
        "snapshot_id": int(row[0]),
        "created_at": row[1],
        "payload": payload,
    }, ""


def recent_keyframes(keyframe_dir, limit):
    if not keyframe_dir.exists():
        return []
    files = sorted(
        [path for path in keyframe_dir.glob("*.jpg") if path.is_file()],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return files[: max(int(limit), 0)]


def render(memory_db_path, keyframe_dir, max_keyframes):
    print("AURA Memory Monitor")
    print(f"time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"memory_db_path: {memory_db_path}")
    print(f"keyframe_dir: {keyframe_dir}")
    print()

    if memory_db_path.exists():
        stat = memory_db_path.stat()
        print(
            "memory.sqlite: present "
            f"size={stat.st_size}B "
            f"mtime={fmt_timestamp(stat.st_mtime)}"
        )
    else:
        print("memory.sqlite: missing")

    snapshot, snapshot_status = load_snapshot(memory_db_path)
    if snapshot is None:
        print(f"latest snapshot: {snapshot_status}")
    else:
        payload = snapshot["payload"]
        scratchpad = payload.get("scratchpad") or {}
        objects = payload.get("objects") or []
        places = payload.get("places") or []
        semantic_rules = payload.get("semantic_rules") or []
        keyframes = payload.get("keyframes") or []
        print(
            "latest snapshot: "
            f"id={snapshot['snapshot_id']} "
            f"created_at={fmt_timestamp(snapshot['created_at'])}"
        )
        print(
            "snapshot counts: "
            f"objects={len(objects)} "
            f"places={len(places)} "
            f"semantic_rules={len(semantic_rules)} "
            f"keyframe_meta={len(keyframes)}"
        )
        checked_locations = scratchpad.get("checked_locations") or []
        checked_text = ", ".join(str(item) for item in checked_locations[-5:]) if checked_locations else "-"
        print("scratchpad:")
        print(f"  instruction   : {str(scratchpad.get('instruction', '')).strip() or '-'}")
        print(f"  task_state    : {str(scratchpad.get('task_state', '')).strip() or '-'}")
        print(f"  goal_summary  : {str(scratchpad.get('goal_summary', '')).strip() or '-'}")
        print(f"  checked       : {checked_text}")
        print(f"  recent_hint   : {str(scratchpad.get('recent_hint', '')).strip() or '-'}")
        print(f"  next_priority : {str(scratchpad.get('next_priority', '')).strip() or '-'}")

    print()
    keyframe_files = recent_keyframes(keyframe_dir, max_keyframes)
    if not keyframe_dir.exists():
        print("keyframe_dir: missing")
    else:
        all_files = [path for path in keyframe_dir.glob("*.jpg") if path.is_file()]
        print(f"keyframe files: {len(all_files)}")
        for path in keyframe_files:
            stat = path.stat()
            print(
                f"  {path.name:<32} "
                f"{fmt_timestamp(stat.st_mtime)} "
                f"{stat.st_size}B"
            )

    print()
    print("note: keyframe files update live; SQLite scratchpad/object counts update only when a memory snapshot is persisted.")


def main():
    parser = argparse.ArgumentParser(description="Render AURA memory monitor output.")
    parser.add_argument("--memory-db-path", required=True)
    parser.add_argument("--keyframe-dir", required=True)
    parser.add_argument("--refresh-sec", type=float, default=2.0)
    parser.add_argument("--max-keyframes", type=int, default=8)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    memory_db_path = Path(args.memory_db_path).expanduser()
    keyframe_dir = Path(args.keyframe_dir).expanduser()
    refresh_sec = max(float(args.refresh_sec), 0.2)

    if args.once:
        render(memory_db_path, keyframe_dir, args.max_keyframes)
        return 0

    while True:
        print("\033c", end="")
        render(memory_db_path, keyframe_dir, args.max_keyframes)
        time.sleep(refresh_sec)


if __name__ == "__main__":
    raise SystemExit(main())
'@

if (-not $Attach) {
    Start-MemoryMonitorTerminal `
        -ResolvedScriptPath $ResolvedScriptPath `
        -RepoDir $RepoDir `
        -Title $Title `
        -MemoryDbPath $MemoryDbPath `
        -KeyframeDir $KeyframeDir `
        -PythonExe $PythonExe `
        -RefreshSec $RefreshSec `
        -MaxKeyframes $MaxKeyframes
    exit 0
}

try {
    $Host.UI.RawUI.WindowTitle = $Title
}
catch {
}

$pythonInvoker = Resolve-PythonInvoker -RequestedPythonExe $PythonExe

$pythonArgs = @()
$pythonArgs += $pythonInvoker.Args
$pythonArgs += @(
    "-c", $monitorCode,
    "--memory-db-path", $MemoryDbPath,
    "--keyframe-dir", $KeyframeDir,
    "--refresh-sec", $RefreshSec.ToString("0.###", $InvariantCulture),
    "--max-keyframes", $MaxKeyframes.ToString($InvariantCulture)
)
if ($Once) {
    $pythonArgs += "--once"
}

Push-Location $RepoDir
try {
    & $pythonInvoker.Exe @pythonArgs
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}

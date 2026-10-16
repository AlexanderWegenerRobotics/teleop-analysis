"""Start-menu / desktop shortcut for the viewer on Windows, so it can be found by search and pinned to the taskbar."""

from __future__ import annotations

import subprocess
import sys
import sysconfig
import tempfile
from pathlib import Path

APP_ID = "teleop.analysis.viewer"
ICON = Path(__file__).with_name("assets") / "teleop-analysis.ico"

_PS = r'''
$ErrorActionPreference = "Stop"
$lnk = "__LNK__"
New-Item -ItemType Directory -Force -Path (Split-Path $lnk) | Out-Null
$sh = New-Object -ComObject WScript.Shell
$s = $sh.CreateShortcut($lnk)
$s.TargetPath = "__TARGET__"
$s.Arguments = "__ARGS__"
$s.WorkingDirectory = "__WORKDIR__"
$s.IconLocation = "__ICON__,0"
$s.Description = "teleop-analysis log viewer"
$s.Save()
try {
Add-Type -TypeDefinition @"
using System;
using System.Runtime.InteropServices;
[StructLayout(LayoutKind.Sequential, Pack = 4)]
public struct PROPERTYKEY { public Guid fmtid; public uint pid; }
[StructLayout(LayoutKind.Explicit)]
public struct PROPVARIANT { [FieldOffset(0)] public ushort vt; [FieldOffset(8)] public IntPtr p; }
[ComImport, Guid("886D8EEB-8CF2-4446-8D02-CDBA1DBDCF99"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
public interface IPropertyStore {
  int GetCount(out uint c); int GetAt(uint i, out PROPERTYKEY k); int GetValue(ref PROPERTYKEY k, out PROPVARIANT v);
  int SetValue(ref PROPERTYKEY k, ref PROPVARIANT v); int Commit();
}
[ComImport, Guid("0000010b-0000-0000-C000-000000000046"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
public interface IPersistFile {
  int GetClassID(out Guid g); int IsDirty(); int Load([MarshalAs(UnmanagedType.LPWStr)] string f, uint m);
  int Save([MarshalAs(UnmanagedType.LPWStr)] string f, bool r); int SaveCompleted([MarshalAs(UnmanagedType.LPWStr)] string f);
  int GetCurFile([MarshalAs(UnmanagedType.LPWStr)] out string f);
}
[ComImport, Guid("00021401-0000-0000-C000-000000000046")] public class ShellLink {}
public static class AppId {
  public static void Set(string lnk, string id) {
    var link = new ShellLink();
    ((IPersistFile)link).Load(lnk, 2);
    var key = new PROPERTYKEY { fmtid = new Guid("9F4C2855-9F79-4B39-A8D0-E1D42DE1D5F3"), pid = 5 };
    var v = new PROPVARIANT { vt = 31, p = Marshal.StringToCoTaskMemUni(id) };
    var store = (IPropertyStore)link;
    store.SetValue(ref key, ref v);
    store.Commit();
    ((IPersistFile)link).Save(lnk, true);
    Marshal.FreeCoTaskMem(v.p);
  }
}
"@
[AppId]::Set($lnk, "__APPID__")
} catch { Write-Host "note: could not set the taskbar id ($_)" }
Write-Host "created $lnk"
'''


def _target() -> tuple[str, str]:
    scripts = Path(sysconfig.get_path("scripts"))
    exe = scripts / "teleop-analysis-gui.exe"
    if exe.exists():
        return str(exe), ""
    pyw = Path(sys.executable).with_name("pythonw.exe")
    return str(pyw if pyw.exists() else Path(sys.executable)), "-m teleop_analysis"


def _stable_icon() -> Path:
    import hashlib
    import os
    import shutil
    data = ICON.read_bytes()
    folder = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "teleop-analysis" / "icons"
    folder.mkdir(parents=True, exist_ok=True)
    out = folder / f"teleop-analysis-{hashlib.sha1(data).hexdigest()[:10]}.ico"
    if not out.exists():
        shutil.copyfile(ICON, out)
    return out


def _refresh_icon_cache() -> None:
    for args in (["ie4uinit.exe", "-show"], ["ie4uinit.exe", "-ClearIconCache"]):
        try:
            subprocess.run(args, capture_output=True, timeout=20)
        except Exception:
            pass


def install(desktop: bool = False) -> list[Path]:
    if sys.platform != "win32":
        raise SystemExit("shortcuts are only made on Windows")
    import os
    target, args = _target()
    icon = _stable_icon()
    places = [Path(os.environ["APPDATA"]) / "Microsoft" / "Windows" / "Start Menu" / "Programs"]
    if desktop:
        places.append(Path.home() / "Desktop")
    made = []
    for place in places:
        lnk = place / "teleop-analysis.lnk"
        ps = (_PS.replace("__LNK__", str(lnk)).replace("__TARGET__", target).replace("__ARGS__", args)
              .replace("__WORKDIR__", str(Path.home())).replace("__ICON__", str(icon)).replace("__APPID__", APP_ID))
        with tempfile.NamedTemporaryFile("w", suffix=".ps1", delete=False, encoding="utf-8-sig") as f:
            f.write(ps)
        r = subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", f.name],
                           capture_output=True, text=True)
        Path(f.name).unlink(missing_ok=True)
        print((r.stdout + r.stderr).strip())
        if r.returncode == 0:
            made.append(lnk)
    _refresh_icon_cache()
    return made


def set_app_id() -> None:
    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(APP_ID)
        except Exception:
            pass


def main(argv=None) -> None:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--desktop", action="store_true", help="also put a shortcut on the desktop")
    a = ap.parse_args(argv)
    made = install(a.desktop)
    if made:
        print("Search for 'teleop-analysis' in the start menu; right-click it -> Pin to taskbar / Pin to Start.")

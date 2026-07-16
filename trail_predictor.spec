import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_all, copy_metadata


streamlit_datas, streamlit_binaries, streamlit_hiddenimports = collect_all("streamlit")

datas = streamlit_datas + copy_metadata("streamlit") + [
    ("app.py", "."),
    ("config/defaults.yaml", "config"),
    (".streamlit/config.toml", ".streamlit"),
]

a = Analysis(
    ["desktop_launcher.py"],
    pathex=["."],
    binaries=streamlit_binaries,
    datas=datas,
    hiddenimports=streamlit_hiddenimports + ["app"],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["pytest", "scipy"],
    noarchive=False,
    optimize=0,
)

# A venv created from Miniconda does not contain its own conda-meta directory.
# PyInstaller may then resolve OpenSSL DLLs from another application on PATH
# (for example MySQL), which is incompatible with Miniconda's _ssl.pyd.
conda_binary_directory = Path(sys.base_prefix) / "Library" / "bin"
openssl_binaries = {
    name: conda_binary_directory / name
    for name in ("libcrypto-3-x64.dll", "libssl-3-x64.dll")
}
missing_openssl = [str(path) for path in openssl_binaries.values() if not path.is_file()]
if missing_openssl:
    raise FileNotFoundError(f"Missing Miniconda OpenSSL runtime: {missing_openssl}")
a.binaries = [
    (destination, str(openssl_binaries[Path(destination).name.lower()]), type_code)
    if Path(destination).name.lower() in openssl_binaries
    else (destination, source, type_code)
    for destination, source, type_code in a.binaries
]
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="TrailTimePredictor",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="TrailTimePredictor",
)

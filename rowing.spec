# -*- mode: python ; coding: utf-8 -*-

import os
import sys

block_cipher = None

project = 'rowing'
scripts = {
    'garmin': 'garmin.py', 
    'gpx': 'gpx.py'
}
datas = [('data', 'data'), ('cloudscraper', 'cloudscraper')]

analyses = {
    n: Analysis(
        [script],
        pathex=[SPECPATH],
        binaries=[],
        datas=datas,
        hiddenimports=['cloudscraper'],
        hookspath=[],
        runtime_hooks=[],
        excludes=[],
        win_no_prefer_redirects=False,
        win_private_assemblies=False,
        cipher=block_cipher,
        noarchive=False
    )
    for n, script in scripts.items()
}

MERGE(*((a, n, os.path.join(project, n)) for n, a in analyses.items()))

pyzs = {}
exes = {}

for n, a in analyses.items():
    pyzs[n] = PYZ(
        a.pure, a.zipped_data, cipher=block_cipher
    )
    exes[n] = EXE(
        pyzs[n],
        a.scripts,
        [],
        exclude_binaries=True,
        name=os.path.join('build', 'pyi.'+ sys.platform, project, n), 
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=True,
        console=True
    )

collect_args = sum((
    (exes[n], a.binaries, a.zipfiles, a.datas) 
    for n, a in analyses.items()
), 
    ()
)
col = COLLECT(
    *collect_args,
    strip=False,
    upx=True,
    upx_exclude=[],
    name=project
)

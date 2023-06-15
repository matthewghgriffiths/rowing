# -*- mode: python ; coding: utf-8 -*-

import os
import sys
import importlib

block_cipher = None

project = 'rowing'
scripts = {
    'world_rowing': 'rowing/world_rowing/app.py',
    'garmin': 'rowing/analysis/garmin.py', 
    'gpx': 'rowing/analysis/files.py'
}
streamlit_path = os.path.split(
    importlib.util.find_spec("streamlit").origin)[0] 
altair_path = os.path.split(
    importlib.util.find_spec("altair").origin)[0] 

datas = [
    ('data', 'data'), 
    ('cloudscraper', 'cloudscraper'),
    ('world_rowing_app', 'world_rowing_app'),
    ('rowing', 'rowing'),
    (streamlit_path + '/static', 'streamlit/static'),
    (altair_path + '/vegalite/v5/schema/vega-lite-schema.json', 
    'altair/vegalite/v5/schema/vega-lite-schema.json'),
]

analyses = {
    n: Analysis(
        [script],
        pathex=[SPECPATH],
        binaries=[],
        datas=datas,
        hiddenimports=[
            'cloudscraper', 
            'pyarrow.vendored.version', 
            'tqdm.auto', 
            'tqdm.autonotebook', 
            'scipy.stats',
            'scipy.spatial.distance',
            'scipy.spatial.transform._rotation_groups',
            'scipy.special.cython_special',
            'scipy.special',
            'scipy.linalg',
            'scipy.integrate',
            'streamlit', 
            'streamlit.runtime.scriptrunner.magic_funcs', 
        ],
        hookspath=['./hooks'],
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

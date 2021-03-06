# -*- mode: python ; coding: utf-8 -*-

import gooey

gooey_root = os.path.dirname(gooey.__file__)
gooey_languages = Tree(os.path.join(gooey_root, 'languages'), prefix = 'gooey/languages') # copy default languages     
gooey_images = Tree("icons", prefix="icons")

block_cipher = None

binaries = []

# import site

# typelib_path = "/usr/local/lib/girepository-1.0/"
# binaries=[(os.path.join(typelib_path, tl), 'gi_typelibs') for tl in os.listdir(typelib_path)]

a = Analysis(['goocompressor.py'],
             pathex=['/Users/volzotan/GIT/compressor'],
             binaries=[],
             datas=[],
             hiddenimports=[],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,
          [],
          gooey_languages, 
          gooey_images,
          name='Compressor',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          upx_exclude=[],
          runtime_tmpdir=None,
          console=False,

          icon=os.path.join(gooey_root, 'images', 'program_icon.ico'))

app = BUNDLE(exe,
          name='Compressor.app',
          icon='icons/program_icon.icns',
          bundle_identifier=None)
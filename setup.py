from cx_Freeze import setup, Executable

import sys


# Dependencies are automatically detected, but it might need
# fine tuning.
build_options = {'packages': [], 'excludes': []}

base = 'Win32GUI' if sys.platform=='win32' else None

executables = [
    Executable('alien_invasion.py', base=base, target_name = 'Alien Invasion.exe')
]

setup(
    name='Alien Invasion',
    version = '0.9.0',
    description = 'A simple game written in pygame.',
    options = {'build_exe': build_options},
    executables = executables
)
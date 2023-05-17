from cx_Freeze import setup, Executable

import sys


build_options = {"packages": ["pygame"], "excludes": ["tkinter"], "include_files": ["images", "ai.ico", "profile"]}

base = "Win32GUI" if sys.platform=="win32" else None

executables = [
    Executable("alien_invasion.py", base=base, target_name="Alien Invasion", icon="ai.ico")
]

msi_data = {
    "Icon": [
        ("IconId", "ai.ico"),
    ],
    "Shortcut": [
        ("DesktopShortcut", "DesktopFolder", "Alien Invasion", "TARGETDIR", "[TARGETDIR]Alien Invasion.exe", None, None, None, None, None, None, "TARGETDIR"),
        ("StartMenuShortcut", "StartMenuFolder", "Alien Invasion", "TARGETDIR", "[TARGETDIR]Alien Invasion.exe", None, None, None, None, None, None, "TARGETDIR")
    ],
}

bdist_msi_options = {
    "data": msi_data,
    "initial_target_dir": r"[ProgramFilesFolder]\Alien Invasion",
}

setup(
    name="Alien Invasion",
    version="1.0.1",
    description="A simple game written in pygame from scratch by EchineF.",
    author_email="hosein.fanai@gmail.com",
    download_url="https://github.com/hosein-fanai/Alien-Invasion-with-Deep-RL-from-scratch/dist/Alien Invasion-1.0.1-win64.msi",
    options = {
        "build_exe": build_options,
        "bdist_msi": bdist_msi_options,
    },
    executables=executables
)
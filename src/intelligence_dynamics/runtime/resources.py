"""Operating-system process memory observations."""

from __future__ import annotations

import os
from pathlib import Path

from ..validation import integer


def resident_bytes(pid: int) -> int | None:
    """Read current resident memory for one process, or None when unavailable."""
    integer(pid, "pid", minimum=1)
    if os.name == "posix" and Path("/proc").is_dir():
        try:
            fields = Path(f"/proc/{pid}/statm").read_text(encoding="ascii").split()
            pages = int(fields[1])
            return integer(pages, "resident pages") * os.sysconf("SC_PAGE_SIZE")
        except (OSError, ValueError, IndexError):
            return None
    if os.name == "nt":
        return _windows_resident_bytes(pid)
    return None


def _windows_resident_bytes(pid):
    import ctypes
    from ctypes import wintypes

    class Counters(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.CloseHandle.restype = wintypes.BOOL
    psapi.GetProcessMemoryInfo.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(Counters),
        wintypes.DWORD,
    ]
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
    handle = kernel.OpenProcess(0x0400 | 0x0010, False, pid)
    if not handle:
        error = ctypes.get_last_error()
        if error == 87:
            return None
        raise ctypes.WinError(error)
    try:
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        if not psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
            raise ctypes.WinError(ctypes.get_last_error())
        return int(counters.WorkingSetSize)
    finally:
        import sys

        primary = sys.exc_info()[1]
        if not kernel.CloseHandle(handle):
            failure = ctypes.WinError(ctypes.get_last_error())
            if primary is None:
                raise failure
            if hasattr(primary, "add_note"):
                primary.add_note(f"handle close failed: {failure}")
            else:
                primary.cleanup_errors = (str(failure),)


def descendant_ids(parents: dict[int, int], root: int) -> tuple[int, ...]:
    """Return a rooted process family from one parent-ID observation."""
    integer(root, "root", minimum=1)
    for pid, parent in parents.items():
        integer(pid, "pid", minimum=1)
        integer(parent, "parent")
    selected = {root}
    while True:
        children = {pid for pid, parent in parents.items() if parent in selected}
        expanded = selected | children
        if expanded == selected:
            return tuple(sorted(selected))
        selected = expanded


def process_tree_resident_bytes(pid: int) -> int | None:
    """Sum available resident-memory observations for a rooted process family."""
    integer(pid, "pid", minimum=1)
    if os.name == "nt":
        parents = _windows_process_parents()
    elif os.name == "posix" and Path("/proc").is_dir():
        parents = {}
        for directory in Path("/proc").iterdir():
            if not directory.name.isdecimal():
                continue
            try:
                tail = (directory / "stat").read_text().rsplit(")", 1)[1].split()
                parents[int(directory.name)] = int(tail[1])
            except (FileNotFoundError, ProcessLookupError):
                continue
    else:
        return None
    if pid not in parents:
        return None
    observed = [resident_bytes(child) for child in descendant_ids(parents, pid)]
    values = [value for value in observed if value is not None]
    return sum(values) if values else None


def _windows_process_parents():
    import ctypes
    from ctypes import wintypes

    class Entry(ctypes.Structure):
        _fields_ = [
            ("dwSize", wintypes.DWORD),
            ("cntUsage", wintypes.DWORD),
            ("th32ProcessID", wintypes.DWORD),
            ("th32DefaultHeapID", ctypes.c_size_t),
            ("th32ModuleID", wintypes.DWORD),
            ("cntThreads", wintypes.DWORD),
            ("th32ParentProcessID", wintypes.DWORD),
            ("pcPriClassBase", wintypes.LONG),
            ("dwFlags", wintypes.DWORD),
            ("szExeFile", wintypes.WCHAR * 260),
        ]

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
    kernel.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    for name in ("Process32FirstW", "Process32NextW"):
        function = getattr(kernel, name)
        function.argtypes = [wintypes.HANDLE, ctypes.POINTER(Entry)]
        function.restype = wintypes.BOOL
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = kernel.CreateToolhelp32Snapshot(2, 0)
    if handle == ctypes.c_void_p(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    result = {}
    try:
        entry = Entry()
        entry.dwSize = ctypes.sizeof(entry)
        success = kernel.Process32FirstW(handle, ctypes.byref(entry))
        while success:
            if entry.th32ProcessID:
                result[int(entry.th32ProcessID)] = int(entry.th32ParentProcessID)
            success = kernel.Process32NextW(handle, ctypes.byref(entry))
        error = ctypes.get_last_error()
        if error != 18:
            raise ctypes.WinError(error)
        return result
    finally:
        import sys

        primary = sys.exc_info()[1]
        if not kernel.CloseHandle(handle):
            failure = ctypes.WinError(ctypes.get_last_error())
            if primary is None:
                raise failure
            if hasattr(primary, "add_note"):
                primary.add_note(f"handle close failed: {failure}")
            else:
                primary.cleanup_errors = (str(failure),)

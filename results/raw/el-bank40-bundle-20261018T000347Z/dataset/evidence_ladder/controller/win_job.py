"""[MXC replacement] Windows Job Object helper: host-side resource limits for the MXC bridge process tree (the MXC V1 processcontainer policy has no memory / CPU / process-count fields).
Limits: JOB_OBJECT_LIMIT_JOB_MEMORY (whole-job committed memory), ACTIVE_PROCESS (processes alive in the job), CPU rate hard cap, KILL_ON_JOB_CLOSE (closing the job kills every member).
Whether a process started INSIDE the MXC container is a member of this job is verified by the tests (tests/phaseMXC_tests.py: memory-bomb and fork-bomb); if it were not, the limits would be
reported as unenforced there. Pure ctypes, Windows only."""
from __future__ import annotations

import ctypes
import os
from ctypes import wintypes

JobObjectBasicLimitInformation = 2
JobObjectExtendedLimitInformation = 9
JobObjectCpuRateControlInformation = 15
LIMIT_ACTIVE_PROCESS = 0x8
LIMIT_JOB_MEMORY = 0x200
LIMIT_KILL_ON_JOB_CLOSE = 0x2000
CPU_ENABLE, CPU_HARD_CAP = 0x1, 0x4


class IO_COUNTERS(ctypes.Structure):
    _fields_ = [(n, ctypes.c_ulonglong) for n in ("ReadOperationCount", "WriteOperationCount", "OtherOperationCount", "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]


class BASIC(ctypes.Structure):
    _fields_ = [("PerProcessUserTimeLimit", ctypes.c_longlong), ("PerJobUserTimeLimit", ctypes.c_longlong), ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD), ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD), ("SchedulingClass", wintypes.DWORD)]


class EXTENDED(ctypes.Structure):
    _fields_ = [("BasicLimitInformation", BASIC), ("IoInfo", IO_COUNTERS), ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]


class CPURATE(ctypes.Structure):
    _fields_ = [("ControlFlags", wintypes.DWORD), ("CpuRate", wintypes.DWORD)]


class Job:
    def __init__(self, memory_mb=None, max_processes=None, cpu_percent=None):
        if os.name != "nt":
            raise OSError("Job objects are Windows-only")
        k = ctypes.WinDLL("kernel32", use_last_error=True)
        k.CreateJobObjectW.restype = wintypes.HANDLE
        k.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        k.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
        self._k = k
        self.handle = k.CreateJobObjectW(None, None)
        if not self.handle:
            raise ctypes.WinError(ctypes.get_last_error())
        info = EXTENDED()
        flags = LIMIT_KILL_ON_JOB_CLOSE
        if memory_mb:
            flags |= LIMIT_JOB_MEMORY
            info.JobMemoryLimit = int(memory_mb) * 1024 * 1024
        if max_processes:
            flags |= LIMIT_ACTIVE_PROCESS
            info.BasicLimitInformation.ActiveProcessLimit = int(max_processes)
        info.BasicLimitInformation.LimitFlags = flags
        if not k.SetInformationJobObject(self.handle, JobObjectExtendedLimitInformation, ctypes.byref(info), ctypes.sizeof(info)):
            raise ctypes.WinError(ctypes.get_last_error())
        if cpu_percent:
            cr = CPURATE(CPU_ENABLE | CPU_HARD_CAP, max(1, min(10000, int(cpu_percent * 100 / (os.cpu_count() or 1)))))
            if not k.SetInformationJobObject(self.handle, JobObjectCpuRateControlInformation, ctypes.byref(cr), ctypes.sizeof(cr)):
                raise ctypes.WinError(ctypes.get_last_error())
        self.limits = {"memory_mb": memory_mb, "max_processes": max_processes, "cpu_percent": cpu_percent}

    def assign(self, popen) -> None:
        if not self._k.AssignProcessToJobObject(self.handle, wintypes.HANDLE(int(popen._handle))):
            raise ctypes.WinError(ctypes.get_last_error())

    def terminate(self) -> None:
        """Kill every process currently in the job (the bridge and anything it started); the job stays usable for the next bridge."""
        if self.handle:
            self._k.TerminateJobObject(self.handle, 1)

    def close(self) -> None:
        if self.handle:
            self._k.CloseHandle(self.handle)
            self.handle = None

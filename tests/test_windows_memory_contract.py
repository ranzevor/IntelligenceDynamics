import ctypes
import unittest
from unittest.mock import Mock, patch

from intelligence_dynamics.runtime.resources import _windows_resident_bytes


class WindowsMemoryContractTests(unittest.TestCase):
    def libraries(self):
        kernel, psapi = Mock(), Mock()
        kernel.OpenProcess.return_value = 123
        kernel.CloseHandle.return_value = True

        def info(handle, pointer, size):
            pointer._obj.WorkingSetSize = 2048
            return True

        psapi.GetProcessMemoryInfo.side_effect = info
        return kernel, psapi

    def test_read_and_handle_release(self):
        kernel, psapi = self.libraries()
        with patch.object(ctypes, "WinDLL", side_effect=[kernel, psapi], create=True):
            self.assertEqual(_windows_resident_bytes(7), 2048)
        kernel.CloseHandle.assert_called_once_with(123)

    def test_access_denial_is_not_missing_observation(self):
        kernel, psapi = self.libraries()
        kernel.OpenProcess.return_value = 0
        with (
            patch.object(ctypes, "WinDLL", side_effect=[kernel, psapi], create=True),
            patch.object(ctypes, "get_last_error", return_value=5, create=True),
            patch.object(
                ctypes,
                "WinError",
                side_effect=lambda code: PermissionError(code, "synthetic denied"),
                create=True,
            ),
        ):
            with self.assertRaises(PermissionError):
                _windows_resident_bytes(7)
        self.assertFalse(kernel.CloseHandle.called)

    def test_exited_pid_and_failed_query(self):
        kernel, psapi = self.libraries()
        kernel.OpenProcess.return_value = 0
        with (
            patch.object(ctypes, "WinDLL", side_effect=[kernel, psapi], create=True),
            patch.object(ctypes, "get_last_error", return_value=87, create=True),
        ):
            self.assertIsNone(_windows_resident_bytes(7))
        kernel, psapi = self.libraries()
        psapi.GetProcessMemoryInfo.side_effect = None
        psapi.GetProcessMemoryInfo.return_value = False
        with (
            patch.object(ctypes, "WinDLL", side_effect=[kernel, psapi], create=True),
            patch.object(ctypes, "get_last_error", return_value=6, create=True),
            patch.object(
                ctypes,
                "WinError",
                side_effect=lambda code: OSError(code, "synthetic query failure"),
                create=True,
            ),
        ):
            with self.assertRaises(OSError):
                _windows_resident_bytes(7)
        kernel.CloseHandle.assert_called_once_with(123)


class WindowsHandleCloseTests(unittest.TestCase):
    def test_close_failure_is_visible(self):
        kernel, psapi = WindowsMemoryContractTests().libraries()
        kernel.CloseHandle.return_value = False
        with (
            patch.object(ctypes, "WinDLL", side_effect=[kernel, psapi], create=True),
            patch.object(ctypes, "get_last_error", return_value=6, create=True),
            patch.object(
                ctypes,
                "WinError",
                side_effect=lambda code: OSError(code, "synthetic close failure"),
                create=True,
            ),
        ):
            with self.assertRaisesRegex(OSError, "synthetic close failure"):
                _windows_resident_bytes(7)

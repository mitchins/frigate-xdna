"""Unit tests: passive host fingerprint (docs/COMPATIBILITY.md §3).

The fingerprint must be buildable entirely from a fake sysfs tree,
must survive total absence with honest nulls, must never read DMI
serial/UUID fields, and must report the pinned appliance runtime
(not host XRT). The embedded version constants are pinned against
the vendor lockfile and Dockerfile sonames so a payload bump
cannot silently desync the report.
"""
import json
import os
import unittest

from frigate_xdna.observability import host_info
from frigate_xdna.runtime_versions import (
    EMBEDDED_FLEXMLRT,
    EMBEDDED_XDNA_SHIM,
    EMBEDDED_XRT,
    embedded_runtime,
)

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))


def w(root: str, rel: str, data: str) -> None:
    path = os.path.join(root, rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(data)


class FakeSysfsTest(unittest.TestCase):
    def setUp(self):
        import tempfile
        self.root = tempfile.mkdtemp(prefix="fx-hi-")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.root, ignore_errors=True)

    def _npu(self) -> str:
        return os.path.join(self.root, "class/accel/accel0/device")

    def build_strix_halo(self) -> None:
        dev = self._npu()
        w(self.root, "class/accel/accel0/device/vendor", "0x1022\n")
        w(self.root, "class/accel/accel0/device/device", "0x17f0\n")
        w(self.root, "class/accel/accel0/device/revision", "0x11\n")
        w(self.root, dev + "/subsystem_vendor", "0xf111\n")
        w(self.root, dev + "/subsystem_device", "0x000a\n")
        w(self.root, dev + "/fw_version", "1.1.2.65\n")
        os.symlink("../../../../module/amdxdna", os.path.join(
            dev, "driver"))
        w(self.root, "module/amdxdna/srcversion",
          "4612EC552523E4C8FB4B5E5\n")
        w(self.root, dev + "/power/control", "auto\n")
        w(self.root, dev + "/power/runtime_status", "suspended\n")
        w(self.root, dev + "/power/autosuspend_delay_ms", "5000\n")
        for name, val in (("product_name", "Desktop\n"),
                          ("bios_vendor", "Framework\n"),
                          ("bios_version", "03.05\n"),
                          ("bios_date", "01/21/2026\n")):
            w(self.root, f"class/dmi/id/{name}", val)

    def test_full_tree_parsed(self):
        self.build_strix_halo()
        info = host_info.collect_host_info(sys_root=self.root)
        npu = info["npu"]
        self.assertEqual(npu["pci_id"], "1022:17f0")
        self.assertEqual(npu["revision"], "0x11")
        self.assertEqual(npu["subsystem_id"], "f111:000a")
        self.assertEqual(npu["driver"], "amdxdna")
        self.assertEqual(npu["driver_srcversion"],
                         "4612EC552523E4C8FB4B5E5")
        self.assertEqual(npu["firmware_version"], "1.1.2.65")
        self.assertEqual(info["power"]["autosuspend_delay_ms"], 5000)
        self.assertEqual(info["power"]["control"], "auto")
        self.assertEqual(info["host"]["dmi"]["bios_version"], "03.05")
        self.assertIn("Linux", info["host"]["uname"])

    def test_uname_omits_hostname(self):
        # Shareable artifact: the host's nodename must not leak.
        info = host_info.collect_host_info(sys_root=self.root)
        import socket
        import os as _os
        hostname = socket.gethostname()
        self.assertNotIn(hostname, info["host"]["uname"])
        u = _os.uname()
        self.assertTrue(info["host"]["uname"].startswith(u.sysname))
        self.assertIn(u.release, info["host"]["uname"])
        self.assertIn(u.machine, info["host"]["uname"])

    def test_absent_tree_is_nulls_not_crash(self):
        info = host_info.collect_host_info(sys_root=self.root)
        self.assertIsNone(info["npu"]["pci_id"])
        self.assertIsNone(info["npu"]["driver"])
        self.assertIsNone(info["power"]["autosuspend_delay_ms"])
        self.assertIsNone(info["host"]["dmi"]["bios_version"])

    def test_dmi_serial_fields_never_read(self):
        self.build_strix_halo()
        # Bait fields: if the collector ever reads them, the report
        # would leak hardware identifiers into public bundles.
        for name in host_info._DMI_FORBIDDEN:
            w(self.root, f"class/dmi/id/{name}", "SECRET\n")
        info = host_info.collect_host_info(sys_root=self.root)
        blob = json.dumps(info)
        self.assertNotIn("SECRET", blob)
        for name in host_info._DMI_FORBIDDEN:
            self.assertNotIn(name, blob)

    def test_format_renders_key_fields(self):
        self.build_strix_halo()
        text = host_info.format_host_info(
            host_info.collect_host_info(sys_root=self.root))
        for needle in ("1022:17f0", "0x11", "4612EC55", "1.1.2.65",
                       "XDNA shim", EMBEDDED_XRT, EMBEDDED_FLEXMLRT):
            self.assertIn(needle, text)


class EmbeddedVersionsTest(unittest.TestCase):
    def test_pinned_against_vendor_lockfile(self):
        lock = json.load(open(os.path.join(
            REPO, "packaging", "vendor.lock.json"), encoding="utf-8"))
        versions = {c["package"]: c["version"] for c in lock["components"]}
        self.assertEqual(versions["xrt-base"], EMBEDDED_XRT)
        self.assertEqual(versions["xdna-driver-plugin"],
                         EMBEDDED_XDNA_SHIM)
        self.assertEqual(versions["flexmlrt"], EMBEDDED_FLEXMLRT)

    def test_pinned_against_dockerfile_sonames(self):
        with open(os.path.join(REPO, "packaging", "Dockerfile"),
                  encoding="utf-8") as f:
            dockerfile = f.read()
        self.assertIn(f"libxrt_core.so.{EMBEDDED_XRT}", dockerfile)
        self.assertIn(f"libxrt_driver_xdna.so.{EMBEDDED_XDNA_SHIM}",
                      dockerfile)

    def test_shape(self):
        rt = embedded_runtime()
        self.assertEqual(rt["schema_version"], 1)
        self.assertEqual(rt["xrt"], EMBEDDED_XRT)
        self.assertEqual(rt["xdna_shim"], EMBEDDED_XDNA_SHIM)
        self.assertEqual(rt["flexmlrt"], EMBEDDED_FLEXMLRT)


if __name__ == "__main__":
    unittest.main()

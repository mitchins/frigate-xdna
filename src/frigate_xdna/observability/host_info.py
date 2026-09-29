"""Passive host fingerprint (docs/COMPATIBILITY.md).

Reads sysfs/procfs and the baked build identity only: no device
open, no NPU context creation, no `xrt-smi`, no reset, no writes.
The container's `/etc/os-release` is never reported as the host
distribution. Serial numbers, MAC/IP addresses, Plus keys and raw
model IDs are out of scope by construction — nothing in the source
set contains them.
"""
from __future__ import annotations

import os
import time

SYS_ROOT = "/sys"
DMI_ROOT = "/sys/class/dmi/id"
ACCEL_CLASS = "class/accel"

# DMI fields that must never be read, let alone reported.
_DMI_FORBIDDEN = ("product_serial", "product_uuid", "board_serial",
                  "chassis_serial", "product_sku")
_DMI_FIELDS = ("product_name", "board_name", "bios_vendor",
               "bios_version", "bios_date")


def _read(path: str) -> str | None:
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return f.read().strip()
    except (OSError, ValueError):
        return None


def _read_int(path: str) -> int | None:
    raw = _read(path)
    if raw is None:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _npu_device(sys_root: str) -> str | None:
    """First accel device directory (this build targets one NPU)."""
    base = os.path.join(sys_root, ACCEL_CLASS)
    try:
        entries = sorted(os.listdir(base))
    except OSError:
        return None
    for name in entries:
        dev = os.path.join(base, name, "device")
        if os.path.isdir(dev):
            return dev
    return None


def _driver_name(dev: str) -> str | None:
    try:
        link = os.readlink(os.path.join(dev, "driver"))
    except OSError:
        return None
    return os.path.basename(link.rstrip("/"))


def _collect_npu(sys_root: str) -> tuple[dict, dict]:
    """(npu, power) records for the first accel device; honest
    nulls when nothing is visible."""
    npu: dict = {}
    power: dict = {}
    dev = _npu_device(sys_root)
    if dev is not None:
        vendor = _read(os.path.join(dev, "vendor"))
        device = _read(os.path.join(dev, "device"))
        npu["pci_id"] = (f"{vendor}:{device}"
                         if vendor and device else None)
        npu["revision"] = _read(os.path.join(dev, "revision"))
        sub_vendor = _read(os.path.join(dev, "subsystem_vendor"))
        sub_device = _read(os.path.join(dev, "subsystem_device"))
        npu["subsystem_id"] = (f"{sub_vendor}:{sub_device}"
                               if sub_vendor and sub_device else None)
        for key in ("pci_id", "subsystem_id"):
            if npu[key]:
                npu[key] = npu[key].replace("0x", "")
        driver = _driver_name(dev)
        npu["driver"] = driver
        npu["driver_srcversion"] = (
            _read(os.path.join(sys_root, "module", driver, "srcversion"))
            if driver else None)
        npu["driver_version"] = (
            _read(os.path.join(sys_root, "module", driver, "version"))
            if driver else None)
        # Standalone fw_version sysfs attribute (amdxdna).
        npu["firmware_version"] = _read(os.path.join(dev, "fw_version"))
        power_root = os.path.join(dev, "power")
        power["control"] = _read(os.path.join(power_root, "control"))
        power["runtime_status"] = _read(
            os.path.join(power_root, "runtime_status"))
        power["autosuspend_delay_ms"] = _read_int(
            os.path.join(power_root, "autosuspend_delay_ms"))
    for key in ("pci_id", "revision", "subsystem_id", "driver",
                "driver_srcversion", "driver_version",
                "firmware_version"):
        npu.setdefault(key, None)
    for key in ("control", "runtime_status", "autosuspend_delay_ms"):
        power.setdefault(key, None)
    return npu, power


def collect_host_info(sys_root: str = SYS_ROOT) -> dict:
    """Passive fingerprint: host, NPU, runtime PM, appliance stack.

    Absent readings are `None` (honest absence), never invented.
    """
    npu, power = _collect_npu(sys_root)
    dmi_root = os.path.join(sys_root, "class/dmi/id")
    dmi = {f: _read(os.path.join(dmi_root, f)) for f in _DMI_FIELDS}

    from ..build_identity import get_build_identity
    from ..runtime_versions import embedded_runtime
    ident = get_build_identity()
    return {
        "schema_version": 1,
        "collected_at": time.time(),
        "host": {
            "uname": " ".join(os.uname()),
            "dmi": dmi,
        },
        "npu": npu,
        "power": power,
        "appliance": {
            "frigate_xdna": ident["version"],
            "revision": ident["revision"],
            "channel": ident["channel"],
            "embedded": embedded_runtime(),
        },
        "note": "passive sysfs/procfs read; no NPU context, no "
                "xrt-smi, no device reset; appliance runtime is "
                "embedded, not host XRT; container os-release is "
                "not host identity",
    }


def format_host_info(info: dict) -> str:
    """Human-readable rendering for `fxdna host-info`."""
    npu, power, app = info["npu"], info["power"], info["appliance"]
    emb = app["embedded"]

    def line(label: str, value) -> str:
        return f"  {label + ':':22} {'' if value is None else value}"

    out = [
        "Host",
        line("uname", info["host"]["uname"]),
        line("product", info["host"]["dmi"].get("product_name")),
        line("BIOS", _join_nonempty(
            info["host"]["dmi"].get("bios_vendor"),
            info["host"]["dmi"].get("bios_version"),
            info["host"]["dmi"].get("bios_date"), sep=" ")),
        "",
        "NPU",
        line("PCI ID", npu.get("pci_id")),
        line("revision", npu.get("revision")),
        line("subsystem", npu.get("subsystem_id")),
        line("driver", npu.get("driver")),
        line("srcversion", npu.get("driver_srcversion")),
        line("driver version", npu.get("driver_version")),
        line("firmware", npu.get("firmware_version")),
        "",
        "Power (runtime PM)",
        line("control", power.get("control")),
        line("runtime_status", power.get("runtime_status")),
        line("autosuspend_ms", power.get("autosuspend_delay_ms")),
        "",
        "Appliance (embedded, not host XRT)",
        line("frigate-xdna", app.get("frigate_xdna")),
        line("revision", app.get("revision")),
        line("channel", app.get("channel")),
        line("XRT", emb.get("xrt")),
        line("XDNA shim", emb.get("xdna_shim")),
        line("FlexMLRT", emb.get("flexmlrt")),
        line("recipe", emb.get("recipe")),
    ]
    return "\n".join(out) + "\n"


def _join_nonempty(*parts, sep: str) -> str | None:
    got = [p for p in parts if p]
    return sep.join(got) if got else None

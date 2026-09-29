#!/usr/bin/env bash
# Host-side evidence collection after a suspected XDNA-related host
# reset (docs/COMPATIBILITY.md). Run on the HOST after reboot:
#
#   sudo ./collect-host-evidence.sh > host-evidence.txt
#
# Read-only. Collects only platform/NPU/reset evidence; never IPs,
# MACs, disk serials, camera identifiers or credentials. Every
# section fails soft: missing data prints "not available" and the
# script continues.
set -u

redact() {
  # IPv6: any hex-and-colon token containing '::' (compressed
  # form), plus the full 7-colon form. Timestamps (19:45:57) have
  # no '::' and too few colons; MACs are redacted first.
  sed -E \
    -e 's/([0-9A-Za-z]{2}:){5}[0-9A-Za-z]{2}/<mac>/g' \
    -e 's/\b([0-9]{1,3}\.){3}[0-9]{1,3}\b/<ip>/g' \
    -e 's/[0-9A-Fa-f:]*::[0-9A-Fa-f:]*/<ip6>/g' \
    -e 's/([0-9A-Fa-f]{1,4}:){7}[0-9A-Fa-f]{1,4}/<ip6>/g' \
    -e 's/(authorization|api[_-]?key|token|secret|password|passwd|passphrase|credential|private[_-]?key)[=: ]+[^ ]+/\1=<redacted>/Ig'
  return 0
}

section() {
  local title="$1"
  printf '\n===== %s =====\n' "$title"
  return 0
}

soft() {
  local cmd="$1"
  shift
  if command -v "$cmd" >/dev/null 2>&1; then
    "$cmd" "$@"
  else
    echo "(command $cmd not available)"
  fi
  return 0
}

section "collected at"
date -u +"%Y-%m-%dT%H:%M:%SZ (UTC)"

section "uname"
uname -a 2>/dev/null || echo "(uname failed)"

section "BIOS / DMI (no serials)"
for f in product_name board_name bios_vendor bios_version bios_date; do
  p="/sys/class/dmi/id/$f"
  if [[ -r "$p" ]]; then printf '%s: %s\n' "$f" "$(cat "$p")"; fi
done
[[ -r /sys/class/dmi/id/product_name ]] || echo "(DMI not readable)"

section "NPU PCI / revision"
for dev in /sys/class/accel/*/device; do
  [[ -d "$dev" ]] || continue
  printf '%s vendor=%s device=%s revision=%s subsystem=%s:%s\n' \
    "$dev" \
    "$(cat "$dev/vendor" 2>/dev/null)" \
    "$(cat "$dev/device" 2>/dev/null)" \
    "$(cat "$dev/revision" 2>/dev/null)" \
    "$(cat "$dev/subsystem_vendor" 2>/dev/null)" \
    "$(cat "$dev/subsystem_device" 2>/dev/null)"
done
[[ -d /sys/class/accel ]] || echo "(no accel class devices)"

section "amdxdna driver metadata"
if [[ -d /sys/module/amdxdna ]]; then
  printf 'srcversion: %s\n' "$(cat /sys/module/amdxdna/srcversion 2>/dev/null || echo not-exposed)"
  printf 'version: %s\n' "$(cat /sys/module/amdxdna/version 2>/dev/null || echo not-exposed)"
  printf 'parameters:\n'
  for p in /sys/module/amdxdna/parameters/*; do
    [[ -e "$p" ]] || continue
    printf '  %s = %s\n' "$(basename "$p")" "$(cat "$p" 2>/dev/null)"
  done
else
  echo "(amdxdna module not loaded)"
fi

section "NPU firmware"
for dev in /sys/class/accel/*/device; do
  [[ -r "$dev/fw_version" ]] && printf '%s fw_version=%s\n' "$dev" "$(cat "$dev/fw_version")"
done

section "runtime PM configuration"
for dev in /sys/class/accel/*/device; do
  [[ -d "$dev" ]] || continue
  printf '%s control=%s runtime_status=%s autosuspend_delay_ms=%s\n' \
    "$dev" \
    "$(cat "$dev/power/control" 2>/dev/null)" \
    "$(cat "$dev/power/runtime_status" 2>/dev/null)" \
    "$(cat "$dev/power/autosuspend_delay_ms" 2>/dev/null)"
done

section "boot list (recent)"
soft journalctl --no-pager --list-boots 2>/dev/null | tail -5 | redact

section "previous boot: reset reason / error lines"
# Filtered kernel lines from the previous boot: reset cause, fabric,
# AER/RAS/MCE, XDNA/AMDGPU/SMU/PSP/IOMMU. Bounded tail, redacted.
if command -v journalctl >/dev/null 2>&1; then
  journalctl -b -1 --no-pager -k 2>/dev/null \
    | grep -Ei 'reset|sync flood|data fabric|aer|ras|mce|machine check|amdxdna|npu|amdgpu|smu|psp|iommu|hard lockup|soft lockup|panic|oops' \
    | tail -n 300 | redact
  echo "(previous-boot kernel lines end)"
  section "previous boot: shutdown/reboot lines (platform only)"
  # Unfiltered any-facility tails can carry arbitrary application
  # logs; keep only platform lifecycle lines near the loss.
  journalctl -b -1 --no-pager -n 300 2>/dev/null \
    | grep -Ei 'systemd-shutdown|shutdown|reboot|powering|restart|watchdog' \
    | tail -n 20 | redact
else
  echo "(journalctl not available; no persistent journal?)"
fi

section "current boot: early amdxdna/npu lines"
if command -v journalctl >/dev/null 2>&1; then
  journalctl -b 0 --no-pager -k 2>/dev/null \
    | grep -Ei 'amdxdna|npu' | head -n 80 | redact
fi

printf '\n===== end =====\n'

# Field test: v0.1.3-rc.1 stability diagnostics (external host)

A conservative, one-machine procedure for the externally reported
Framework Desktop / Ryzen AI Max+ 395 running Ubuntu 26.04.1,
kernel `7.0.0-34-generic`, where the host has reset under
`0x08000800` (data fabric sync flood) — see O-003 in
[COMPATIBILITY.md](COMPATIBILITY.md).

This is an **observation**, not a fix, a stress test or a support
commitment. The goal is durable evidence of how far the production
path gets before anything fails.

## Exact image

```text
ghcr.io/mitchins/frigate-xdna@sha256:af7c1902e822c29c45213d3f8a2b1aa7f84ad1fedb0da66445df3f7c0094bdf8
  (v0.1.3-rc.1, source ec8e426d0693d9f1b33c5dc605d189ad2238ec94)
```

Pin the image by digest, never by tag, for this test.

## Before you start — change nothing

Do **not** change any of the following. We want the existing stack
exactly as it is:

- BIOS
- kernel
- NPU firmware
- runtime PM configuration (`power/control`, autosuspend)
- GPU drivers/configuration

## Step A — preserve the existing evidence (passive only)

No NPU use in this step.

```bash
# Appliance fingerprint (passive):
docker run --rm --user 10001:10001 --group-add "$(stat -c %g /dev/accel/accel0)" \
  --device /dev/accel/accel0 --ulimit memlock=-1:-1 \
  -v /path/to/xdna-data:/data -e FXDNA_DATA_DIR=/data \
  ghcr.io/mitchins/frigate-xdna@sha256:af7c1902e822c29c45213d3f8a2b1aa7f84ad1fedb0da66445df3f7c0094bdf8 \
  host-info --json > host-info.json

# Host-side baseline (run on the host, from the repo checkout):
sudo tools/collect-host-evidence.sh > host-evidence-baseline.txt
```

Keep both files.

## Step B — smoke only

Use the exact RC appliance in diagnostic mode with your already
prepared/configured model (the compose command override; nothing
else changes):

```yaml
services:
  xdna:
    image: ghcr.io/mitchins/frigate-xdna@sha256:af7c1902e822c29c45213d3f8a2b1aa7f84ad1fedb0da66445df3f7c0094bdf8
    command:
      - stability
      - run
      - --configured
      - --profile
      - smoke
    # everything else identical to your normal xdna service
```

If you run the container directly instead of compose, pass the same
words as the command after the image reference:

```bash
... <image>@sha256:af7c1902... stability run --configured --profile smoke
```

Smoke takes a few minutes: it activates the production worker, runs
one inference, then 5 warmups + 25 measured inferences.

**If smoke FAILS or the host resets: STOP.**

Do not rerun. After the machine comes back:

```bash
# Passive/reporting only — these never open the NPU:
<image> stability report --last > stability-report.txt
<image> stability report --last --json > stability.json
<image> diagnose --out diagnose
sudo tools/collect-host-evidence.sh > host-evidence-after-reset.txt
```

Preserve the run directory under `<data>/stability/` exactly as it
is — do not delete or edit anything in it.

## Step C — gentle (only if smoke passed)

Same command, `gentle` instead of `smoke`. Gentle adds a 120-second
low-steady phase at 10% of your model's measured capacity.

**If gentle FAILS or the host resets: STOP.**

Do not retry until the last run has been inspected and, where the
journal shows an interrupted run, explicitly acknowledged:

```bash
<image> stability report --last
# review it, then:
<image> stability acknowledge --last --reason "reviewed reset and collected host logs"
```

Acknowledgement clears only the diagnostic latch; it never touches
production safety state.

## Step D — if gentle passes: stop there

A smoke + gentle pass is useful evidence by itself. Do **not** run
`pm`, `saturation`, `extended` or `coexistence` profiles, and do not
start any synthetic GPU stress alongside. We decide the next
diagnostic after reviewing your artifacts.

## What to send back

- `host-info.json` (Step A)
- `stability report --last --json` output (after each run)
- the `diagnose` bundle directory
- after a host reset, additionally:
  - `host-evidence-after-reset.txt` (previous-boot kernel evidence)
  - the reported system reset reason from your host

## Reading the output after a reset — what it does and does not say

The durable journal identifies the **last known position**, not
causality.

Correct:

> "The last durable checkpoint was H3 LOW_STEADY, with N completed
> requests approximately ≤1 second before host/process
> disappearance."

Incorrect:

> "H3 caused the reset."

No causal conclusion is drawn from the journal alone.

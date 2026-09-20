# Operator workflow

These commands describe the implemented product interface (`fxdna` CLI +
appliance image, Tasks 01–04). Hardware acceptance (Task 05) may still
extend them.

## First launch

On the Docker host, confirm a supported accelerator node exists. Container images cannot supply the host kernel/firmware. Obtain its numeric group using `stat -c '%g' /dev/accel/accel0`; put that value in the Compose environment as `NPU_GID`.

Copy the example Compose and `.env.example`, choose an actual built/released image and set selected model references. For Plus, provide the API key in `secrets/PLUS_API_KEY`; keep the file private. Frigate itself retains its normal Plus credential too; this does not require sharing Frigate's entire config directory with the sidecar.

```
docker compose up -d xdna
docker compose exec -T xdna fxdna status
docker compose exec -T xdna fxdna wait plus://MODEL_A --timeout 1800
```

`wait` defaults to PREPARED. It is not a claim the model has already run on hardware. Model activation and its bounded native checks occur before the successful Frigate handshake.

Join Frigate to the same Docker network and use `tcp://xdna:5555`, or publish the sidecar port only on a trusted reachable address and use that address. Host-networked Frigate cannot resolve a Compose-only service name automatically; use the appropriate published endpoint. Keep raw ZMQ off the public Internet.

Do not point Frigate at an unprepared cold model and expect it to wait through compilation. Sidecar liveness/Compose health is not model readiness.

## Prepare an update without changing the active selection

```
docker compose exec -T xdna fxdna prepare plus://MODEL_B
docker compose exec -T xdna fxdna status plus://MODEL_B
docker compose exec -T xdna fxdna wait plus://MODEL_B --timeout 1800
```

The request persists; no daemon restart is required. Updating `FXDNA_MODELS` and restarting also works, but restarting an active sidecar has the normal worker interruption.

If the job is WAITING_FOR_DEVICE, the recipe needs device access while the current worker owns it. Do not kill the current worker blindly. Schedule explicit `prepare --maintenance` or prepare before starting inference. CPU-only compilation may overlap only after that property and resource impact are validated for the shipped recipe.

Once B is prepared, update Frigate's model ID and reinitialize through its normal mechanism. The old model must stop sending requests; the sidecar will drain it and activate B in a fresh process. Conflicting live clients get MODEL_IN_USE. For a deliberately coordinated forced transition, the local administrator can use `activate B --maintenance --wait`.

## Unexpected cold miss

The model can be saved and prepared, but stock rc2 initial model failure is not auto-polled. After `wait` reports prepared, restart/reinitialize Frigate's detector via normal configuration/service behaviour. Do not modify the plugin, lie about readiness, or stretch inference timeouts to minutes.

## Status, cache and offline operation

```
docker compose exec -T xdna fxdna status --json
docker compose exec -T xdna fxdna cache list
docker compose exec -T xdna fxdna cache prune
```

Prune is a dry run until `--apply`; configured/active/rollback/evidence pins stay protected. Back up `/data` with a coherent SQLite backup/snapshot procedure. Do not copy a live DB while omitting WAL or remove model directories by hand.

Network/Plus unavailability must not stop cached detection. New downloads fail visibly; existing verified artifacts remain usable. `FXDNA_OFFLINE=true` makes this mode explicit.

## Memory bounds

Compiler-child memory is bounded by the container/cgroup (`mem_limit:
8g` in `examples/compose.yaml`), never by `RLIMIT_AS`: an
address-space cap breaks large VA mappings (a 512 MiB device mapping
fails with ENOMEM) while RSS stays far below any real limit. Each
compile phase reports VmPeak alongside RSS in the artifact's
`compile_stats`. Native runs without a container are the operator's
responsibility (e.g. `systemd-run --scope -p MemoryMax=8G`).

## Device-memory (CMA) readings

CMA counters read inside a container/LXC guest are unreliable (a
`CmaTotal: 0 kB` view proves filtering). Record CMA only on the raw
Proxmox host at lifecycle boundaries:

```sh
grep -E 'Cma(Total|Free)' /proc/meminfo
```

before compile, after compiler exit, and after validation. A large
`mmap … ENOMEM` on a device mapping with healthy host CMA points at
process address-space pressure, not device exhaustion.

## Faults and suspected host reset

Read status/logs first. `doctor` is passive; do not run `doctor --hardware` while a worker owns the NPU. A native/device fault or interrupted sensitive operation may set a persistent safety inhibition.

Never resolve inhibition by deleting locks/state files. Preserve diagnostics and explicitly review the implicated model/operation before `recover`. A host reboot is not a signal to automatically retry a suspect model.

A client timeout during a host stall does not by itself imply a native failure. Inspect separately: successful NPU completions, client-delivered responses, deadline misses and worker generation. The service must not reload models just because a REQ socket timed out.

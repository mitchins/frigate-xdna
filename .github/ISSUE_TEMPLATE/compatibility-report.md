---
name: Compatibility observation
about: Report a working or failing host/platform stack for docs/COMPATIBILITY.md
labels: compatibility
blank_issues_enabled: false
body:
  - type: markdown
    attributes:
      value: |
        Reports here become **observations**, not support commitments.
        A PASS means "this workload completed on this stack", never
        "this stack is certified stable". See
        [docs/COMPATIBILITY.md](../blob/main/docs/COMPATIBILITY.md).
  - type: textarea
    id: hardware
    attributes:
      label: Hardware / BIOS / kernel
      description: One line, e.g. "Framework Desktop, Ryzen AI Max+ 395, BIOS 03.05 (PI 1.0.0.2), kernel 7.0.14-14-pve"
    validations:
      required: true
  - type: dropdown
    id: result
    attributes:
      label: Result class
      options:
        - PASS — completed the explicitly stated test
        - LIMITED — useful success, insufficient duration/scope
        - FAIL — recoverable test/runtime failure
        - RESET — host reset / hard loss
        - UNKNOWN — incomplete evidence
    validations:
      required: true
  - type: textarea
    id: workload
    attributes:
      label: Model and workload
      description: Model family/resolution, duration, inference count, concurrent GPU load if any
    validations:
      required: true
  - type: textarea
    id: artifacts
    attributes:
      label: Diagnostic artifacts
      description: |
        Run and attach (do not paste raw logs):
        ```
        fxdna host-info --json > host-info.json
        fxdna stability report --last --json > stability.json
        fxdna diagnose --out diagnose
        ```
        After a host reset, if possible:
        ```
        sudo tools/collect-host-evidence.sh > host-evidence.txt
        ```
  - type: textarea
    id: notes
    attributes:
      label: What happened / what you observed

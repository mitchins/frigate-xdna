"""Deployment-contract tests (v0.1.1 checkpoint 1).

Assert the shipped Compose files carry the mandatory appliance
settings (memlock ulimits, resource bounds, non-root hardening, no
published port by default, public release-image default) and that the
README install path agrees with those files.

Stdlib only: PyYAML is not a project dependency, so this module has
a minimal indentation-based parser covering exactly the constructs
used by examples/compose*.yaml (nested maps, dash lists, scalars,
comments). It is a test aid, not a general YAML implementation.
"""
from __future__ import annotations

import os
import re
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
EXAMPLES = os.path.join(REPO, "examples")
README = os.path.join(REPO, "README.md")


def parse_subset(text: str):
    """Parse a small YAML subset into nested dicts/lists.

    Supports `key: value`, `key:` + indented block, `- item` and
    `- key: value` list entries, `#` comments and quoted scalars.
    Anything else raises ValueError (fail loudly, not silently).
    """
    lines = []
    for raw in text.splitlines():
        stripped = raw.split("#", 1)[0].rstrip()
        if stripped.strip():
            lines.append((len(raw) - len(raw.lstrip(" ")), stripped.strip()))
    if any("\t" in raw for raw in text.splitlines()):
        raise ValueError("tabs are not valid indentation")
    pos = [0]

    def scalar(token: str):
        token = token.strip()
        if len(token) >= 2 and token[0] == token[-1] and token[0] in "\"'":
            return token[1:-1]
        if token in ("true", "false"):
            return token == "true"
        try:
            return int(token)
        except ValueError:
            return token

    def parse_block(indent: int):
        if pos[0] < len(lines) and lines[pos[0]][1].startswith("- "):
            return parse_list(indent)
        return parse_map(indent)

    def parse_map(indent: int):
        out = {}
        while pos[0] < len(lines):
            ind, content = lines[pos[0]]
            if ind < indent:
                break
            if ind > indent or content.startswith("- "):
                raise ValueError(f"unexpected line at {ind}: {content}")
            if ":" not in content:
                raise ValueError(f"not a mapping line: {content}")
            key, _, rest = content.partition(":")
            key, rest = key.strip(), rest.strip()
            pos[0] += 1
            if rest:
                out[key] = scalar(rest)
            else:
                if pos[0] < len(lines) and lines[pos[0]][0] > indent:
                    out[key] = parse_block(lines[pos[0]][0])
                else:
                    out[key] = None
        return out

    def parse_list(indent: int):
        out = []
        while pos[0] < len(lines):
            ind, content = lines[pos[0]]
            if ind != indent or not content.startswith("- "):
                break
            item = content[2:].strip()
            pos[0] += 1
            if not item:
                if pos[0] < len(lines) and lines[pos[0]][0] > indent:
                    out.append(parse_block(lines[pos[0]][0]))
                else:
                    out.append(None)
            elif ":" in item and not _looks_scalar(item):
                key, _, rest = item.partition(":")
                entry = {key.strip(): scalar(rest)}
                if pos[0] < len(lines) and lines[pos[0]][0] > indent:
                    if lines[pos[0]][1].startswith("- "):
                        raise ValueError("nested list in list item")
                    entry.update(parse_map(lines[pos[0]][0]))
                out.append(entry)
            else:
                out.append(scalar(item))
        return out

    doc = parse_map(0)
    if pos[0] != len(lines):
        raise ValueError("trailing unparsed lines")
    return doc


def _looks_scalar(item: str) -> bool:
    """A list item with a colon is still a scalar when no colon at
    interpolation depth 0 is followed by whitespace or end of string
    (YAML plain-scalar rule: `a:b` is scalar, `a: b` is a mapping)."""
    stripped = item.strip()
    if (len(stripped) >= 2 and stripped[0] in "\"'"
            and stripped[0] == stripped[-1]):
        return True
    depth = 0
    i = 0
    while i < len(item):
        if item[i] == "$" and item[i:i + 2] == "${":
            depth += 1
            i += 2
            continue
        if item[i] == "}" and depth:
            depth -= 1
        elif item[i] == ":" and depth == 0:
            after = item[i + 1:i + 2]
            if after in ("", " ", "\t"):
                return False
        i += 1
    return True


def load_example(name: str):
    with open(os.path.join(EXAMPLES, name), encoding="utf-8") as f:
        return parse_subset(f.read())


def read_example(name: str) -> str:
    with open(os.path.join(EXAMPLES, name), encoding="utf-8") as f:
        return f.read()


class TestCanonicalCompose(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.doc = load_example("compose.yaml")
        cls.xdna = cls.doc["services"]["xdna"]

    def test_memlock_unlimited(self):
        memlock = self.xdna["ulimits"]["memlock"]
        self.assertEqual(memlock["soft"], -1)
        self.assertEqual(memlock["hard"], -1)

    def test_resource_bounds(self):
        self.assertEqual(str(self.xdna["cpus"]), "4.0")
        self.assertEqual(self.xdna["mem_limit"], "8g")
        self.assertEqual(self.xdna["memswap_limit"], "8g")
        self.assertEqual(self.xdna["pids_limit"], 512)

    def test_non_root_hardening_kept(self):
        self.assertEqual(str(self.xdna["user"]), "10001:10001")
        self.assertTrue(self.xdna["read_only"])
        self.assertIn("ALL", self.xdna["cap_drop"])
        self.assertIn("no-new-privileges:true", self.xdna["security_opt"])
        self.assertNotIn("privileged", self.xdna)
        self.assertNotIn("ports", self.xdna)

    def test_device_and_data(self):
        self.assertIn("/dev/accel/accel0:/dev/accel/accel0",
                      self.xdna["devices"])
        self.assertIn("xdna-data:/data", self.xdna["volumes"])
        self.assertEqual(
            self.xdna["environment"]["FXDNA_DATA_DIR"], "/data")

    def test_default_image_is_public_release(self):
        image = self.xdna["image"]
        self.assertTrue(
            image.startswith("${FXDNA_IMAGE:-ghcr.io/mitchins/"),
            f"default image must be the public release image: {image}")
        self.assertNotIn("0.1-dev", image)

    def test_required_settings_fail_fast(self):
        env = self.xdna["environment"]
        self.assertIn(":?", env["FXDNA_MODELS"])
        group_add = self.xdna["group_add"]
        self.assertTrue(any(":?" in str(g) for g in group_add),
                        "NPU_GID must be required")

    def test_no_docker_socket_or_broad_caps(self):
        text = read_example("compose.yaml")
        self.assertNotIn("docker.sock", text)
        self.assertNotIn("NET_ADMIN", text)
        self.assertNotIn("SYS_ADMIN", text)


class TestNetworkingVariants(unittest.TestCase):
    def test_host_port_overlay_is_loopback_by_default(self):
        doc = load_example("compose.host-port.yaml")
        ports = doc["services"]["xdna"]["ports"]
        self.assertEqual(len(ports), 1)
        self.assertIn("${FXDNA_BIND_IP:-127.0.0.1}:5555:5555", ports)

    def test_host_port_overlay_warns_about_name_resolution(self):
        text = read_example("compose.host-port.yaml")
        self.assertIn("127.0.0.1", text)
        self.assertIn("network_mode", text)  # host-mode caveat documented

    def test_plus_overlay_passes_key_through_environment(self):
        doc = load_example("compose.plus.yaml")
        env = doc["services"]["xdna"]["environment"]
        self.assertIn("PLUS_API_KEY", env)
        self.assertIn(":?", str(env["PLUS_API_KEY"]))
        self.assertNotIn("secrets", doc["services"]["xdna"])
        text = read_example("compose.plus.yaml")
        self.assertNotIn("PLUS_API_KEY_FILE", text)


class TestLocalOverlay(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.doc = load_example("compose.local.yaml")
        cls.xdna = cls.doc["services"]["xdna"]

    def test_model_dir_default_and_ro_mount(self):
        env = self.xdna["environment"]
        self.assertEqual(env["FXDNA_MODEL_DIR"], "/models")
        volumes = self.xdna["volumes"]
        self.assertEqual(len(volumes), 1)
        bind = str(volumes[0])
        self.assertIn("FXDNA_MODEL_HOST_DIR", bind)
        self.assertIn(":?", bind)
        self.assertTrue(bind.endswith(":/models:ro"),
                        f"model mount must be read-only: {bind}")

    def test_overlay_adds_nothing_else(self):
        self.assertEqual(set(self.xdna), {"environment", "volumes"})
        text = read_example("compose.local.yaml")
        for banned in ("PLUS_API_KEY", "ports:", "privileged",
                       "docker.sock", "NET_ADMIN", "SYS_ADMIN",
                       "secrets:"):
            self.assertNotIn(banned, text)

    def test_merged_render_needs_no_plus_credential(self):
        base = load_example("compose.yaml")
        merged_env = dict(base["services"]["xdna"]["environment"])
        merged_env.update(self.xdna["environment"])
        merged_volumes = list(base["services"]["xdna"]["volumes"])
        merged_volumes += self.xdna["volumes"]
        self.assertEqual(merged_env["FXDNA_MODEL_DIR"], "/models")
        self.assertIn(":?", merged_env["FXDNA_MODELS"])
        self.assertIn("xdna-data:/data", merged_volumes)
        self.assertTrue(any(str(v).endswith(":/models:ro")
                            for v in merged_volumes))
        merged_text = (read_example("compose.yaml")
                       + read_example("compose.local.yaml"))
        self.assertNotIn("PLUS_API_KEY", merged_text)

    def test_env_example_documents_host_dir(self):
        with open(os.path.join(EXAMPLES, ".env.example"),
                  encoding="utf-8") as f:
            text = f.read()
        self.assertIn("FXDNA_MODEL_HOST_DIR", text)
        self.assertIn("local://yolov9-t-320", text)

    def test_portainer_table_covers_host_dir(self):
        with open(README, encoding="utf-8") as f:
            text = f.read()
        self.assertIn("FXDNA_MODEL_HOST_DIR", text)


def readme_portainer_block(text: str):
    """The fenced YAML merged stack shown for Portainer users."""
    marker = "Local stack (`examples/compose.yaml`"
    start = text.index(marker)
    fence = text.index("```yaml", start)
    end = text.index("```", fence + 7)
    return text[fence + len("```yaml"):end]


class TestLocalReadme(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(README, encoding="utf-8") as f:
            cls.text = f.read()

    def test_both_users_on_first_screen(self):
        self.assertIn(
            "Use Frigate+ or compatible local YOLO ONNX models."
            " Local models\nrequire no account or API key.", self.text)

    def test_mount_table_names_all_four_places(self):
        for path in ("/srv/frigate/models/yolov9-t-320.onnx",
                     "/models/yolov9-t-320.onnx",
                     "local://yolov9-t-320",
                     "/config/models/yolov9-t-320.onnx"):
            self.assertIn(path, self.text)

    def test_local_frigate_block_is_frigate_contract(self):
        for marker in ("model_type: yolo-generic",
                       "input_tensor: nchw",
                       "input_dtype: float",
                       "labelmap_path: /labelmap/coco-80.txt",
                       "width: 320",
                       "height: 320"):
            self.assertIn(marker, self.text)
        self.assertIn("The\nONNX bytes must match.", self.text)
        self.assertIn("`labelmap_path` belongs to Frigate",
                      self.text)

    def test_export_routes_are_links_not_recipes(self):
        self.assertIn("https://docs.frigate.video/configuration/"
                      "object_detectors/", self.text)
        self.assertIn("https://docs.ultralytics.com/modes/export/",
                      self.text)

    def test_no_model_freedom_marketing(self):
        lowered = self.text.lower()
        for banned in ("fully supported", "model freedom",
                       "bring your own", "unlock the"):
            self.assertNotIn(banned, lowered)

    def test_compatibility_separates_graph_from_hardware(self):
        for tier in ("Graph contract", "XDNA inference",
                     "Frigate end-to-end", "Pending (C7)"):
            self.assertIn(tier, self.text)
        self.assertIn("is not hardware\nqualification", self.text)

    def test_lifecycle_is_startup_not_watching(self):
        self.assertIn("picked up at the next container start",
                      self.text)
        self.assertIn("prepare --refresh", self.text)
        self.assertIn("The directory is not\nwatched", self.text)
        self.assertIn("never\nswitches until Frigate binds",
                      self.text)

    def test_local_path_names_its_release(self):
        self.assertIn("need frigate-xdna 0.1.2 or newer", self.text)
        self.assertIn("the 0.1.1 image does not\nunderstand them",
                      self.text)

    def test_sequential_shell_uses_exports_and_repeat_f(self):
        block = self.text[self.text.index("### Path B"):
                          self.text.index("### Portainer")]
        self.assertIn("export NPU_GID=", block)
        self.assertIn("export FXDNA_MODEL_HOST_DIR=", block)
        self.assertIn("export FXDNA_MODELS=local://yolov9-t-320",
                      block)
        ups = [line for line in block.splitlines()
               if "docker compose" in line]
        self.assertGreaterEqual(len(ups), 2)
        for line in ups:
            self.assertIn("-f compose.yaml -f compose.local.yaml",
                          line)

    def test_portainer_block_matches_shipped_overlay(self):
        doc = parse_subset(readme_portainer_block(self.text))
        xdna = doc["services"]["xdna"]
        self.assertEqual(
            xdna["environment"]["FXDNA_MODEL_DIR"], "/models")
        volumes = [str(v) for v in xdna["volumes"]]
        self.assertIn("xdna-data:/data", volumes)
        self.assertTrue(
            any(v.endswith(":/models:ro") and "FXDNA_MODEL_HOST_DIR" in v
                for v in volumes))
        self.assertIn("ghcr.io/mitchins/frigate-xdna",
                      str(xdna["image"]))
        self.assertEqual(xdna["ulimits"]["memlock"]["soft"], -1)
        self.assertNotIn("PLUS_API_KEY", str(xdna["environment"]))


class TestReadmeAgreesWithFiles(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(README, encoding="utf-8") as f:
            cls.text = f.read()
        image = load_example("compose.yaml")["services"]["xdna"]["image"]
        m = re.search(r":-([^}]+)\}", str(image))
        assert m, f"cannot extract default image from {image}"
        cls.default_image = m.group(1)

    def test_readme_pulls_the_shipped_default_image(self):
        self.assertIn(f"docker pull {self.default_image}", self.text)

    def test_structure_what_why_how_before_install(self):
        positions = [self.text.index("## What is this?"),
                     self.text.index("## Why use it?"),
                     self.text.index("## How do I run it?"),
                     self.text.index("## Install")]
        self.assertEqual(positions, sorted(positions))

    def test_both_networking_arrangements_documented(self):
        self.assertIn("tcp://xdna:5555", self.text)
        self.assertIn("tcp://127.0.0.1:5555", self.text)

    def test_single_env_credential_route(self):
        self.assertIn("PLUS_API_KEY", self.text)
        self.assertNotIn("PLUS_API_KEY_FILE", self.text)
        self.assertNotIn("0600", self.text)
        # The key is never logged and stays out of status output.
        self.assertIn("never logged", self.text)

    def test_no_promotional_or_invented_claims(self):
        lowered = self.text.lower()
        for banned in ("unlock", "seamless", "enterprise-grade",
                       "coral", "watts", "10x", "100x"):
            self.assertNotIn(banned, lowered)
        self.assertIn("has not been measured", self.text)

    def test_performance_figures_carry_boundaries(self):
        self.assertIn("Frigate-reported", self.text)
        self.assertIn("9.97 ms", self.text)
        self.assertIn("Not native p50", self.text)
        self.assertIn("485 MiB", self.text)
        self.assertIn("not all of Frigate", self.text)
        self.assertIn("percentiles were not instrumented", self.text)

    def test_user_states_are_documented(self):
        for marker in ("waiting for Frigate",
                       "elapsed=", "requirement failure",
                       "Preparation failed", "health --ready",
                       "recover <model> --acknowledge",
                       "memlock"):
            self.assertIn(marker, self.text)


if __name__ == "__main__":
    unittest.main()

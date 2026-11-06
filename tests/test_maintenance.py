import importlib.util
import shutil
import tempfile
import unittest
from pathlib import Path

from tools.check_maintenance import (
    parse_lock,
    requirement,
    validate_action,
    validate_configuration,
    validate_dependencies,
)


class MaintenanceTests(unittest.TestCase):
    def test_updated_dependency_versions_remain_valid(self):
        self.assertEqual(
            parse_lock("--index-url https://pypi.org/simple\nnumpy==3.0.0\n"), {"numpy": "3.0.0"}
        )
        self.assertEqual(requirement("PyYAML==7.1.0rc1"), ("pyyaml", "7.1.0rc1"))

    def test_floating_or_redirected_dependencies_are_rejected(self):
        for line in (
            "numpy>=2",
            "numpy==2.*",
            "numpy @ https://example.invalid/a.whl",
            "--extra-index-url https://example.invalid",
            "numpy==",
        ):
            with self.subTest(line=line), self.assertRaises(ValueError):
                parse_lock("--index-url https://pypi.org/simple\n" + line)
        with self.assertRaises(ValueError):
            parse_lock("--index-url https://example.invalid/simple\nnumpy==2.2.6")

    def copy_dependency_files(self, root):
        project = Path(__file__).resolve().parents[1]
        for source in (
            project / "pyproject.toml",
            project / "requirements-dev.txt",
            project / "requirements-native.txt",
            *project.glob("requirements-*.lock"),
        ):
            shutil.copy2(source, root / source.name)

    def test_complete_dependency_hashes_and_updates_are_allowed(self):
        for value in ("1a" * 32, "b2" * 32):
            with self.subTest(value=value):
                self.assertEqual(
                    parse_lock(
                        "--index-url https://pypi.org/simple\n"
                        f"--require-hashes\nnumpy==3.0.0 --hash=sha256:{value}\n"
                    ),
                    {"numpy": "3.0.0"},
                )
        self.assertEqual(
            parse_lock(
                "--index-url https://pypi.org/simple\n"
                f"numpy==3.0.0 --hash sha256:{'cd' * 32} --hash=sha256:{'ef' * 32}\n"
            ),
            {"numpy": "3.0.0"},
        )

    def test_incomplete_or_unlocked_hashed_dependencies_are_rejected(self):
        for line in (
            "numpy==3.0.0",
            "numpy==3.0.0 --hash=sha256:" + "a" * 63,
            "numpy==3.0.0 --hash=sha256:" + "g" * 64,
            "numpy==3.0.0 --hash=md5:" + "a" * 32,
            "numpy==3.* --hash=sha256:" + "a" * 64,
            "numpy @ https://example.invalid/a.whl --hash=sha256:" + "a" * 64,
        ):
            with self.subTest(line=line), self.assertRaises(ValueError):
                parse_lock("--index-url https://pypi.org/simple\n--require-hashes\n" + line)

    def test_implicit_hash_mode_requires_all_dependency_hashes(self):
        with self.assertRaisesRegex(ValueError, "without a hash"):
            parse_lock(
                "--index-url https://pypi.org/simple\n"
                f"numpy==3.0.0 --hash=sha256:{'ab' * 32}\nwheel==1.0.0\n"
            )
        self.assertEqual(
            parse_lock(
                "--index-url https://pypi.org/simple\n"
                f"numpy==3.0.0 --hash=sha256:{'ab' * 32}\n"
                f"wheel==1.0.0 --hash=sha256:{'cd' * 32}\n"
            ),
            {"numpy": "3.0.0", "wheel": "1.0.0"},
        )

    def test_development_hash_mode_covers_included_locks_and_parsers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.copy_dependency_files(root)
            dev = root / "requirements-dev.txt"
            original_dev = dev.read_text(encoding="utf-8")
            hashes = " --hash=sha256:" + "ab" * 32
            hashed_dev = []
            for line in original_dev.splitlines():
                if line.startswith("-r "):
                    lock = root / line[3:]
                    rows = lock.read_text(encoding="utf-8").splitlines()
                    lock.write_text(
                        "\n".join(row if row.startswith("--") else row + hashes for row in rows)
                        + "\n",
                        encoding="utf-8",
                    )
                    hashed_dev.append(line)
                else:
                    hashed_dev.append(line + hashes)
            hashed_dev = "\n".join(hashed_dev) + "\n"
            dev.write_text(hashed_dev, encoding="utf-8")
            validate_dependencies(root)
            dev.write_text(original_dev, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unhashed"):
                validate_dependencies(root)
            dev.write_text(hashed_dev, encoding="utf-8")
            arrays = root / "requirements-arrays.lock"
            arrays.write_text(
                arrays.read_text(encoding="utf-8").replace(hashes, ""), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "unhashed"):
                validate_dependencies(root)

    def test_lock_comments_layout_and_continuations_are_allowed(self):
        text = (
            "# package lock\n\n"
            "numpy==3.0.0 \\\n"
            f"  --hash=sha256:{'ab' * 32}  # published wheel\n"
            "--index-url=https://pypi.org/simple\n--require-hashes\n"
        )
        self.assertEqual(parse_lock(text), {"numpy": "3.0.0"})
        for text in (
            "--index-url https://pypi.org/simple\nnumpy==3.0.0\nnumpy==3.0.0",
            "--index-url https://pypi.org/simple",
            "numpy==3.0.0",
            "--index-url https://pypi.org/simple\nnumpy==3.0.0 \\",
        ):
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_lock(text)

    def test_development_includes_are_order_independent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.copy_dependency_files(root)
            path = root / "requirements-dev.txt"
            lines = path.read_text(encoding="utf-8").splitlines()
            path.write_text(
                "# development dependencies\n\n" + "\n".join(reversed(lines)) + "\n",
                encoding="utf-8",
            )
            validate_dependencies(root)
            text = path.read_text(encoding="utf-8").replace(
                "-r requirements-build.lock", "--requirement=requirements-build.lock"
            )
            text = text.replace("-r requirements-arrays.lock", "-rrequirements-arrays.lock")
            text = text.replace('python_version < "3.11"', "python_version<'3.11'")
            path.write_text(text, encoding="utf-8")
            validate_dependencies(root)

    def test_development_dependency_contract_rejects_semantic_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.copy_dependency_files(root)
            path = root / "requirements-dev.txt"
            original = path.read_text(encoding="utf-8")
            invalid = (
                original.replace("-r requirements-build.lock\n", ""),
                original + "-r requirements-build.lock\n",
                original.replace("requirements-lint.lock", "requirements-other.lock"),
                original.replace('python_version < "3.11"', 'python_version < "3.12"'),
                original + "pyyaml==6.0.3\n",
                original + "unknown==1.0.0\n",
            )
            for text in invalid:
                with self.subTest(text=text), self.assertRaises(ValueError):
                    path.write_text(text, encoding="utf-8")
                    validate_dependencies(root)

    def test_native_include_uses_dependency_semantics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.copy_dependency_files(root)
            path = root / "requirements-native.txt"
            path.write_text(
                "# native dependencies\n\n--requirement requirements-native.lock # locked\n",
                encoding="utf-8",
            )
            validate_dependencies(root)
            for text in (
                "",
                "-r requirements-arrays.lock\n",
                "-r requirements-native.lock\n-r requirements-native.lock\n",
                "-r requirements-native.lock\nunknown==1.0.0\n",
            ):
                with self.subTest(text=text), self.assertRaises(ValueError):
                    path.write_text(text, encoding="utf-8")
                    validate_dependencies(root)

    def test_updated_action_commit_is_allowed(self):
        validate_action("actions/setup-python@" + "1a" * 20)
        validate_action("actions/checkout@" + "b2" * 20)

    def test_floating_truncated_or_wrong_source_action_is_rejected(self):
        for reference in (
            "actions/checkout@v4",
            "actions/setup-python@main",
            "actions/checkout@" + "a" * 39,
            "unknown/setup-python@" + "a" * 40,
            "actions/upload-artifact@" + "0" * 40,
        ):
            with self.subTest(reference=reference), self.assertRaises(ValueError):
                validate_action(reference)

    @unittest.skipUnless(
        importlib.util.find_spec("yaml"), "development YAML parser is not installed"
    )
    def test_real_configuration_and_permission_negative(self):
        project = Path(__file__).resolve().parents[1]
        validate_configuration(project)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / ".github/workflows").mkdir(parents=True)
            text = (project / ".github/workflows/ci.yml").read_text(encoding="utf-8")
            (root / ".github/workflows/ci.yml").write_text(
                text.replace("contents: read", "contents: write"), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "read-only"):
                validate_configuration(root)

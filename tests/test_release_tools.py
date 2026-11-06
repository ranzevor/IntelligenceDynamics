import base64
import csv
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from tools.build_release import (
    build_release,
    export_commit,
    external_destination,
    git_identity,
    source_git_tree,
    write_source_archive,
)
from tools.smoke_wheel import verify_assets


class ReleaseToolTests(unittest.TestCase):
    def test_destination_guard_preserves_existing_and_source_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            source = parent / "source"
            source.mkdir()
            marker = source / "kept.txt"
            marker.write_text("keep", encoding="utf-8")
            for target in (source, source / "build"):
                with self.subTest(target=target), self.assertRaisesRegex(ValueError, "outside"):
                    external_destination(source, target)
            existing = parent / "existing"
            existing.mkdir()
            with self.assertRaises(FileExistsError):
                external_destination(source, existing)
            self.assertEqual(marker.read_text(encoding="utf-8"), "keep")
            self.assertEqual(external_destination(source, parent / "new"), parent / "new")

    @unittest.skipUnless(os.name == "nt", "Windows directory junction")
    def test_destination_rejects_junction_ancestor(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, target, link = root / "source", root / "target", root / "linked"
            source.mkdir()
            target.mkdir()
            result = subprocess.run(
                ["cmd", "/d", "/c", "mklink", "/J", str(link), str(target)],
                capture_output=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            try:
                with self.assertRaisesRegex(ValueError, "reparse"):
                    external_destination(source, link / "new-release")
                self.assertFalse((target / "new-release").exists())
            finally:
                link.rmdir()

    def test_tools_load_from_non_source_directory(self):
        project = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            for name in ("build_release.py", "smoke_wheel.py"):
                with self.subTest(tool=name):
                    result = subprocess.run(
                        [sys.executable, "-I", "-B", str(project / "tools" / name), "--help"],
                        cwd=directory,
                        capture_output=True,
                    )
                    self.assertEqual(result.returncode, 0, result.stderr)

    def test_archive_contains_only_registered_names_and_original_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "sample.txt").write_bytes("合成材料\n".encode())
            (root / ".git").mkdir()
            (root / ".git/private").write_text("excluded", encoding="utf-8")
            output = root / "source.zip"
            write_source_archive(root, ["sample.txt"], output, 946684800)
            with zipfile.ZipFile(output) as archive:
                self.assertEqual(len(archive.namelist()), 1)
                self.assertEqual(
                    archive.read(archive.namelist()[0]), (root / "sample.txt").read_bytes()
                )
            with self.assertRaises(FileExistsError):
                write_source_archive(root, ["sample.txt"], output, 946684800)

    def test_release_requires_a_commit_and_preserves_destination_on_rejection(self):
        from intelligence_dynamics.validation import PROJECT_FILES

        project = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            source = parent / "source"
            source.mkdir()
            for name in PROJECT_FILES:
                target = source / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(project / name, target)
                target.chmod(0o644)
            destination = parent / "release"
            with self.assertRaisesRegex(ValueError, "exact Git commit"):
                build_release(source, destination)
            self.assertFalse(destination.exists())

    def test_source_tree_matches_git_object_order_and_blob_framing(self):
        self.assertEqual(source_git_tree({}), "4b825dc642cb6eb9a060e54bf8d69288fbee4904")
        entries = {
            "a.b": b"before directory\n",
            "a/empty": b"",
            "a/deep/value": b"nested\n",
            "a0": b"after directory\n",
            "中文.txt": "合成材料\n".encode("utf-8"),
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, data in entries.items():
                target = root / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
                target.chmod(0o644)
            subprocess.run(
                ["git", "init", "--quiet", "--object-format=sha1", str(root)],
                check=True,
                capture_output=True,
            )
            subprocess.run(
                ["git", "-C", str(root), "-c", "core.autocrlf=false", "add", "--all"],
                check=True,
                capture_output=True,
            )
            expected = (
                subprocess.run(
                    ["git", "-C", str(root), "write-tree"], check=True, capture_output=True
                )
                .stdout.decode()
                .strip()
            )
            self.assertEqual(source_git_tree(entries), expected)
            self.assertEqual(source_git_tree(dict(reversed(list(entries.items())))), expected)

    def test_plain_source_has_no_claimed_commit(self):
        with tempfile.TemporaryDirectory() as directory:
            identity = git_identity(Path(directory), [])
            self.assertIsNone(identity["commit"])
            self.assertIsNone(identity["tree"])

    def test_commit_identity_accepts_exact_bytes_and_rejects_dirty_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            def git(*args):
                return subprocess.run(
                    ["git", "-C", str(root), *args], check=True, capture_output=True
                ).stdout

            git("init", "--quiet")
            git("config", "core.autocrlf", "false")
            (root / "sample.txt").write_bytes(b"original\n")
            git("add", "sample.txt")
            git(
                "-c",
                "user.name=Release Fixture",
                "-c",
                "user.email=release@example.invalid",
                "-c",
                "commit.gpgsign=false",
                "commit",
                "--quiet",
                "-m",
                "synthetic fixture",
            )
            self.assertEqual(
                git_identity(root, ["sample.txt"])["commit"],
                git("rev-parse", "HEAD").decode().strip(),
            )
            (root / "sample.txt").write_bytes(b"modified\n")
            self.assertIsNone(git_identity(root, ["sample.txt"])["commit"])
            child = root / "child"
            child.mkdir()
            self.assertIsNone(git_identity(child, [])["commit"])

    def test_head_change_cannot_bind_new_bytes_to_the_old_commit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original_run = subprocess.run

            def git(*args):
                return original_run(
                    ["git", "-C", str(root), *args], check=True, capture_output=True
                ).stdout

            git("init", "--quiet")
            git("config", "core.autocrlf", "false")

            def commit(content):
                (root / "sample.txt").write_bytes(content)
                git("add", "sample.txt")
                git(
                    "-c",
                    "user.name=Release Fixture",
                    "-c",
                    "user.email=release@example.invalid",
                    "-c",
                    "commit.gpgsign=false",
                    "commit",
                    "--quiet",
                    "-m",
                    "synthetic fixture",
                )
                return git("rev-parse", "HEAD").decode().strip()

            first = commit(b"original\n")
            second = commit(b"second\n")
            git("update-ref", "HEAD", first, second)
            (root / "sample.txt").write_bytes(b"original\n")

            def advance_after_read(command, **kwargs):
                result = original_run(command, **kwargs)
                if command[-2:] == ["rev-parse", "HEAD"]:
                    git("update-ref", "HEAD", second, first)
                    (root / "sample.txt").write_bytes(b"second\n")
                return result

            with patch("tools.build_release.subprocess.run", side_effect=advance_after_read):
                identity = git_identity(root, ["sample.txt"])
            self.assertIsNone(identity["commit"])
            self.assertEqual(identity["base_commit"], first)

    def test_commit_export_uses_object_bytes_after_working_file_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            root = parent / "repository"
            root.mkdir()

            def git(*args):
                return subprocess.run(
                    ["git", "-C", str(root), *args], check=True, capture_output=True
                ).stdout

            git("init", "--quiet")
            git("config", "core.autocrlf", "false")
            (root / "sample.txt").write_bytes(b"original\n")
            git("add", "sample.txt")
            git(
                "-c",
                "user.name=Release Fixture",
                "-c",
                "user.email=release@example.invalid",
                "-c",
                "commit.gpgsign=false",
                "commit",
                "--quiet",
                "-m",
                "synthetic fixture",
            )
            commit = git("rev-parse", "HEAD").decode().strip()
            (root / "sample.txt").write_bytes(b"working change\n")
            export_commit(root, commit, ["sample.txt"], parent / "export")
            self.assertEqual((parent / "export/sample.txt").read_bytes(), b"original\n")
            with self.assertRaisesRegex(ValueError, "boundary"):
                export_commit(root, commit, ["unregistered.txt"], parent / "invalid")
            self.assertFalse((parent / "invalid").exists())

    def test_change_after_commit_validation_cannot_enter_the_build(self):
        from intelligence_dynamics.validation import PROJECT_FILES

        project = Path(__file__).resolve().parents[1]
        original_run = subprocess.run
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            root = parent / "repository"
            root.mkdir()
            for name in PROJECT_FILES:
                target = root / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(project / name, target)
                target.chmod(0o644)

            def git(*args):
                return original_run(
                    ["git", "-C", str(root), *args], check=True, capture_output=True
                ).stdout

            git("init", "--quiet")
            git("config", "core.autocrlf", "false")
            git("add", "--all")
            git(
                "-c",
                "user.name=Release Fixture",
                "-c",
                "user.email=release@example.invalid",
                "-c",
                "commit.gpgsign=false",
                "commit",
                "--quiet",
                "-m",
                "synthetic fixture",
            )
            module = root / "src/intelligence_dynamics/__init__.py"
            original = module.read_bytes()

            def change_after_identity(command, **kwargs):
                result = original_run(command, **kwargs)
                if command[-2] == "rev-parse" and command[-1].endswith("^{tree}"):
                    module.write_bytes(original + b"\n")
                return result

            output = parent / "release"
            with patch("tools.build_release.subprocess.run", side_effect=change_after_identity):
                with self.assertRaisesRegex(ValueError, "captured Git commit"):
                    build_release(root, output)
            self.assertEqual(
                (output / "source/src/intelligence_dynamics/__init__.py").read_bytes(), original
            )
            self.assertFalse((output / "artifacts").exists())


class ReleaseAssetTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.make_release("0.1.0")

    @staticmethod
    def entry(name, data):
        return {"path": name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}

    def make_release(self, version):
        self.version = version
        self.wheel = self.root / f"intelligence_dynamics-{version}-py3-none-any.whl"
        self.source = self.root / f"IntelligenceDynamics-{version}-source.zip"
        self.source_bytes = {
            "src/intelligence_dynamics/__init__.py": f'__version__ = "{version}"\n'.encode(),
            "src/intelligence_dynamics/cli.py": b"def main():\n    return 0\n",
            "src/intelligence_dynamics/py.typed": b"",
            "LICENSE": b"Synthetic license fixture\n",
            "configs/synthetic.json": b'{"synthetic":true}\n',
            "pyproject.toml": (
                '[project]\nname = "intelligence-dynamics"\n'
                f'version = "{version}"\nrequires-python = ">=3.10"\n'
                'license = {text = "Apache-2.0"}\n'
                '[project.optional-dependencies]\narrays = ["numpy==2.2.6"]\n'
                '[project.scripts]\nintelligence-dynamics = "intelligence_dynamics.cli:main"\n'
                '[tool.setuptools]\nlicense-files = ["LICENSE"]\n'
                '[tool.setuptools.packages.find]\nwhere = ["src"]\n'
                '[tool.setuptools.package-data]\nintelligence_dynamics = ["py.typed"]\n'
            ).encode(),
        }
        self.write_source()
        self.write_wheel(self.expected_wheel())
        self.manifest = {
            "format": "intelligence-dynamics-release-v1",
            "project": "IntelligenceDynamics",
            "version": version,
        }
        self.refresh_manifests()

    def write_source(self):
        with zipfile.ZipFile(self.source, "w") as archive:
            for name, data in sorted(self.source_bytes.items()):
                archive.writestr(f"IntelligenceDynamics-{self.version}/{name}", data)

    def expected_wheel(self):
        dist = f"intelligence_dynamics-{self.version}.dist-info/"
        entries = {
            name.removeprefix("src/"): data
            for name, data in self.source_bytes.items()
            if name.startswith("src/")
        }
        entries.update(
            {
                dist + "LICENSE": self.source_bytes["LICENSE"],
                dist + "METADATA": (
                    "Metadata-Version: 2.2\nName: intelligence-dynamics\n"
                    f"Version: {self.version}\nRequires-Python: >=3.10\nLicense-File: LICENSE\n"
                    "License: Apache-2.0\nProvides-Extra: arrays\n"
                    'Requires-Dist: numpy==2.2.6; extra == "arrays"\n\n'
                ).encode(),
                dist + "WHEEL": b"Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n\n",
                dist + "entry_points.txt": (
                    b"[console_scripts]\nintelligence-dynamics = intelligence_dynamics.cli:main\n"
                ),
                dist + "top_level.txt": b"intelligence_dynamics\n",
            }
        )
        return entries

    def write_wheel(self, entries):
        entries = dict(entries)
        record = f"intelligence_dynamics-{self.version}.dist-info/RECORD"
        entries.pop(record, None)
        stream = io.StringIO(newline="")
        writer = csv.writer(stream, lineterminator="\n")
        for name, data in sorted(entries.items()):
            checksum = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
            writer.writerow((name, "sha256=" + checksum, len(data)))
        writer.writerow((record, "", ""))
        entries[record] = stream.getvalue().encode()
        with zipfile.ZipFile(self.wheel, "w") as archive:
            for name, data in sorted(entries.items()):
                archive.writestr(name, data)

    def refresh_manifests(self):
        self.manifest["assets"] = [
            self.entry(p.name, p.read_bytes()) for p in (self.wheel, self.source)
        ]
        with zipfile.ZipFile(self.wheel) as archive:
            self.manifest["wheel_files"] = [
                self.entry(name, archive.read(name)) for name in sorted(archive.namelist())
            ]
        self.manifest["source_files"] = [
            self.entry(name, data) for name, data in sorted(self.source_bytes.items())
        ]
        identities = [
            {"path": row["path"], "sha256": row["sha256"]} for row in self.manifest["source_files"]
        ]
        self.manifest["source"] = {"tree": source_git_tree(self.source_bytes)}
        self.manifest["source_manifest_sha256"] = hashlib.sha256(
            json.dumps(
                identities, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ).encode()
        ).hexdigest()
        self.save_manifest()

    def save_manifest(self):
        (self.root / "release-manifest.json").write_text(
            json.dumps(self.manifest), encoding="utf-8"
        )
        (self.root / "SHA256SUMS").write_text(
            "".join(
                hashlib.sha256(p.read_bytes()).hexdigest() + "  " + p.name + "\n"
                for p in (self.wheel, self.source, self.root / "release-manifest.json")
            ),
            encoding="utf-8",
        )

    def test_complete_asset_set_is_accepted(self):
        manifest, wheel, source = verify_assets(self.root)
        self.assertEqual(manifest, self.manifest)
        self.assertEqual((wheel, source), (self.wheel, self.source))

    def test_unlisted_consumed_wheel_is_rejected(self):
        self.manifest["assets"] = [
            row for row in self.manifest["assets"] if row["path"] != self.wheel.name
        ]
        self.save_manifest()
        with self.assertRaisesRegex(ValueError, "unique and complete"):
            verify_assets(self.root)

    def test_duplicate_and_extra_assets_are_rejected(self):
        self.manifest["assets"].append(dict(self.manifest["assets"][0]))
        self.save_manifest()
        with self.assertRaisesRegex(ValueError, "unique and complete"):
            verify_assets(self.root)
        self.manifest["assets"].pop()
        self.save_manifest()
        (self.root / "extra.whl").write_bytes(b"unlisted")
        with self.assertRaisesRegex(ValueError, "extra assets"):
            verify_assets(self.root)

    def test_asset_hash_update_cannot_hide_changed_wheel_contents(self):
        entries = self.expected_wheel()
        entries["intelligence_dynamics/__init__.py"] += b"\n"
        self.write_wheel(entries)
        data = self.wheel.read_bytes()
        self.manifest["assets"][0].update(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
        self.save_manifest()
        with self.assertRaisesRegex(ValueError, "archive entry identity"):
            verify_assets(self.root)

    def test_self_consistent_wheel_manifest_cannot_hide_missing_promised_files(self):
        for name in (
            "intelligence_dynamics/py.typed",
            "intelligence_dynamics/cli.py",
            "intelligence_dynamics-0.1.0.dist-info/LICENSE",
        ):
            with self.subTest(name=name):
                entries = self.expected_wheel()
                del entries[name]
                self.write_wheel(entries)
                self.refresh_manifests()
                with self.assertRaisesRegex(ValueError, "wheel (package members|license mapping)"):
                    verify_assets(self.root)

    def test_self_consistent_wheel_manifest_cannot_hide_source_content_mismatch(self):
        for name in (
            "intelligence_dynamics/__init__.py",
            "intelligence_dynamics-0.1.0.dist-info/LICENSE",
        ):
            with self.subTest(name=name):
                entries = self.expected_wheel()
                entries[name] += b"changed\n"
                self.write_wheel(entries)
                self.refresh_manifests()
                with self.assertRaisesRegex(ValueError, "wheel package content"):
                    verify_assets(self.root)

    def test_entrypoint_and_metadata_must_match_packaging_declarations(self):
        dist = "intelligence_dynamics-0.1.0.dist-info/"
        for name, content in (
            (
                dist + "entry_points.txt",
                b"[console_scripts]\nother = intelligence_dynamics.cli:main\n",
            ),
            (
                dist + "METADATA",
                self.expected_wheel()[dist + "METADATA"].replace(b"0.1.0", b"0.9.0"),
            ),
        ):
            with self.subTest(name=name):
                entries = self.expected_wheel()
                entries[name] = content
                self.write_wheel(entries)
                self.refresh_manifests()
                with self.assertRaisesRegex(ValueError, "wheel (entry points|metadata)"):
                    verify_assets(self.root)

    def test_wrong_or_missing_source_manifest_identity_is_rejected(self):
        for value in ("0" * 64, "a" * 63, "G" * 64, None):
            with self.subTest(value=value):
                self.manifest["source_manifest_sha256"] = value
                self.save_manifest()
                with self.assertRaisesRegex(ValueError, "source manifest identity"):
                    verify_assets(self.root)
        del self.manifest["source_manifest_sha256"]
        self.save_manifest()
        with self.assertRaisesRegex(ValueError, "source manifest identity"):
            verify_assets(self.root)

    def test_wrong_or_malformed_git_tree_is_rejected_after_checksum_update(self):
        for value in ("0" * 40, "1" * 40, "a" * 39, "G" * 40, None, 123, {}):
            with self.subTest(value=value):
                self.manifest["source"]["tree"] = value
                self.save_manifest()
                with self.assertRaisesRegex(ValueError, "source Git tree"):
                    verify_assets(self.root)
        del self.manifest["source"]["tree"]
        self.save_manifest()
        with self.assertRaisesRegex(ValueError, "source Git tree"):
            verify_assets(self.root)
        del self.manifest["source"]
        self.save_manifest()
        with self.assertRaisesRegex(ValueError, "source identity"):
            verify_assets(self.root)

    def test_document_change_requires_matching_tree_and_allows_new_tree(self):
        previous = self.manifest["source"]["tree"]
        self.source_bytes["README.md"] = b"Updated synthetic documentation\n"
        self.write_source()
        self.refresh_manifests()
        updated = self.manifest["source"]["tree"]
        self.assertNotEqual(updated, previous)
        self.manifest["source"]["tree"] = previous
        self.save_manifest()
        with self.assertRaisesRegex(ValueError, "source Git tree"):
            verify_assets(self.root)
        self.manifest["source"]["tree"] = updated
        self.manifest["source_files"].reverse()
        self.save_manifest()
        verify_assets(self.root)

    def test_duplicate_source_identity_is_rejected(self):
        self.manifest["source_files"].append(dict(self.manifest["source_files"][0]))
        self.save_manifest()
        with self.assertRaisesRegex(ValueError, "duplicate source manifest"):
            verify_assets(self.root)

    def test_source_identity_uses_sorted_paths_and_excludes_transport_sizes(self):
        self.manifest["source_files"].reverse()
        self.save_manifest()
        verify_assets(self.root)
        self.manifest["source_files"][0]["bytes"] += 1
        self.save_manifest()
        with self.assertRaisesRegex(ValueError, "archive entry identity"):
            verify_assets(self.root)

    def test_new_version_and_added_source_module_are_accepted(self):
        self.root = self.root / "updated"
        self.root.mkdir()
        self.make_release("0.2.0")
        self.source_bytes["src/intelligence_dynamics/added.py"] = b"value = 2\n"
        self.source_bytes["src/intelligence_dynamics/py.typed"] = b"# Updated type marker\n"
        self.write_source()
        self.write_wheel(self.expected_wheel())
        self.refresh_manifests()
        manifest, _, _ = verify_assets(self.root)
        self.assertEqual(manifest["version"], "0.2.0")

    def test_declared_package_data_is_checked_in_source_and_wheel(self):
        self.source_bytes["pyproject.toml"] = self.source_bytes["pyproject.toml"].replace(
            b'["py.typed"]', b'["py.typed", "data/sample.txt"]'
        )
        data_path = "src/intelligence_dynamics/data/sample.txt"
        self.source_bytes[data_path] = b"package data\n"
        self.write_source()
        self.write_wheel(self.expected_wheel())
        self.refresh_manifests()
        verify_assets(self.root)
        entries = self.expected_wheel()
        del entries[data_path.removeprefix("src/")]
        self.write_wheel(entries)
        self.refresh_manifests()
        with self.assertRaisesRegex(ValueError, "wheel package members"):
            verify_assets(self.root)
        del self.source_bytes[data_path]
        self.write_source()
        self.write_wheel(self.expected_wheel())
        self.refresh_manifests()
        with self.assertRaisesRegex(ValueError, "declared package data"):
            verify_assets(self.root)

    def test_record_contents_are_checked_after_manifest_updates(self):
        with zipfile.ZipFile(self.wheel) as archive:
            entries = {name: archive.read(name) for name in archive.namelist()}
        record = "intelligence_dynamics-0.1.0.dist-info/RECORD"
        entries[record] = entries[record].replace(b"sha256=", b"sha512=", 1)
        with zipfile.ZipFile(self.wheel, "w") as archive:
            for name, data in entries.items():
                archive.writestr(name, data)
        self.refresh_manifests()
        with self.assertRaisesRegex(ValueError, "wheel RECORD identity"):
            verify_assets(self.root)

    @unittest.skipUnless(os.name == "nt", "Windows directory junction")
    def test_consumer_rejects_junction_before_resolving_archive_root(self):
        from tools.smoke_wheel import smoke

        link = self.root.parent / (self.root.name + "-linked")
        result = subprocess.run(
            ["cmd", "/d", "/c", "mklink", "/J", str(link), str(self.root)], capture_output=True
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        try:
            with self.assertRaisesRegex(ValueError, "reparse"):
                verify_assets(link)
            with self.assertRaisesRegex(ValueError, "reparse"):
                smoke(link, self.root / "not-created")
            self.assertFalse((self.root / "not-created").exists())
        finally:
            link.rmdir()

    def test_dependency_and_license_metadata_must_match_source(self):
        name = "intelligence_dynamics-0.1.0.dist-info/METADATA"
        for old, new, message in (
            (b"numpy==2.2.6", b"numpy==9.9.9", "wheel dependencies"),
            (b"License: Apache-2.0", b"License: Proprietary", "wheel license metadata"),
            (b"Provides-Extra: arrays", b"Provides-Extra: other", "wheel dependencies"),
        ):
            with self.subTest(change=new):
                entries = self.expected_wheel()
                entries[name] = entries[name].replace(old, new)
                self.write_wheel(entries)
                self.refresh_manifests()
                with self.assertRaisesRegex(ValueError, message):
                    verify_assets(self.root)

    def test_dependency_metadata_accepts_equivalent_formatting_and_legal_upgrade(self):
        name = "intelligence_dynamics-0.1.0.dist-info/METADATA"
        entries = self.expected_wheel()
        entries[name] = entries[name].replace(
            b'numpy==2.2.6; extra == "arrays"', b"numpy == 2.2.6 ; extra=='arrays'"
        )
        entries[name] = b"\n".join(reversed(entries[name].strip().splitlines())) + b"\n\n"
        self.write_wheel(entries)
        self.refresh_manifests()
        verify_assets(self.root)
        self.source_bytes["pyproject.toml"] = self.source_bytes["pyproject.toml"].replace(
            b"numpy==2.2.6", b"numpy==2.2.7"
        )
        self.write_source()
        entries[name] = entries[name].replace(b"2.2.6", b"2.2.7")
        self.write_wheel(entries)
        self.refresh_manifests()
        verify_assets(self.root)

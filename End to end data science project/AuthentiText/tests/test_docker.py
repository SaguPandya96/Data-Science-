from __future__ import annotations

import os
import unittest
from pathlib import Path
from unittest.mock import patch

from authentitext.api import application_root

REPO_ROOT = Path(__file__).resolve().parents[1]


class DockerConfigurationTests(unittest.TestCase):
    def test_image_is_pinned_non_root_and_health_checked(self) -> None:
        dockerfile = (REPO_ROOT / "Dockerfile").read_text(encoding="utf-8")
        required = (
            "FROM python:3.14.6-slim-bookworm",
            "AUTHENTITEXT_ROOT=/app",
            "pip==26.1.2",
            "COPY data/metadata ./data/metadata",
            "USER authentitext",
            "EXPOSE 8000",
            "HEALTHCHECK",
            "/health/ready",
            "authentitext.api:app",
            '"--no-access-log"',
        )
        for fragment in required:
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, dockerfile)
        self.assertNotIn("COPY .", dockerfile)
        self.assertNotIn("apt-get", dockerfile)

    def test_compose_mounts_artifacts_read_only_and_drops_privileges(self) -> None:
        compose = (REPO_ROOT / "compose.yaml").read_text(encoding="utf-8")
        required = (
            "./artifacts/baselines/id:/app/artifacts/baselines/id:ro",
            "read_only: true",
            "cap_drop:",
            "- ALL",
            "no-new-privileges:true",
            "${AUTHENTITEXT_PORT:-8000}:8000",
        )
        for fragment in required:
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, compose)

    def test_build_context_excludes_private_and_large_local_state(self) -> None:
        ignored = set((REPO_ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines())
        self.assertTrue(
            {
                ".git",
                ".env",
                "artifacts",
                "data/raw",
                "data/interim",
                "data/processed",
                "tests",
            }
            <= ignored
        )

    def test_runtime_root_can_be_configured_for_installed_package(self) -> None:
        configured = REPO_ROOT / "container-root"
        with patch.dict(os.environ, {"AUTHENTITEXT_ROOT": str(configured)}):
            self.assertEqual(application_root(), configured.resolve())

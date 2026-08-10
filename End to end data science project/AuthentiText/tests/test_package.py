import tomllib
import unittest
from pathlib import Path

import authentitext


class PackageTests(unittest.TestCase):
    def test_version_matches_project_metadata(self) -> None:
        project_file = Path(__file__).resolve().parents[1] / "pyproject.toml"
        with project_file.open("rb") as handle:
            project_version = tomllib.load(handle)["project"]["version"]
        self.assertEqual(authentitext.__version__, project_version)


if __name__ == "__main__":
    unittest.main()

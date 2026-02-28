#!/usr/bin/env python3
"""
Validate that start_system.py properly propagates environment variables to backend.

This script prevents issues where Cloud SQL environment variables are set in
pre-loading but not passed to the main backend process.
"""

import re
import sys
from pathlib import Path


class EnvVarValidator:
    """Validates environment variable propagation in start_system.py"""

    REQUIRED_CLOUD_SQL_VARS = {
        "Ironcliw_DB_TYPE",
        "Ironcliw_DB_CONNECTION_NAME",
        "Ironcliw_DB_HOST",
        "Ironcliw_DB_PORT",
        "Ironcliw_DB_PASSWORD",
    }

    def __init__(self, start_system_path: Path):
        self.path = start_system_path
        self.content = start_system_path.read_text()
        self.errors = []
        self.warnings = []

    def validate(self) -> bool:
        """Run all validation checks"""
        print("🔍 Validating environment variable propagation...")

        self.check_env_dict_assignments()
        self.check_cloud_sql_config_propagation()
        self.check_host_and_port_set()
        self.check_pre_loading_vs_backend_env()

        # Print results
        if self.errors:
            print("\n❌ VALIDATION FAILED\n")
            for error in self.errors:
                print(f"  ❌ {error}")

        if self.warnings:
            print("\n⚠️  WARNINGS\n")
            for warning in self.warnings:
                print(f"  ⚠️  {warning}")

        if not self.errors and not self.warnings:
            print("\n✅ All environment variable checks passed!")
            return True

        return len(self.errors) == 0

    def check_env_dict_assignments(self):
        """Check that Cloud SQL env vars are set in the env dict passed to backend"""
        # Find where env dict is created and passed to backend subprocess
        env_pattern = r'env\["Ironcliw_DB_(\w+)"\]\s*='

        found_vars = set()
        for match in re.finditer(env_pattern, self.content):
            var_name = f"Ironcliw_DB_{match.group(1)}"
            found_vars.add(var_name)

        missing_vars = self.REQUIRED_CLOUD_SQL_VARS - found_vars

        if missing_vars:
            self.errors.append(
                f"Missing Cloud SQL env vars in backend env dict: {', '.join(missing_vars)}"
            )
            self.errors.append(
                "These vars must be set in the 'env' dict before passing to backend subprocess"
            )

    def check_cloud_sql_config_propagation(self):
        """Ensure Cloud SQL config is loaded AND propagated to env dict"""
        # Check if config is loaded
        if "db_config = json.load(f)" not in self.content:
            self.errors.append("Cloud SQL config file loading not found")
            return

        # Check if config values are assigned to env dict
        config_to_env_pattern = r'env\["Ironcliw_DB_\w+"\]\s*=\s*db_config'

        if not re.search(config_to_env_pattern, self.content):
            self.warnings.append("Cloud SQL config loaded but may not be propagated to env dict")

    def check_host_and_port_set(self):
        """Specifically check that Ironcliw_DB_HOST and Ironcliw_DB_PORT are set"""
        # Check for Ironcliw_DB_HOST = 127.0.0.1
        if 'env["Ironcliw_DB_HOST"] = "127.0.0.1"' not in self.content:
            if 'env["Ironcliw_DB_HOST"]' not in self.content:
                self.errors.append('Ironcliw_DB_HOST must be set to "127.0.0.1" for proxy connection')
            else:
                self.warnings.append('Ironcliw_DB_HOST is set but may not be "127.0.0.1"')

        # Check for Ironcliw_DB_PORT
        if 'env["Ironcliw_DB_PORT"]' not in self.content:
            self.errors.append("Ironcliw_DB_PORT must be set in env dict")

    def check_pre_loading_vs_backend_env(self):
        """
        Check that env vars set during pre-loading are ALSO set in backend env dict.

        This catches the issue where os.environ is set during pre-loading
        but not propagated to the subprocess env dict.
        """
        # Find os.environ assignments
        os_environ_pattern = r'os\.environ\["(Ironcliw_DB_\w+)"\]'
        os_environ_vars = set(re.findall(os_environ_pattern, self.content))

        # Find env dict assignments
        env_dict_pattern = r'env\["(Ironcliw_DB_\w+)"\]'
        env_dict_vars = set(re.findall(env_dict_pattern, self.content))

        # Warn if vars are set in os.environ but not in env dict
        missing_in_env_dict = os_environ_vars - env_dict_vars

        if missing_in_env_dict:
            self.warnings.append(
                f"Variables set in os.environ but not in subprocess env dict: "
                f"{', '.join(missing_in_env_dict)}"
            )
            self.warnings.append("Pre-loading env vars won't be visible to backend subprocess!")


def main():
    """Main validation entry point"""
    repo_root = Path(__file__).parent.parent.parent
    start_system_path = repo_root / "start_system.py"

    if not start_system_path.exists():
        print(f"❌ ERROR: {start_system_path} not found")
        sys.exit(1)

    validator = EnvVarValidator(start_system_path)
    success = validator.validate()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

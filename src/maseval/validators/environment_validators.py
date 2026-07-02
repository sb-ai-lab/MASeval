"""Validators for environment setup and runtime configuration errors.

Detects missing environment variables and credentials, config parsing errors (YAML/JSON),
dependency import issues, file permission errors, and missing files.
"""

from __future__ import annotations

import re

from maseval.validators.base import BaseValidator, CheckFn, Finding, Span


class EnvironmentSetupValidator(BaseValidator):
    """Category 3: Environment and Setup validation errors.

    Detects missing environment variables and credentials, YAML/JSON configuration
    parsing errors, dependency import failures, file access permissions, and missing files.
    """

    EXPLANATIONS = {
        "missing_environment_variable_validator": "A required environment variable was not set: the process tried to "
        "read it (e.g. via os.environ) and got a KeyError or an unset value.",
        "missing_credential_validator": "A credential needed to run was missing or unusable: an API key "
        "could not be found, credentials were absent, or a token in the "
        "environment had expired.",
        "config_parse_error_validator": "A configuration file could not be loaded: invalid YAML/JSON syntax, "
        "a parser/scanner error, or a required configuration field was missing.",
        "dependency_import_validator": "A Python dependency could not be imported: the module/package was "
        "not installed (ModuleNotFoundError/ImportError) or its version did "
        "not match the requirement.",
        "file_permission_validator": "A file operation was denied by the operating system: insufficient "
        "permissions to read or write the target file (PermissionError / "
        "access denied).",
        "file_not_found_validator": "A required file or directory did not exist at the expected path "
        "(FileNotFoundError / no such file or directory).",
    }

    def get_checks(self) -> list[CheckFn]:
        """Return the environment validation check functions.

        Returns:
            list[CheckFn]: List of check methods.
        """
        return [
            self.check_missing_env_var,
            self.check_missing_credential,
            self.check_config_parse_error,
            self.check_dependency_import,
            self.check_file_permission,
            self.check_file_not_found,
        ]

    def _scan(
        self, spans: list[Span], metric_name: str, rules: list[tuple]
    ) -> list[Finding]:
        """Scan spans against regex rules to identify environment errors.

        Args:
            spans: List of normalized spans to inspect.
            metric_name: Identifier of the metric being evaluated.
            rules: List of ``(regex_pattern, failure_type)`` tuples, or
                ``(regex_pattern, failure_type, context_cue)`` 3-tuples. When a
                ``context_cue`` is present the rule only fires if it also matches
                in the span text. At most one finding is emitted per span.

        Returns:
            list[Finding]: List of generated findings for matched regex rules.
        """
        findings: list[Finding] = []
        for span in spans:
            text = span.get("text", "")
            if not text:
                continue
            for rule in rules:
                pattern, failure_type = rule[0], rule[1]
                cue = rule[2] if len(rule) > 2 else None
                if cue is not None and not re.search(cue, text, re.IGNORECASE):
                    continue
                m = re.search(pattern, text, re.IGNORECASE)
                if m:
                    findings.append(
                        self._make_finding(
                            span,
                            metric_name=metric_name,
                            failure_type=failure_type,
                            explanation=self.EXPLANATIONS[metric_name],
                            start=m.start(),
                            end=m.end(),
                        )
                    )
                    break
        return findings

    def check_missing_env_var(self, spans: list[Span]) -> list[Finding]:
        """Detect errors caused by missing required environment variables.

        Args:
            spans: List of normalized spans to inspect.

        Returns:
            list[Finding]: List of findings for missing environment variables.
        """
        rules = [
            (
                r"missing (environment|env) (variable|var)|"
                r"(environment|env) (variable|var)[^.\n]{0,40}(not set|missing|undefined|empty)",
                "env_missing_variable",
            ),
            (
                r"os\.environ\[[^\]]+\][^.\n]{0,40}keyerror|"
                r"keyerror\W+'?[A-Z][A-Z0-9_]+'?[^.\n]{0,40}environ|"
                r"keyerror[^.\n]{0,40}environment",
                "env_missing_variable",
            ),
            (
                r"environment variable[^.\n]{0,30}required|required environment variable",
                "env_missing_variable",
            ),
        ]
        return self._scan(spans, "missing_environment_variable_validator", rules)

    def check_missing_credential(self, spans: list[Span]) -> list[Finding]:
        """Detect missing or expired API keys and authentication credentials.

        Args:
            spans: List of normalized spans to inspect.

        Returns:
            list[Finding]: List of findings for missing credentials.
        """
        rules = [
            (
                r"api[_ ]?key not (found|set|provided|configured)|"
                r"(no|missing) api[_ ]?key",
                "env_missing_credential",
            ),
            (
                r"credentials? (missing|not found|not set|not configured|unavailable)|"
                r"missing credentials?",
                "env_missing_credential",
            ),
            (r"token (expired|has expired)|expired token", "env_missing_credential"),
        ]
        return self._scan(spans, "missing_credential_validator", rules)

    def check_config_parse_error(self, spans: list[Span]) -> list[Finding]:
        """Detect syntax and parser errors when loading configuration files (YAML/JSON).

        Args:
            spans: List of normalized spans to inspect.

        Returns:
            list[Finding]: List of findings for configuration parsing errors.
        """
        rules = [
            (
                r"scannererror|parsererror|invalid yaml|"
                r"yaml[^.\n]{0,20}(parse|syntax) error|could not (parse|load) yaml",
                "config_invalid_yaml",
            ),
            (
                r"jsondecodeerror|invalid json|"
                r"json[^.\n]{0,20}(parse|syntax) error|could not (parse|load) json",
                "config_invalid_json",
            ),
            (
                r"missing required (config|configuration) (field|key|option|parameter)|"
                r"(config|configuration) [^.\n]{0,30}(field|key) [^.\n]{0,20}(missing|required|not found)|"
                r"required (config|configuration) [^.\n]{0,20}(missing|not found)",
                "config_missing_field",
            ),
        ]
        return self._scan(spans, "config_parse_error_validator", rules)

    def check_dependency_import(self, spans: list[Span]) -> list[Finding]:
        """Detect missing Python packages, import failures, and version mismatches.

        Args:
            spans: List of normalized spans to inspect.

        Returns:
            list[Finding]: List of findings for dependency import failures.
        """
        rules = [
            (r"modulenotfounderror|no module named", "env_dependency_error"),
            (
                r"importerror|cannot import name|failed to import",
                "env_dependency_error",
            ),
            (
                # Bounded so a stray "requires" and "version" far apart on one
                # (JSON-flattened) line do not span-match; needs a real version
                # constraint or an explicit version conflict/mismatch.
                r"(package|module|dependency|library)[^.\n]{0,20}version (conflict|mismatch)|"
                r"version (conflict|mismatch)|incompatible (version|dependency)|"
                r"requires? [\w.\-]+\s*[<>=~!]=?\s*[\w.\-]+",
                "env_dependency_error",
            ),
        ]
        return self._scan(spans, "dependency_import_validator", rules)

    def check_file_permission(self, spans: list[Span]) -> list[Finding]:
        """Detect OS-level file access denied and permission errors.

        Args:
            spans: List of normalized spans to inspect.

        Returns:
            list[Finding]: List of findings for permission errors.
        """
        rules = [
            (r"permissionerror|\[errno 13\]|errno 13", "env_permission_error"),
            (
                r"permission denied|access (is )?denied|operation not permitted",
                "env_permission_error",
            ),
            (
                # Require an actual permission signal near the file op, so a plain
                # read/parse failure ("failed to read PDF, file is corrupted") is
                # not mislabeled as a permission error.
                r"(cannot|could not|unable to|failed to) (write|read|open|access)"
                r"[^.\n]{0,40}(permission|denied|read-only|errno 13)",
                "env_permission_error",
            ),
        ]
        return self._scan(spans, "file_permission_validator", rules)

    def check_file_not_found(self, spans: list[Span]) -> list[Finding]:
        """Detect missing file or directory errors (FileNotFoundError).

        Args:
            spans: List of normalized spans to inspect.

        Returns:
            list[Finding]: List of findings for missing files or directories.
        """
        rules = [
            (r"filenotfounderror", "env_file_not_found"),
            (r"no such file or directory", "env_file_not_found"),
            (
                # Bounded distance so "not found"/"does not exist" must sit close
                # to the file/dir/path token, not anywhere in a long prose span.
                r"(file|directory|path)\b[^.\n]{0,30}(not found|does not exist|doesn't exist)|"
                r"(cannot|could not|unable to|failed to) (find|locate|open) (the )?(file|directory|path)\b",
                "env_file_not_found",
            ),
        ]
        return self._scan(spans, "file_not_found_validator", rules)

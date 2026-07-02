"""Validators module for detecting errors and failure patterns in agent traces.

This package provides deterministic validation checks across various trace formats
(TRAIL, Who&When, Pumpkin) to identify issues related to API calls, environment setup,
provider constraints, and tool usage schemas.
"""

from maseval.validators.base import (
    BaseValidator,
    Finding,
    Span,
    detect_format,
    severity_for,
    to_spans,
)
from maseval.validators.provider_validators import ProviderValidator
from maseval.validators.tool_schema_validators import ToolSchemaValidator
from maseval.validators.api_validators import ApiHttpValidator
from maseval.validators.environment_validators import EnvironmentSetupValidator

from maseval.validators.run_validators import (
    ALL_VALIDATORS,
    run_on_dir,
    run_on_file,
    run_on_trace,
)

__all__ = [
    "BaseValidator",
    "Finding",
    "Span",
    "detect_format",
    "severity_for",
    "to_spans",
    "ProviderValidator",
    "ToolSchemaValidator",
    "ApiHttpValidator",
    "EnvironmentSetupValidator",
    "ALL_VALIDATORS",
    "run_on_trace",
    "run_on_file",
    "run_on_dir",
]

"""Signal-block harness and validation. Plan §9."""
from .validator import ValidationError, ValidationResult, validate_signal_block

__all__ = ["ValidationError", "ValidationResult", "validate_signal_block"]

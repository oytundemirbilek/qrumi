"""Module to define all errors."""


class QRumiError(Exception):
    """Base error class for all errors."""


class QuantumDecisionError(QRumiError):
    """Error raised when something goes wrong about defining quantum decisions."""

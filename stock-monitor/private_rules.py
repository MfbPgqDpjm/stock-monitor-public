"""Explicit extension points; private parameters and rules are not distributed."""

from typing import Any, Callable


class PrivateConfigurationRequired(RuntimeError):
    """A private configuration or calculation has not been supplied."""


_parameters: dict[str, Any] = {}
_rules: dict[str, Callable[..., Any]] = {}


def configure_private_rules(*, parameters=None, rules=None) -> None:
    """Replace configuration explicitly; no files or plugins are auto-loaded."""
    new_parameters = dict(parameters or {})
    new_rules = dict(rules or {})
    if any(not callable(rule) for rule in new_rules.values()):
        raise TypeError("Every private rule must be callable")
    _parameters.clear()
    _parameters.update(new_parameters)
    _rules.clear()
    _rules.update(new_rules)


def private_parameter(name: str) -> Any:
    value = _parameters.get(name)
    if value is None:
        raise PrivateConfigurationRequired(f"Private parameter required: {name}")
    return value


def positive_window(name: str) -> int:
    value = private_parameter(name)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"Positive integer window required: {name}")
    return value


def private_rule(name: str, **context: Any) -> Any:
    rule = _rules.get(name)
    if rule is None:
        raise PrivateConfigurationRequired(f"Private rule required: {name}")
    return rule(**context)

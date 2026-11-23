import ast
import __future__
import importlib.util
import inspect
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "stock-monitor"
spec = importlib.util.spec_from_file_location("public_private_rules", SOURCE / "private_rules.py")
rules = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rules)


class PublicBoundaryTests(unittest.TestCase):
    def tearDown(self):
        rules.configure_private_rules()

    def test_missing_configuration_fails_closed(self):
        with self.assertRaises(rules.PrivateConfigurationRequired):
            rules.positive_window("example")
        with self.assertRaises(rules.PrivateConfigurationRequired):
            rules.private_rule("example")

    def test_explicit_configuration_and_rule(self):
        rules.configure_private_rules(parameters={"example": 3}, rules={"echo": lambda **ctx: ctx})
        self.assertEqual(rules.positive_window("example"), 3)
        self.assertEqual(rules.private_rule("echo", value="test"), {"value": "test"})
        rules.configure_private_rules()
        with self.assertRaises(rules.PrivateConfigurationRequired):
            rules.private_rule("echo")

    def test_invalid_windows(self):
        for value in (True, 0, -1, "3", 1.5):
            rules.configure_private_rules(parameters={"example": value})
            with self.assertRaises(ValueError):
                rules.positive_window("example")

    def test_invalid_rule_is_not_installed(self):
        with self.assertRaises(TypeError):
            rules.configure_private_rules(rules={"example": None})

    def test_no_literal_indicator_windows(self):
        for path in SOURCE.glob("*.py"):
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                    continue
                if node.func.attr not in {"rolling", "ewm"}:
                    continue
                values = list(node.args) + [kw.value for kw in node.keywords if kw.arg in {"span", "window", "halflife", "alpha", "com"}]
                for value in values:
                    self.assertNotIsInstance(value, ast.Constant, f"{path.name}:{node.lineno}: literal indicator parameter")

    def test_private_calculations_delegate_without_dependencies(self):
        names = {"check_individual_stock_signal", "check_benchmark_signal", "_compute_indicators", "_classify_trend_age", "_market_condition", "_diagnose"}
        found = set()
        for filename in ("strategy.py", "momentum_scorer.py", "macro_cycle_dashboard.py"):
            tree = ast.parse((SOURCE / filename).read_text())
            for node in ast.walk(tree):
                if not isinstance(node, ast.FunctionDef) or node.name not in names:
                    continue
                namespace = {"private_rule": rules.private_rule, "MOMENTUM_EMA_WINDOW": None}
                module = ast.Module(body=[node], type_ignores=[])
                exec(compile(module, filename, "exec", flags=__future__.annotations.compiler_flag), namespace)
                function = namespace[node.name]
                kwargs = {name: None for name in inspect.signature(function).parameters}
                with self.assertRaises(rules.PrivateConfigurationRequired):
                    function(**kwargs)
                rules.configure_private_rules(rules={node.name: lambda **context: "delegated"})
                self.assertEqual(function(**kwargs), "delegated")
                rules.configure_private_rules()
                found.add(node.name)
        self.assertEqual(found, names)


if __name__ == "__main__":
    unittest.main()

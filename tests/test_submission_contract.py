import ast
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "main.py"
ALLOWED_MAIN_IMPORTS = {"__future__", "dataclasses", "math", "typing"}
FORBIDDEN_NAMES = {
    "http",
    "httpx",
    "kaggle",
    "kaggle_environments",
    "openai",
    "os",
    "pathlib",
    "random",
    "requests",
    "socket",
    "subprocess",
    "time",
    "urllib",
}


class SubmissionContractTest(unittest.TestCase):
    def test_main_imports_only_allowed_standard_library_modules(self):
        tree = ast.parse(MAIN.read_text(encoding="utf-8"))
        imports = set()

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module.split(".")[0])

        self.assertLessEqual(imports, ALLOWED_MAIN_IMPORTS)
        self.assertFalse(imports & FORBIDDEN_NAMES)

    def test_agent_wrapper_is_final_function(self):
        tree = ast.parse(MAIN.read_text(encoding="utf-8"))
        functions = [node.name for node in tree.body if isinstance(node, ast.FunctionDef)]

        self.assertGreater(len(functions), 0)
        self.assertEqual(functions[-1], "agent")


if __name__ == "__main__":
    unittest.main()

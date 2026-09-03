from pathlib import Path
import unittest


class EntrypointTests(unittest.TestCase):
    def test_run_pipeline_script_is_self_contained(self):
        script = Path("run_pipeline.sh").read_text(encoding="utf-8")

        self.assertIn('cd "${SCRIPT_DIR}"', script)
        self.assertIn('"${SCRIPT_DIR}/run_pipeline.py"', script)
        self.assertNotIn('refactor/run_pipeline.py', script)
        self.assertNotIn('SCRIPT_DIR}/.."', script)

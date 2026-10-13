"""Run the standard-library unittest suite; persist actual status and counts."""
import json, sys, unittest, platform
from pathlib import Path
from time import perf_counter
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
start=perf_counter()
suite=unittest.defaultTestLoader.discover(str(ROOT/'tests'))
result=unittest.TextTestRunner(verbosity=2).run(suite)
(ROOT/'outputs').mkdir(exist_ok=True)
(ROOT/'outputs/test_results.json').write_text(json.dumps({
    'tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),
    'skipped':len(result.skipped),'success':result.wasSuccessful(),'runtime_seconds':perf_counter()-start,
    'python':sys.version,'platform':platform.platform(),
    'failure_details':[{'test':str(t),'traceback':msg} for t,msg in result.failures+result.errors]},indent=2),encoding='utf-8')
sys.exit(not result.wasSuccessful())

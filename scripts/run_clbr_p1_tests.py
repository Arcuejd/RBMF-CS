"""User-run tests, synthetic references and provenance without replacing old logs."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import json
import platform
import time
import unittest
from clbr_common import ROOT, new_output, preservation, source_hash, write

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    out = new_output(args.output)
    before = preservation()
    out.mkdir()
    started = time.perf_counter()
    suite = unittest.defaultTestLoader.discover(str(ROOT/'tests'))
    with (out/'tests.log').open('w', encoding='utf-8') as log:
        result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
    after = preservation()
    record = {'passed': result.wasSuccessful() and before == after, 'tests_run': result.testsRun,
              'failures': len(result.failures), 'errors': len(result.errors), 'skipped': len(result.skipped),
              'source_sha256': source_hash(), 'preservation': after, 'seconds': time.perf_counter()-started,
              'python': platform.python_version(),
              'details': [{'test': str(t), 'traceback': msg} for t, msg in result.failures+result.errors]}
    write(out/'tests.json', record)
    print(json.dumps(record, ensure_ascii=False, indent=2))
    raise SystemExit(0 if record['passed'] else 1)

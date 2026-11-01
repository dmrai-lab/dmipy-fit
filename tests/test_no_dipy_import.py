"""No shipped dmipy-fit module imports dipy: every module imports with dipy made unimportable, and the dipy
gradient-table interchange refuses by name when dipy is missing."""
import json
import subprocess
import sys
import textwrap

BLOCK = textwrap.dedent('''
    import importlib, json, pkgutil, sys

    class BlockDipy:
        def find_spec(self, name, path=None, target=None):
            if name == 'dipy' or name.startswith('dipy.'):
                raise ImportError('dipy is blocked: ' + name)

    sys.meta_path.insert(0, BlockDipy())
    import dmipy_fit
    failed = []
    for mod in pkgutil.walk_packages(dmipy_fit.__path__, 'dmipy_fit.'):
        if '.tests' in mod.name or mod.name.endswith('.conftest'):
            continue
        try:
            importlib.import_module(mod.name)
        except ImportError as error:
            failed.append((mod.name, str(error)))
    assert not [m for m in sys.modules if m == 'dipy' or m.startswith('dipy.')]
    print('FAILED', json.dumps(failed))
''')


def _run(code):
    return subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, timeout=900)


def test_every_module_imports_without_dipy():
    out = _run(BLOCK)
    assert out.returncode == 0, out.stderr[-2000:]
    failed = json.loads(out.stdout.strip().splitlines()[-1].split(' ', 1)[1])
    # a module may be missing for lack of an optional dependency other than dipy (torch), never for dipy
    assert not [f for f in failed if 'dipy' in f[1]], failed
    assert not [f for f in failed if 'torch' not in f[1]], failed


def test_dipy_interchange_refuses_by_name_without_dipy():
    out = _run(BLOCK.split('import dmipy_fit\n')[0] + textwrap.dedent('''
        from dmipy_fit.core.acquisition_scheme import gtab_dipy2dmipy
        try:
            gtab_dipy2dmipy(object())
        except ImportError as error:
            assert 'pip install dipy' in str(error)
            print('REFUSED')
    '''))
    assert out.returncode == 0 and 'REFUSED' in out.stdout, out.stderr[-2000:]

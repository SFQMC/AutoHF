import pytest

import importlib
import glob
import subprocess

ex_files = sorted(glob.glob("examples/*/*.py"))


class TestExamples:
  @pytest.mark.dev
  @pytest.mark.weekly
  @pytest.mark.parametrize("fname", ex_files)
  def test_ex(self, fname):
    # skip input file for now
    if "02_inputfile" in fname:
      return None
    try:
      res = subprocess.run(
        ["python", fname],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
      )
      assert res.returncode == 0
    except Exception as e:
      print(e)
      assert False

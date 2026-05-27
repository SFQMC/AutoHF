import pytest
import warnings

from autohf.options import process_args, RepOptions, SlaterType


class TestProcessArgs:
  """Test process_args function for parameter processing and validation."""

  def test_process_args_defaults(self):
    """Test process_args with no input uses defaults."""
    args = process_args(None, display=False)

    # Check some default values
    assert args.steps > 0
    assert args.batch_size > 0
    assert args.ansatz is not None
    assert isinstance(args.ansatz, str)

  def test_process_args_custom_values(self):
    """Test process_args with custom values."""
    custom = {
      "steps": 100,
      "batch_size": 4,
      "ansatz": "SD_ROT",
      "noncollinear": True,
    }

    args = process_args(custom, display=False)

    assert args.steps == 100
    assert args.batch_size == 4
    assert args.ansatz == "SD_ROT"
    assert args.noncollinear is True

  def test_deprecated_numTrials_warns(self, capfd):
    """Test deprecated 'numTrials' parameter shows warning."""
    with warnings.catch_warnings(record=True) as w:
      warnings.simplefilter("always")

      args = process_args({"numTrials": 8}, display=False)

      # Should warn about deprecation
      assert len(w) > 0
      assert any("deprecated" in str(warning.message).lower() for warning in w)
      assert any("numTrials" in str(warning.message) for warning in w)

      # Should also print
      captured = capfd.readouterr()
      assert "numTrials" in captured.out
      assert "deprecated" in captured.out.lower()

      # Should map to batch_size
      assert args.batch_size == 8

  def test_deprecated_numSteps_warns(self, capfd):
    """Test deprecated 'numSteps' parameter shows warning."""
    with warnings.catch_warnings(record=True) as w:
      warnings.simplefilter("always")

      args = process_args({"numSteps": 200}, display=False)

      assert len(w) > 0
      assert any("numSteps" in str(warning.message) for warning in w)

      captured = capfd.readouterr()
      assert "numSteps" in captured.out

      # Should map to steps
      assert args.steps == 200

  def test_deprecated_dumpTrials_warns(self, capfd):
    """Test deprecated 'dumpTrials' parameter shows warning."""
    with warnings.catch_warnings(record=True) as w:
      warnings.simplefilter("always")

      args = process_args({"dumpTrials": True}, display=False)

      assert len(w) > 0
      assert any("dumpTrials" in str(warning.message) for warning in w)

      captured = capfd.readouterr()
      assert "dumpTrials" in captured.out

      # Should map to dump_batch
      assert args.dump_batch is True

  def test_all_deprecated_parameters_together(self):
    """Test using multiple deprecated parameters simultaneously."""
    deprecated_params = {
      "numTrials": 5,
      "numSteps": 150,
      "dumpTrials": False,
    }

    with warnings.catch_warnings(record=True) as w:
      warnings.simplefilter("always")

      args = process_args(deprecated_params, display=False)

      # Should have 3 warnings
      assert len(w) >= 3

      # All parameters should map correctly
      assert args.batch_size == 5
      assert args.steps == 150
      assert args.dump_batch is False

  def test_process_args_display_mode(self, capfd):
    """Test process_args with display=True shows settings."""
    args = process_args({"steps": 50}, display=True)

    captured = capfd.readouterr()
    assert "AutoHF Settings" in captured.out

  def test_process_args_inputfile(self):
    """Test process_args handles inputfile parameter."""
    args = process_args({"inputfile": "test.toml"}, display=False)

    assert args.inputfile == "test.toml"

  def test_process_args_ansatz_string(self):
    """Test process_args with various ansatz strings."""
    for ansatz in ["SD_ROT", "SD", "DIAG", "SDC", "SD_ROTC"]:
      args = process_args({"ansatz": ansatz}, display=False)
      assert args.ansatz == ansatz

  def test_process_args_noncollinear_flag(self):
    """Test process_args noncollinear flag."""
    args_col = process_args({"noncollinear": False}, display=False)
    args_nc = process_args({"noncollinear": True}, display=False)

    assert args_col.noncollinear is False
    assert args_nc.noncollinear is True

  def test_process_args_riemannian_flag(self):
    """Test process_args riemannian optimization flag."""
    args_std = process_args({"riemannian": False}, display=False)
    args_riem = process_args({"riemannian": True}, display=False)

    assert args_std.riemannian is False
    assert args_riem.riemannian is True

  def test_process_args_gpu_flag(self):
    """Test process_args GPU flag."""
    args_cpu = process_args({"gpu": False}, display=False)
    args_gpu = process_args({"gpu": True}, display=False)

    assert args_cpu.gpu is False
    assert args_gpu.gpu is True

  def test_process_args_optimizer_methods(self):
    """Test process_args with different optimizer methods."""
    for method in ["lbfgs", "adam", "sgd"]:
      args = process_args({"opt_method": method}, display=False)
      assert args.opt_method == method

  def test_process_args_state0_scale(self):
    """Test process_args with state0_scale parameter."""
    args = process_args({"state0_scale": 0.5}, display=False)
    assert args.state0_scale == 0.5

  def test_process_args_verbose_flag(self):
    """Test process_args with verbose parameter."""
    args = process_args({"verbose": 2}, display=False)
    # Should accept verbose level
    assert hasattr(args, 'verbose')

  def test_process_args_output_file(self):
    """Test process_args with output file parameter."""
    args = process_args({"output": "results.pkl"}, display=False)
    assert args.output == "results.pkl"

  def test_process_args_dump_batch_flag(self):
    """Test process_args dump_batch flag."""
    args_no_dump = process_args({"dump_batch": False}, display=False)
    args_dump = process_args({"dump_batch": True}, display=False)

    assert args_no_dump.dump_batch is False
    assert args_dump.dump_batch is True

  def test_process_args_custom_ansatz(self):
    """Test process_args with CUSTOM ansatz."""
    args = process_args({"ansatz": "CUSTOM"}, display=False)
    assert args.ansatz == "CUSTOM"

  def test_process_args_accepts_string_ansatz(self):
    """Test that string ansatz values are accepted (validation happens later)."""
    # process_args returns SimpleNamespace, not dict
    settings = process_args({"ansatz": "SD_ROT"}, display=False)
    assert settings.ansatz == "SD_ROT"

  def test_process_args_overwrite_defaults(self):
    """Test that custom values override defaults."""
    # Get default steps
    args_default = process_args(None, display=False)
    default_steps = args_default.steps

    # Override with custom value
    custom_steps = default_steps + 100
    args_custom = process_args({"steps": custom_steps}, display=False)

    assert args_custom.steps == custom_steps
    assert args_custom.steps != default_steps

  def test_process_args_complex_flag(self):
    """Test process_args force_complex flag."""
    args = process_args({"force_complex": True}, display=False)
    assert args.force_complex is True

  def test_process_args_approx_expm_flag(self):
    """Test process_args approx_expm flag."""
    args = process_args({"approx_expm": True}, display=False)
    assert args.approx_expm is True


class TestRepOptions:
  """Test RepOptions enum."""

  def test_rep_options_values(self):
    """Test RepOptions enum has expected values."""
    assert hasattr(RepOptions, "SLATERDET_ROT")
    assert hasattr(RepOptions, "SLATERDET")
    assert hasattr(RepOptions, "DIAG")
    assert hasattr(RepOptions, "SLATERDETC")
    assert hasattr(RepOptions, "SLATERDET_ROTC")

  def test_rep_options_unique(self):
    """Test RepOptions enum values are unique."""
    values = [opt.value for opt in RepOptions]
    assert len(values) == len(set(values))


class TestSlaterType:
  """Test SlaterType enum."""

  def test_slater_type_values(self):
    """Test SlaterType enum has expected values."""
    assert hasattr(SlaterType, "COLLINEAR")
    assert hasattr(SlaterType, "UHF")
    assert hasattr(SlaterType, "NONCOLLINEAR")
    assert hasattr(SlaterType, "GHF")
    assert hasattr(SlaterType, "CLOSED")
    assert hasattr(SlaterType, "RHF")

  def test_slater_type_aliases(self):
    """Test SlaterType aliases."""
    # UHF and COLLINEAR should be same
    assert SlaterType.UHF == SlaterType.COLLINEAR
    # GHF and NONCOLLINEAR should be same
    assert SlaterType.GHF == SlaterType.NONCOLLINEAR
    # RHF and CLOSED should be same
    assert SlaterType.RHF == SlaterType.CLOSED

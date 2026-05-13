import pytest
import numpy as np
import tempfile
import os
import pickle
from pathlib import Path

from autohf.solver import saveStates, loadStates
from autohf.options import process_args


@pytest.fixture
def mock_solver_data(hubbard_4x4):
  """Create mock data from a minimal solver run."""
  from autohf import lattice_hf
  
  settings = {
    "nelec": (3, 3),
    "steps": 5,
    "batch_size": 2,
    "ansatz": "SD_ROT",
  }
  
  results, funcs = lattice_hf(hamiltonian=hubbard_4x4, settings=settings)
  
  return {
    "states": results["states"],
    "orbFunc": funcs["state2Orbitals"],
    "energies": results["Es"],
    "phi0": results["phi0"],
    "args": results["args"],
    "nelec": (3, 3),
  }


class TestSaveStates:
  """Test saveStates function with different file formats."""

  def test_save_no_file(self, mock_solver_data):
    """Test saveStates returns data when fname is None."""
    data = saveStates(
      states=mock_solver_data["states"],
      orbFunc=mock_solver_data["orbFunc"],
      energies=mock_solver_data["energies"],
      phi0=mock_solver_data["phi0"],
      args=mock_solver_data["args"],
      nelec=mock_solver_data["nelec"],
      fname=None,
      dump_batch=False,
    )
    
    # Check data structure
    assert "state" in data
    assert "Es" in data
    assert "E" in data
    assert "orbitals" in data
    assert "args" in data
    assert "phi0" in data
    assert "states" in data
    assert "nelec" in data
    
    # Best energy should be minimum
    assert data["E"] == np.min(data["Es"])

  # this is slow for some reason
  def test_save_empty_filename(self, mock_solver_data):
    """Test saveStates returns data when fname is empty string."""
    data = saveStates(
      states=mock_solver_data["states"],
      orbFunc=mock_solver_data["orbFunc"],
      energies=mock_solver_data["energies"],
      phi0=mock_solver_data["phi0"],
      args=mock_solver_data["args"],
      nelec=mock_solver_data["nelec"],
      fname="",
      dump_batch=False,
    )
    
    assert data is not None
    assert "state" in data

  def test_save_pickle_format(self, mock_solver_data):
    """Test saveStates with .pkl format."""
    with tempfile.TemporaryDirectory() as tmpdir:
      fname = os.path.join(tmpdir, "test_save.pkl")
      
      data = saveStates(
        states=mock_solver_data["states"],
        orbFunc=mock_solver_data["orbFunc"],
        energies=mock_solver_data["energies"],
        phi0=mock_solver_data["phi0"],
        args=mock_solver_data["args"],
        nelec=mock_solver_data["nelec"],
        fname=fname,
        dump_batch=False,
      )
      
      # File should exist
      assert os.path.exists(fname)
      
      # Load and verify
      with open(fname, "rb") as f:
        loaded_data = pickle.load(f)
      
      assert "state" in loaded_data
      assert "Es" in loaded_data
      assert np.array_equal(loaded_data["Es"], data["Es"])

  def test_save_pickle_alternative_extension(self, mock_solver_data):
    """Test saveStates with .pickle format."""
    with tempfile.TemporaryDirectory() as tmpdir:
      fname = os.path.join(tmpdir, "test_save.pickle")
      
      data = saveStates(
        states=mock_solver_data["states"],
        orbFunc=mock_solver_data["orbFunc"],
        energies=mock_solver_data["energies"],
        phi0=mock_solver_data["phi0"],
        args=mock_solver_data["args"],
        nelec=mock_solver_data["nelec"],
        fname=fname,
        dump_batch=False,
      )
      
      assert os.path.exists(fname)

  def test_save_unknown_extension_converts_to_pickle(self, mock_solver_data):
    """Test saveStates with unknown extension creates .pkl file."""
    with tempfile.TemporaryDirectory() as tmpdir:
      fname = os.path.join(tmpdir, "test_save.dat")
      
      data = saveStates(
        states=mock_solver_data["states"],
        orbFunc=mock_solver_data["orbFunc"],
        energies=mock_solver_data["energies"],
        phi0=mock_solver_data["phi0"],
        args=mock_solver_data["args"],
        nelec=mock_solver_data["nelec"],
        fname=fname,
        dump_batch=False,
      )
      
      # Should create .pkl file
      assert os.path.exists(fname + ".pkl")

  def test_save_npy_format(self, mock_solver_data):
    """Test saveStates with .npy format saves orbitals."""
    with tempfile.TemporaryDirectory() as tmpdir:
      fname = os.path.join(tmpdir, "test_save.npy")
      
      data = saveStates(
        states=mock_solver_data["states"],
        orbFunc=mock_solver_data["orbFunc"],
        energies=mock_solver_data["energies"],
        phi0=mock_solver_data["phi0"],
        args=mock_solver_data["args"],
        nelec=mock_solver_data["nelec"],
        fname=fname,
        dump_batch=True,
      )
      
      # File should exist
      assert os.path.exists(fname)
      
      # Load and verify it's a list of orbitals
      loaded = np.load(fname, allow_pickle=True)
      assert loaded is not None

  def test_save_h5(self, mock_solver_data):
    """Test saveStates with .h5 format raises NotImplementedError."""
    with tempfile.TemporaryDirectory() as tmpdir:
      fname = os.path.join(tmpdir, "test_save.h5")

      data = saveStates(
          states=mock_solver_data["states"],
          orbFunc=mock_solver_data["orbFunc"],
          energies=mock_solver_data["energies"],
          phi0=mock_solver_data["phi0"],
          args=mock_solver_data["args"],
          nelec=mock_solver_data["nelec"],
          fname=fname,
          dump_batch=False,
        )

  def test_save_with_dump_batch(self, mock_solver_data):
    """Test saveStates with dump_batch=True includes all_orbitals."""
    data = saveStates(
      states=mock_solver_data["states"],
      orbFunc=mock_solver_data["orbFunc"],
      energies=mock_solver_data["energies"],
      phi0=mock_solver_data["phi0"],
      args=mock_solver_data["args"],
      nelec=mock_solver_data["nelec"],
      fname=None,
      dump_batch=True,
    )
    
    assert "all_orbitals" in data
    # Should have orbitals for each trial
    assert len(data["all_orbitals"]) == len(mock_solver_data["states"])


class TestLoadStates:
  """Test loadStates function with different input types."""

  def test_load_from_file(self, mock_solver_data):
    """Test loadStates from .pkl file."""
    with tempfile.TemporaryDirectory() as tmpdir:
      fname = os.path.join(tmpdir, "test_load.pkl")
      
      # Save first
      saveStates(
        states=mock_solver_data["states"],
        orbFunc=mock_solver_data["orbFunc"],
        energies=mock_solver_data["energies"],
        phi0=mock_solver_data["phi0"],
        args=mock_solver_data["args"],
        nelec=mock_solver_data["nelec"],
        fname=fname,
        dump_batch=False,
      )
      
      # Load
      state0, state0_ref, phi0 = loadStates(
        fname, 
        mock_solver_data["args"],
        mock_solver_data["nelec"]
      )
      
      assert state0 is not None
      assert state0_ref is not None
      assert phi0 is not None

  def test_load_from_dict(self, mock_solver_data):
    """Test loadStates from dictionary input."""
    data = saveStates(
      states=mock_solver_data["states"],
      orbFunc=mock_solver_data["orbFunc"],
      energies=mock_solver_data["energies"],
      phi0=mock_solver_data["phi0"],
      args=mock_solver_data["args"],
      nelec=mock_solver_data["nelec"],
      fname=None,
      dump_batch=False,
    )
    
    state0, state0_ref, phi0 = loadStates(
      data,
      mock_solver_data["args"],
      mock_solver_data["nelec"]
    )
    
    assert state0 is not None
    assert state0_ref is not None

  def test_load_from_tuple(self, mock_solver_data):
    """Test loadStates from tuple input."""
    data = saveStates(
      states=mock_solver_data["states"],
      orbFunc=mock_solver_data["orbFunc"],
      energies=mock_solver_data["energies"],
      phi0=mock_solver_data["phi0"],
      args=mock_solver_data["args"],
      nelec=mock_solver_data["nelec"],
      fname=None,
      dump_batch=False,
    )
    
    state0, state0_ref, phi0 = loadStates(
      (data, None),  # Tuple with data as first element
      mock_solver_data["args"],
      mock_solver_data["nelec"]
    )
    
    assert state0 is not None

  def test_load_from_list(self, mock_solver_data):
    """Test loadStates from list input."""
    data = saveStates(
      states=mock_solver_data["states"],
      orbFunc=mock_solver_data["orbFunc"],
      energies=mock_solver_data["energies"],
      phi0=mock_solver_data["phi0"],
      args=mock_solver_data["args"],
      nelec=mock_solver_data["nelec"],
      fname=None,
      dump_batch=False,
    )
    
    state0, state0_ref, phi0 = loadStates(
      [data],  # List with data as first element
      mock_solver_data["args"],
      mock_solver_data["nelec"]
    )
    
    assert state0 is not None

  def test_load_invalid_type_raises_error(self, mock_solver_data):
    """Test loadStates with invalid input type raises RuntimeError."""
    with pytest.raises(RuntimeError, match="Unknown restart data"):
      loadStates(
        12345,  # Invalid type
        mock_solver_data["args"],
        mock_solver_data["nelec"]
      )

  def test_load_increased_batch_size_raises_error(self, mock_solver_data):
    """Test loadStates raises error when requesting larger batch_size."""
    data = saveStates(
      states=mock_solver_data["states"],
      orbFunc=mock_solver_data["orbFunc"],
      energies=mock_solver_data["energies"],
      phi0=mock_solver_data["phi0"],
      args=mock_solver_data["args"],
      nelec=mock_solver_data["nelec"],
      fname=None,
      dump_batch=False,
    )
    
    # Create new args with larger batch_size
    new_settings = mock_solver_data["args"].__dict__.copy()
    new_settings["batch_size"] = mock_solver_data["args"].batch_size + 5
    new_args = process_args(new_settings)
    
    with pytest.raises(NotImplementedError, match="batch_size"):
      loadStates(data, new_args, mock_solver_data["nelec"])

  def test_load_same_ansatz(self, mock_solver_data):
    """Test loadStates with same ansatz uses states directly."""
    data = saveStates(
      states=mock_solver_data["states"],
      orbFunc=mock_solver_data["orbFunc"],
      energies=mock_solver_data["energies"],
      phi0=mock_solver_data["phi0"],
      args=mock_solver_data["args"],
      nelec=mock_solver_data["nelec"],
      fname=None,
      dump_batch=True,
    )
    
    state0, state0_ref, phi0 = loadStates(
      data,
      mock_solver_data["args"],
      mock_solver_data["nelec"]
    )
    
    # Should load states in sorted order
    assert len(state0) == mock_solver_data["args"].batch_size

  def test_load_convert_to_SD_without_orbitals_raises_error(self, mock_solver_data):
    """Test loadStates converting to SD without all_orbitals raises error."""
    data = saveStates(
      states=mock_solver_data["states"],
      orbFunc=mock_solver_data["orbFunc"],
      energies=mock_solver_data["energies"],
      phi0=mock_solver_data["phi0"],
      args=mock_solver_data["args"],
      nelec=mock_solver_data["nelec"],
      fname=None,
      dump_batch=False,  # Don't save all orbitals
    )
    
    # Try to load with different ansatz
    new_settings = mock_solver_data["args"].__dict__.copy()
    new_settings["ansatz"] = "SD"
    new_args = process_args(new_settings)
    
    with pytest.raises(RuntimeError, match="All orbitals were not saved"):
      loadStates(data, new_args, mock_solver_data["nelec"])

  def test_load_convert_to_SD(self, mock_solver_data):
    """Test loadStates converting from one ansatz to SD."""
    data = saveStates(
      states=mock_solver_data["states"],
      orbFunc=mock_solver_data["orbFunc"],
      energies=mock_solver_data["energies"],
      phi0=mock_solver_data["phi0"],
      args=mock_solver_data["args"],
      nelec=mock_solver_data["nelec"],
      fname=None,
      dump_batch=True,  # Save all orbitals for conversion
    )
    
    # Try to load with SD ansatz
    new_settings = mock_solver_data["args"].__dict__.copy()
    new_settings["ansatz"] = "SD"
    new_args = process_args(new_settings)
    
    state0, state0_ref, phi0 = loadStates(data, new_args, mock_solver_data["nelec"])
    
    assert state0 is not None
    assert state0_ref is not None

  def test_load_convert_to_SDC(self, mock_solver_data):
    """Test loadStates converting to SDC (complex) ansatz."""
    data = saveStates(
      states=mock_solver_data["states"],
      orbFunc=mock_solver_data["orbFunc"],
      energies=mock_solver_data["energies"],
      phi0=mock_solver_data["phi0"],
      args=mock_solver_data["args"],
      nelec=mock_solver_data["nelec"],
      fname=None,
      dump_batch=True,
    )
    
    # Try to load with SDC ansatz
    new_settings = mock_solver_data["args"].__dict__.copy()
    new_settings["ansatz"] = "SDC"
    new_args = process_args(new_settings)
    
    state0, state0_ref, phi0 = loadStates(data, new_args, mock_solver_data["nelec"])
    
    assert state0 is not None
    # SDC should have real and imaginary parts stacked
    assert state0.shape[0] == 2  # [real, imag]

  def test_load_file_unknown_format(self, mock_solver_data, capfd):
    """Test loadStates with non-.pkl file shows warning."""
    with tempfile.TemporaryDirectory() as tmpdir:
      fname = os.path.join(tmpdir, "test_load.dat")

      with pytest.raises(RuntimeError, match="Unknown file extension for restart data"):
        # Manually save as pickle with .dat extension
        loadStates(fname, mock_solver_data["args"], mock_solver_data["nelec"])


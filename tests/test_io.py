import pytest
import tempfile
import os
from autohf.io import read_input_params


class TestIO:
  """Test input/output functions."""

  def test_read_input_params_basic(self):
    """Test reading a basic TOML file."""
    toml_content = """
[hamiltonian]
U = 2.5

[lattice]
L1 = 4
L2 = 4
type = "square"

[misc_params]
nelec = [7, 7]

[autohf]
steps = 500
ansatz = "SD"
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
      f.write(toml_content)
      temp_file = f.name
    
    try:
      params = read_input_params(temp_file)
      
      # Verify structure
      assert 'hamiltonian' in params
      assert 'lattice' in params
      assert 'misc_params' in params
      assert 'autohf' in params
      
      # Verify values
      assert params['hamiltonian']['U'] == 2.5
      assert params['lattice']['L1'] == 4
      assert params['lattice']['L2'] == 4
      assert params['lattice']['type'] == 'square'
      assert params['misc_params']['nelec'] == [7, 7]
      assert params['autohf']['steps'] == 500
      assert params['autohf']['ansatz'] == 'SD'
      
    finally:
      os.unlink(temp_file)

  def test_read_input_params_full_config(self):
    """Test reading a complete configuration file."""
    toml_content = """
[hamiltonian]
U = 2.5
U1 = 1.5
U2 = 1.0
J = 0.5

[lattice]
L1 = 4
L2 = 4
boundary1 = "PBC"
boundary2 = "PBC"
type = "square"

[misc_params]
nelec = [7, 7]

[autohf]
steps = 500
opt_method = "lbfgs"
ansatz = "SD_ROT"
riemannian = true
batch_size = 8
gpu = false
noncollinear = true
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
      f.write(toml_content)
      temp_file = f.name
    
    try:
      params = read_input_params(temp_file)
      
      # Verify hamiltonian parameters
      assert params['hamiltonian']['U'] == 2.5
      assert params['hamiltonian']['U1'] == 1.5
      assert params['hamiltonian']['U2'] == 1.0
      assert params['hamiltonian']['J'] == 0.5
      
      # Verify autohf parameters
      assert params['autohf']['opt_method'] == 'lbfgs'
      assert params['autohf']['riemannian'] is True
      assert params['autohf']['batch_size'] == 8
      assert params['autohf']['gpu'] is False
      assert params['autohf']['noncollinear'] is True
      
    finally:
      os.unlink(temp_file)

  def test_read_input_params_empty_sections(self):
    """Test reading a file with minimal content."""
    toml_content = """
[autohf]
steps = 100
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
      f.write(toml_content)
      temp_file = f.name
    
    try:
      params = read_input_params(temp_file)
      
      assert 'autohf' in params
      assert params['autohf']['steps'] == 100
      
    finally:
      os.unlink(temp_file)

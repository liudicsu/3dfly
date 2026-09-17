"""Tests for realistic fly model."""

import pytest
from pathlib import Path


def test_fly_meshes_exist():
    """Test that fly mesh files exist."""
    assets_dir = Path(__file__).parent.parent / "assets" / "fly_meshes"
    
    assert assets_dir.exists(), "Fly meshes directory should exist"
    
    required_meshes = [
        "thorax_body.obj",
        "head_body.obj",
        "abdomen_1_body.obj",
        "wing_left_membrane.obj",
        "wing_right_membrane.obj",
    ]
    
    for mesh_file in required_meshes:
        mesh_path = assets_dir / mesh_file
        assert mesh_path.exists(), f"Required mesh file {mesh_file} should exist"
        assert mesh_path.stat().st_size > 0, f"Mesh file {mesh_file} should not be empty"


def test_mjcf_generation():
    """Test MJCF XML generation with realistic meshes."""
    # Import directly to avoid OpenGL issues
    import sys
    from pathlib import Path
    
    # Read and execute mjcf module
    mjcf_path = Path(__file__).parent.parent / "threedfly" / "sim" / "mjcf.py"
    with open(mjcf_path, 'r') as f:
        mjcf_code = f.read()
    
    # Execute to define the function
    namespace = {'__file__': str(mjcf_path)}
    exec(mjcf_code, namespace)
    get_fly_mjcf = namespace['get_fly_mjcf']
    
    # Get MJCF XML
    mjcf_xml = get_fly_mjcf()
    
    assert len(mjcf_xml) > 1000, "MJCF XML should be substantial"
    
    # Check that mesh references are present
    required_content = [
        "meshdir=",
        "thorax_mesh",
        "head_mesh",
        "abdomen_mesh",
        "wing_left_mesh",
        "wing_right_mesh",
        'type="mesh"',
    ]
    
    for content in required_content:
        assert content in mjcf_xml, f"MJCF should contain '{content}'"


def test_mjcf_has_cameras():
    """Test that MJCF still has camera definitions."""
    import sys
    from pathlib import Path
    
    # Read and execute mjcf module
    mjcf_path = Path(__file__).parent.parent / "threedfly" / "sim" / "mjcf.py"
    with open(mjcf_path, 'r') as f:
        mjcf_code = f.read()
    
    namespace = {'__file__': str(mjcf_path)}
    exec(mjcf_code, namespace)
    get_fly_mjcf = namespace['get_fly_mjcf']
    
    mjcf_xml = get_fly_mjcf()
    
    # Ensure cameras are still present
    assert "left_eye" in mjcf_xml
    assert "right_eye" in mjcf_xml
    assert 'name="left_eye"' in mjcf_xml
    assert 'name="right_eye"' in mjcf_xml


def test_fly_license_exists():
    """Test that fly model license file exists."""
    assets_dir = Path(__file__).parent.parent / "assets" / "fly_meshes"
    license_file = assets_dir / "FLYBODY_LICENSE"
    
    assert license_file.exists(), "Flybody license file should exist"
    assert license_file.stat().st_size > 0, "License file should not be empty"


def test_fly_readme_exists():
    """Test that fly model README exists."""
    assets_dir = Path(__file__).parent.parent / "assets" / "fly_meshes"
    readme_file = assets_dir / "README.md"
    
    assert readme_file.exists(), "Fly meshes README should exist"
    assert readme_file.stat().st_size > 0, "README should not be empty"
    
    # Check that README contains key information
    with open(readme_file, 'r') as f:
        readme_content = f.read()
    
    assert "flybody" in readme_content.lower()
    assert "Apache" in readme_content
    assert "citation" in readme_content.lower()

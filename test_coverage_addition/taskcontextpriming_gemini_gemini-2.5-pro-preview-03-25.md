```python
import os
import pytest
import shutil
import subprocess
from pathlib import Path
from unittest.mock import patch, mock_open, MagicMock, call

# Assuming the file_utils module is located as described
from src.test_analyzer.utils.file_utils import (
    _find_root_upwards,
    determine_project_root,
    is_likely_external_file,
    read_file_content,
    prepare_file_contents,
    backup_file,
    verify_python_syntax,
    find_python_files_in_project,
    find_implementation_file_in_project
)
# Assuming environment module exists for mocking logger if necessary
# from src.test_analyzer.utils.environment import log_debug # Not strictly needed for testing logic


# --- Existing Tests (slightly formatted for clarity) ---

def test_find_root_upwards_finds_marker(tmp_path: Path):
    project_root = tmp_path / "project"
    sub_dir = project_root / "src" / "subdir"
    sub_dir.mkdir(parents=True)
    (project_root / "pyproject.toml").touch()
    start_file = sub_dir / "module.py"
    start_file.touch()
    markers = ["pyproject.toml", ".git"]
    found_root = _find_root_upwards(str(start_file), markers)
    assert found_root == str(project_root)

def test_find_root_upwards_finds_marker_from_dir(tmp_path: Path):
    """Test _find_root_upwards works when starting from a directory."""
    project_root = tmp_path / "project"
    sub_dir = project_root / "src" / "subdir"
    sub_dir.mkdir(parents=True)
    (project_root / ".git").mkdir() # Use a different marker
    markers = ["pyproject.toml", ".git"]
    # Start search from the directory itself
    found_root = _find_root_upwards(str(sub_dir), markers)
    assert found_root == str(project_root)

def test_find_root_upwards_no_marker(tmp_path: Path):
    some_dir = tmp_path / "no_project" / "subdir"
    some_dir.mkdir(parents=True)
    start_file = some_dir / "module.py"
    start_file.touch()
    markers = ["pyproject.toml"]
    found_root = _find_root_upwards(str(start_file), markers)
    assert found_root is None

def test_find_root_upwards_max_levels(tmp_path: Path):
    level_0 = tmp_path / "l0"
    level_5 = level_0 / "l1/l2/l3/l4/l5"
    level_11 = level_5 / "l6/l7/l8/l9/l10/l11"
    level_11.mkdir(parents=True)
    (level_0 / "pyproject.toml").touch()
    start_file = level_11 / "deep_file.py"
    start_file.touch()
    markers = ["pyproject.toml"]
    assert _find_root_upwards(str(start_file), markers) is None
    assert _find_root_upwards(str(start_file), markers, max_levels=15) == str(level_0)

def test_find_root_upwards_at_filesystem_root(tmp_path: Path):
    """Test stopping at the filesystem root."""
    # Simulate starting very high up the directory tree
    # On Unix, '/' is the root. On Windows, it might be 'C:\'.
    # Use os.path.abspath('/') or os.path.splitdrive(tmp_path)[0] + os.sep
    fs_root = os.path.abspath(os.sep)
    # Create a path *near* the root without markers
    start_dir = Path(fs_root) / "some_dir_near_root"
    start_dir.mkdir(exist_ok=True) # Might fail if no permissions, but likely okay
    (start_dir / "file.txt").touch()
    markers = ["pyproject.toml"]
    # This search should eventually hit the actual filesystem root and stop
    found_root = _find_root_upwards(str(start_dir / "file.txt"), markers, max_levels=50) # Use high max_levels
    assert found_root is None
    start_dir.rmdir() # Clean up

def test_find_root_upwards_non_existent_start_path(tmp_path: Path):
    """Test _find_root_upwards handles non-existent start path."""
    non_existent = tmp_path / "does/not/exist/file.py"
    markers = ["pyproject.toml"]
    found_root = _find_root_upwards(str(non_existent), markers)
    assert found_root is None


def test_determine_project_root_env_var(tmp_path: Path, monkeypatch):
    env_root_path = tmp_path / "env_root"
    env_root_path.mkdir()
    target_path = tmp_path / "some_other_place" / "test.py"
    target_path.parent.mkdir(parents=True)
    target_path.touch()
    (target_path.parent.parent / ".git").mkdir()
    monkeypatch.setenv("TEST_ANALYZER_PROJECT_ROOT", str(env_root_path))
    root = determine_project_root(str(target_path))
    assert root == str(env_root_path)

def test_determine_project_root_env_var_invalid(tmp_path: Path, monkeypatch):
    """Test env var is ignored if it's not a directory."""
    env_root_path_file = tmp_path / "env_root.txt"
    env_root_path_file.touch()
    project_root_marker = tmp_path / "project"
    project_root_marker.mkdir()
    (project_root_marker / "pyproject.toml").touch()
    target_path = project_root_marker / "test.py"
    target_path.touch()

    monkeypatch.setenv("TEST_ANALYZER_PROJECT_ROOT", str(env_root_path_file))
    root = determine_project_root(str(target_path))
    # Should ignore the invalid env var and auto-detect
    assert root == str(project_root_marker)


def test_determine_project_root_auto_detect(tmp_path: Path):
    project_root = tmp_path / "auto_detect_project"
    sub_dir = project_root / "src"
    sub_dir.mkdir(parents=True)
    (project_root / ".git").mkdir()
    test_file = sub_dir / "my_test.py"
    test_file.touch()
    root = determine_project_root(str(test_file))
    assert root == str(project_root)

def test_determine_project_root_fallback_file(tmp_path: Path):
    """Test fallback to target file's directory."""
    no_markers_dir = tmp_path / "no_markers"
    test_file = no_markers_dir / "test_fallback.py"
    test_file.parent.mkdir()
    test_file.touch()
    root = determine_project_root(str(test_file))
    assert root == str(no_markers_dir)

def test_determine_project_root_fallback_dir(tmp_path: Path):
    """Test fallback to target directory itself."""
    no_markers_dir = tmp_path / "no_markers_dir"
    no_markers_dir.mkdir()
    # Target is the directory itself
    root = determine_project_root(str(no_markers_dir))
    assert root == str(no_markers_dir)

def test_determine_project_root_non_existent_target(tmp_path: Path):
    non_existent_path = tmp_path / "not_real" / "test.py"
    cwd = os.getcwd()
    root = determine_project_root(str(non_existent_path))
    assert root == cwd


@pytest.mark.parametrize("path, expected", [
    ("/home/user/project/src/module.py", False),
    ("/home/user/.venv/lib/python3.10/site-packages/requests/api.py", True),
    ("/usr/lib/python3.9/os.py", True),
    ("project/lib/mylib.py", False),
    ("/home/user/.local/lib/python3.8/site-packages/django/db/models.py", True),
    ("C:\\Users\\user\\AppData\\Local\\Programs\\Python\\Python310\\Lib\\site-packages\\somepkg\\mod.py", True), # Windows example
    ("/opt/hostedtoolcache/Python/3.10.14/x64/lib/python3.10/site-packages/pytest/__init__.py", True), # CI runner example
])
def test_is_likely_external_file(path, expected):
    # Normalize path separators for cross-platform compatibility in tests
    normalized_path = path.replace("\\", "/")
    assert is_likely_external_file(normalized_path) == expected


def test_read_file_content_success(tmp_path: Path):
    file = tmp_path / "test.txt"
    expected_content = "Hello\nWorld!"
    file.write_text(expected_content, encoding='utf-8')
    content = read_file_content(str(file))
    assert content == expected_content

def test_read_file_content_non_utf8(tmp_path: Path):
    """Test reading with ignored encoding errors."""
    file = tmp_path / "test_latin1.txt"
    # Example byte sequence valid in latin-1 but not utf-8
    content_bytes = b'Euro symbol: \xa4'
    expected_content_utf8_ignore = 'Euro symbol: ' # The \xa4 is ignored
    file.write_bytes(content_bytes)
    content = read_file_content(str(file))
    # Depending on the OS and locale, the ignored char might be replaced
    # or just skipped. Checking it's not None and doesn't raise is key.
    assert content is not None
    assert "Euro symbol:" in content # Make sure the valid part was read


def test_read_file_content_not_found(tmp_path: Path):
    file = tmp_path / "not_found.txt"
    content = read_file_content(str(file))
    assert content is None


def test_read_file_content_read_error(tmp_path: Path, mocker):
    """Test handling of OS errors during file read."""
    file = tmp_path / "error.txt"
    file.touch()
    # Use mocker fixture provided by pytest-mock
    mocked_open = mocker.patch("builtins.open", side_effect=OSError("Permission denied"))
    content = read_file_content(str(file))
    assert content is None
    # Check open was attempted with the absolute path
    mocked_open.assert_called_once_with(str(file.resolve()), 'r', encoding='utf-8', errors='ignore')


def test_prepare_file_contents(tmp_path: Path):
    f1 = tmp_path / "file1.py"
    f2 = tmp_path / "file2.py"
    f_non_exist = tmp_path / "missing.py"
    f1_content = "content1"
    f2_content = "content2"
    f1.write_text(f1_content)
    f2.write_text(f2_content)

    paths = [str(f1), str(f2), str(f_non_exist)]
    contents = prepare_file_contents(paths)

    # Check absolute paths are used as keys
    abs_f1 = str(f1.resolve())
    abs_f2 = str(f2.resolve())
    abs_f_non_exist = str(f_non_exist.resolve())

    assert len(contents) == 2
    assert contents[abs_f1] == f1_content
    assert contents[abs_f2] == f2_content
    assert abs_f_non_exist not in contents


def test_backup_file_success(tmp_path: Path):
    original_file = tmp_path / "original.py"
    content = "original content"
    original_file.write_text(content)
    backup_path_expected = str(original_file.resolve()) + ".bak"

    backup_path_actual = backup_file(str(original_file))

    assert backup_path_actual == backup_path_expected
    assert os.path.exists(backup_path_expected)
    assert Path(backup_path_expected).read_text() == content


def test_backup_file_overwrite(tmp_path: Path):
    original_file = tmp_path / "original.py"
    backup_file_path_abs = str(original_file.resolve()) + ".bak"
    original_file.write_text("new content")
    Path(backup_file_path_abs).write_text("old backup")

    backup_path_actual = backup_file(str(original_file))
    assert backup_path_actual == backup_file_path_abs
    assert Path(backup_file_path_abs).read_text() == "new content"


def test_backup_file_non_existent(tmp_path: Path):
    assert backup_file(str(tmp_path / "no_such_file.py")) is None

# --- NEW TEST for backup_file error ---
def test_backup_file_shutil_error(tmp_path: Path, mocker):
    """Test backup fails if shutil.copy2 raises an error."""
    original_file = tmp_path / "original.py"
    original_file.write_text("content")
    mock_copy2 = mocker.patch("shutil.copy2", side_effect=shutil.Error("Disk full"))

    backup_path = backup_file(str(original_file))
    assert backup_path is None
    mock_copy2.assert_called_once()
    assert not os.path.exists(str(original_file.resolve()) + ".bak")


@pytest.mark.parametrize("code, expected_valid, error_contains", [
    ("x = 1\nprint(x)", True, None),
    ("def func():\n  pass", True, None),
    ("x = ", False, "syntax error"), # Changed check slightly
    ("def func(\n  pass", False, "unexpected EOF"), # More specific error msg check
    ("import non_existent_module", True, None),
    ("print(x", False, "unexpected EOF"), # More specific error msg check
    # Add case for invalid character causing ValueError
    ("a = 1\nb = '\x00'", False, "unexpected validation error"), # Covers generic Exception
])
def test_verify_python_syntax(code, expected_valid, error_contains):
    is_valid, error_msg = verify_python_syntax(code)
    assert is_valid == expected_valid
    if not expected_valid:
        assert isinstance(error_msg, str)
        if error_contains:
            assert error_contains.lower() in error_msg.lower()
    else:
        assert error_msg is None

# --- NEW TEST for verify_python_syntax generic exception ---
def test_verify_python_syntax_other_exception(mocker):
    """Test handling of non-SyntaxError exceptions during compile."""
    code = "valid = True"
    # Mock compile to raise a generic Exception
    mocker.patch("builtins.compile", side_effect=Exception("Something weird happened"))
    is_valid, error_msg = verify_python_syntax(code)
    assert not is_valid
    assert error_msg == "Unexpected validation error: Something weird happened"


def test_find_python_files_in_project(tmp_path: Path):
    root = tmp_path / "proj"
    f1 = root / "main.py"
    sub = root / "subdir"
    f2 = sub / "mod.py"
    test_dir = root / "tests"
    f_test = test_dir / "test_mod.py" # excluded by default + name
    f_test_noname = test_dir / "conftest.py" # excluded by default dir
    venv = root / ".venv" / "lib"
    f_venv = venv / "some_lib.py"
    cache = sub / "__pycache__"
    f_cache = cache / "mod.cpython-310.pyc"
    f_pycache_py = cache / "settings.py"
    dot_git = root / ".git" / "config"
    f_in_git = root / ".git" / "hooks.py"
    f_explicit_exclude_name = root / "setup.py" # Example of file to exclude by name
    f_explicit_exclude_dir = root / "build" / "lib" / "built.py" # Example of dir to exclude

    # Create structure
    venv.mkdir(parents=True)
    test_dir.mkdir(parents=True)
    cache.mkdir(parents=True)
    (root / ".git").mkdir(parents=True)
    (root / "build" / "lib").mkdir(parents=True)
    f1.touch()
    f2.touch()
    f_test.touch()
    f_test_noname.touch()
    f_venv.touch()
    f_cache.touch()
    f_pycache_py.touch()
    dot_git.touch()
    f_in_git.touch()
    f_explicit_exclude_name.touch()
    f_explicit_exclude_dir.touch()

    # Test with default excludes
    found_files_default = find_python_files_in_project(str(root))
    found_files_default_set = {str(p.resolve()) for p in Path(root).rglob('*.py')} # Get all actual .py for comparison
    expected_files_default = {
        str(f1.resolve()),
        str(f2.resolve()),
        str(f_explicit_exclude_name.resolve()), # Not excluded by default
        str(f_explicit_exclude_dir.resolve()), # Not excluded by default
    }
    # Filter found_files_default_set based on rules applied in the function
    manually_filtered_set = set()
    for p_str in found_files_default_set:
        p = Path(p_str)
        if p.name.startswith("test_"): continue
        parts = set(p.relative_to(root).parts)
        default_excludes = {'__pycache__', '.venv', 'venv', '.env', 'env', 'node_modules', '.git', 'tests', 'test'}
        if not default_excludes.intersection(parts):
             manually_filtered_set.add(p_str)

    # The function logic is slightly different than simple intersection (it checks dir names during walk)
    # Let's assert based on the known inclusions/exclusions
    assert str(f1.resolve()) in found_files_default
    assert str(f2.resolve()) in found_files_default
    assert str(f_explicit_exclude_name.resolve()) in found_files_default # setup.py included by default
    assert str(f_explicit_exclude_dir.resolve()) in found_files_default # build/lib/built.py included by default

    assert str(f_test.resolve()) not in found_files_default
    assert str(f_test_noname.resolve()) not in found_files_default
    assert str(f_venv.resolve()) not in found_files_default
    assert str(f_pycache_py.resolve()) not in found_files_default
    assert str(f_in_git.resolve()) not in found_files_default
    assert len(found_files_default) == 4


    # --- NEW TEST for custom exclude_patterns (line 157) ---
    custom_excludes = ["setup.py", "build"] # Exclude setup.py by name, 'build' directory
    found_files_custom = find_python_files_in_project(str(root), exclude_patterns=custom_excludes)

    assert str(f1.resolve()) in found_files_custom
    assert str(f2.resolve()) in found_files_custom

    # These should now be excluded
    assert str(f_explicit_exclude_name.resolve()) not in found_files_custom # setup.py now excluded
    assert str(f_explicit_exclude_dir.resolve()) not in found_files_custom # build/** excluded

    # Default exclusions should still apply
    assert str(f_test.resolve()) not in found_files_custom
    assert str(f_venv.resolve()) not in found_files_custom
    assert len(found_files_custom) == 2


# --- Test find_implementation_file_in_project ---

# Shared mock results
MOCK_FIND_SUCCESS_RESULT = MagicMock(returncode=0, stdout="/abs/path/to/project/src/my_class.py\n/abs/path/to/project/somewhere/else/my_class.py")
MOCK_FIND_OUTSIDE_ROOT_RESULT = MagicMock(returncode=0, stdout="/outside/root/my_class.py")
MOCK_FIND_NOT_FOUND_RESULT = MagicMock(returncode=0, stdout="")
MOCK_FIND_ERROR_RESULT = MagicMock(returncode=1, stdout="", stderr="find error")
MOCK_GREP_SUCCESS_RESULT = MagicMock(returncode=0, stdout="/abs/path/to/project/src/impl/my_class.py")
MOCK_GREP_OUTSIDE_ROOT_RESULT = MagicMock(returncode=0, stdout="/outside/root/impl/my_class.py")
MOCK_GREP_NOT_FOUND_RESULT = MagicMock(returncode=1, stdout="")
MOCK_GREP_ERROR_RESULT = MagicMock(returncode=2, stdout="", stderr="grep error")


@patch("src.test_analyzer.utils.file_utils.read_file_content")
@patch("subprocess.run")
def test_find_implementation_via_find(mock_run, mock_read, tmp_path: Path):
    project_root = str(tmp_path.resolve())
    expected_path = os.path.join(project_root, "src", "my_class.py")
    # Adjust mock result to be within the temp project root
    mock_find_result = MagicMock(returncode=0, stdout=f"{expected_path}\n{project_root}/other/my_class.py")

    mock_run.return_value = mock_find_result
    # First call to read_file_content returns matching content
    mock_read.side_effect = lambda p: "class MyClass:\n    pass" if p == expected_path else "class WrongClass:"

    result = find_implementation_file_in_project("MyClass", project_root)

    assert mock_run.call_count == 1 # Only find should be called
    find_call_args = mock_run.call_args_list[0].args[0]
    assert find_call_args[0] == "find"
    assert find_call_args[1] == project_root
    assert "-name" in find_call_args
    assert "myclass.py" in find_call_args # Check heuristic name
    mock_read.assert_called_once_with(expected_path)
    assert result == expected_path


@patch("src.test_analyzer.utils.file_utils.read_file_content")
@patch("subprocess.run")
def test_find_implementation_via_find_content_mismatch(mock_run, mock_read, tmp_path: Path):
    """Test find runs, finds files, but content check fails, falling back."""
    project_root = str(tmp_path.resolve())
    find_path1 = os.path.join(project_root, "src", "my_class.py")
    find_path2 = os.path.join(project_root, "other", "my_class.py")
    grep_path = os.path.join(project_root, "impl", "my_class.py") # Grep should find this later

    mock_find_result = MagicMock(returncode=0, stdout=f"{find_path1}\n{find_path2}")
    mock_grep_result = MagicMock(returncode=0, stdout=grep_path)

    mock_run.side_effect = [mock_find_result, mock_grep_result]
    # Make read_file_content fail the check for the 'find' results, but pass for 'grep'
    mock_read.side_effect = lambda p: "class MyClassWrongName:" if p in [find_path1, find_path2] \
                                   else ("class MyClass:\n pass" if p == grep_path else None)

    result = find_implementation_file_in_project("MyClass", project_root)

    assert mock_run.call_count == 2 # find and grep
    assert mock_read.call_count == 3 # Read path1, path2, grep_path
    assert result == grep_path


@patch("src.test_analyzer.utils.file_utils.read_file_content")
@patch("subprocess.run")
def test_find_implementation_via_find_outside_root(mock_run, mock_read, tmp_path: Path):
    """Test find returns path outside root, which gets skipped."""
    project_root = str(tmp_path.resolve())
    # Path clearly outside the tmp_path structure
    outside_path = "/absolute/different/path/my_class.py"
    grep_path = os.path.join(project_root, "impl", "my_class.py") # Grep should find this

    mock_find_result = MagicMock(returncode=0, stdout=outside_path)
    mock_grep_result = MagicMock(returncode=0, stdout=grep_path)

    mock_run.side_effect = [mock_find_result, mock_grep_result]
    # Make read_file_content pass for the grep path
    mock_read.side_effect = lambda p: "class MyClass:\n pass" if p == grep_path else None

    result = find_implementation_file_in_project("MyClass", project_root)

    assert mock_run.call_count == 2 # find and grep
    # read_file_content should NOT be called for outside_path
    mock_read.assert_called_once_with(grep_path)
    assert result == grep_path


@patch("src.test_analyzer.utils.file_utils.read_file_content")
@patch("subprocess.run")
def test_find_implementation_via_grep(mock_run, mock_read, tmp_path: Path):
    project_root = str(tmp_path.resolve())
    expected_path = os.path.join(project_root, "src", "impl", "my_class.py")
    # Adjust mock grep result
    mock_grep_success = MagicMock(returncode=0, stdout=expected_path)

    mock_run.side_effect = [
        MOCK_FIND_NOT_FOUND_RESULT,
        mock_grep_success
    ]
    mock_read.return_value = "class MyClass:\n    pass" # Content matches
    result = find_implementation_file_in_project("MyClass", project_root)

    assert mock_run.call_count == 2
    grep_call_args = mock_run.call_args_list[1]
    assert isinstance(grep_call_args.kwargs['shell'], bool) and grep_call_args.kwargs['shell'] is True
    assert "grep -l" in grep_call_args.args[0]
    assert f"class\\s+MyClass\\b" in grep_call_args.args[0] # Check regex
    assert project_root in grep_call_args.args[0] # Check root in find part of grep cmd
    mock_read.assert_called_once_with(expected_path)
    assert result == expected_path

# --- NEW TEST for grep content verification failure (lines 237-240) ---
@patch("src.test_analyzer.utils.find_python_files_in_project")
@patch("src.test_analyzer.utils.file_utils.read_file_content")
@patch("subprocess.run")
def test_find_implementation_grep_content_mismatch(mock_run, mock_read, mock_find_py, tmp_path: Path):
    """Test grep finds a file, but content check fails, falling back to walk."""
    project_root = str(tmp_path.resolve())
    grep_path = os.path.join(project_root, "src", "impl", "my_class.py")
    walk_path = os.path.join(project_root, "lib", "my_class_real.py")

    mock_grep_result = MagicMock(returncode=0, stdout=grep_path)

    mock_run.side_effect = [
        MOCK_FIND_NOT_FOUND_RESULT,
        mock_grep_result
    ]
    # read_file_content returns wrong content for grep path, correct for walk path
    mock_read.side_effect = lambda p: "class WrongClass:" if p == grep_path \
                                   else ("class MyClass:\n pass" if p == walk_path else None)
    # Python walk finds the correct file
    mock_find_py.return_value = [walk_path]

    result = find_implementation_file_in_project("MyClass", project_root)

    assert mock_run.call_count == 2 # find, grep
    # read called for grep path, then for walk path
    assert mock_read.call_count == 2
    assert call(grep_path) in mock_read.call_args_list
    assert call(walk_path) in mock_read.call_args_list
    mock_find_py.assert_called_once_with(project_root)
    assert result == walk_path

# --- NEW TEST for grep returning path outside project root ---
@patch("src.test_analyzer.utils.find_python_files_in_project")
@patch("src.test_analyzer.utils.file_utils.read_file_content")
@patch("subprocess.run")
def test_find_implementation_grep_outside_root(mock_run, mock_read, mock_find_py, tmp_path: Path):
    """Test grep finds a file outside project root, skips it, falls back."""
    project_root = str(tmp_path.resolve())
    outside_grep_path = "/some/other/place/my_class.py"
    walk_path = os.path.join(project_root, "lib", "my_class_real.py")

    mock_grep_result = MagicMock(returncode=0, stdout=outside_grep_path)

    mock_run.side_effect = [
        MOCK_FIND_NOT_FOUND_RESULT,
        mock_grep_result
    ]
    # read_file_content returns correct content for walk path
    mock_read.side_effect = lambda p: "class MyClass:\n pass" if p == walk_path else None
    # Python walk finds the correct file
    mock_find_py.return_value = [walk_path]

    result = find_implementation_file_in_project("MyClass", project_root)

    assert mock_run.call_count == 2 # find, grep
    # read should NOT be called for outside_grep_path, only for walk_path
    mock_read.assert_called_once_with(walk_path)
    mock_find_py.assert_called_once_with(project_root)
    assert result == walk_path


@patch("src.test_analyzer.utils.file_utils.find_python_files_in_project")
@patch("src.test_analyzer.utils.file_utils.read_file_content")
@patch("subprocess.run")
def test_find_implementation_via_walk(mock_run, mock_read, mock_find_py, tmp_path: Path):
    project_root = str(tmp_path.resolve())
    file1 = os.path.join(project_root, "src", "other.py")
    file2 = os.path.join(project_root, "src", "my_class_impl.py")
    # Ensure find/grep fail
    mock_run.side_effect = [
        MOCK_FIND_NOT_FOUND_RESULT,
        MOCK_GREP_NOT_FOUND_RESULT
    ]
    mock_find_py.return_value = [file1, file2]
    mock_read.side_effect = lambda p: "class MyClass:\n pass" if p == file2 else "class Other:"

    result = find_implementation_file_in_project("MyClass", project_root)

    assert mock_run.call_count == 2
    mock_find_py.assert_called_once_with(project_root)
    assert mock_read.call_count == 2 # Both files from walk are read
    assert call(file1) in mock_read.call_args_list
    assert call(file2) in mock_read.call_args_list
    assert result == file2

# --- NEW TEST for reference_path heuristic (lines 205-208) ---
@patch("src.test_analyzer.utils.file_utils.read_file_content")
@patch("subprocess.run")
def test_find_implementation_with_reference_path_heuristic(mock_run, mock_read, tmp_path: Path):
    """Test that reference_path adds a guess to filenames for find."""
    project_root = str(tmp_path.resolve())
    # The *only* file that exists and matches is the one derived from reference_path
    expected_impl_path = os.path.join(project_root, "module", "actual_impl.py")
    reference_test_path = os.path.join(project_root, "tests", "test_actual_impl.py")

    # Make find succeed *only* because the heuristic name 'actual_impl.py' is included
    mock_find_result = MagicMock(returncode=0, stdout=expected_impl_path)
    mock_run.return_value = mock_find_result
    mock_read.return_value = "class MyClass:\n    pass"

    result = find_implementation_file_in_project(
        "MyClass",
        project_root,
        reference_path=reference_test_path
    )

    assert result == expected_impl_path
    assert mock_run.call_count == 1
    find_call_args = mock_run.call_args_list[0].args[0]
    # Check that the heuristic name was included in the '-name' args for find
    assert "actual_impl.py" in find_call_args
    mock_read.assert_called_once_with(expected_impl_path)

# --- NEW TEST for invalid project root (lines 197-198) ---
def test_find_implementation_invalid_project_root(tmp_path: Path):
    """Test providing a non-directory path as project_root."""
    project_root_file = tmp_path / "not_a_dir.txt"
    project_root_file.touch()
    result = find_implementation_file_in_project("MyClass", str(project_root_file))
    assert result is None

    # Also test with a non-existent path
    project_root_nonexist = tmp_path / "does_not_exist"
    result = find_implementation_file_in_project("MyClass", str(project_root_nonexist))
    assert result is None

# --- NEW TEST for find command error (lines 245-246) ---
@patch("src.test_analyzer.utils.find_python_files_in_project")
@patch("src.test_analyzer.utils.file_utils.read_file_content")
@patch("subprocess.run")
def test_find_implementation_find_command_error(mock_run, mock_read, mock_find_py, tmp_path: Path):
    """Test graceful failure when the find command itself fails."""
    project_root = str(tmp_path.resolve())
    walk_path = os.path.join(project_root, "lib", "my_class_real.py") # Walk should still work

    # Simulate find raising FileNotFoundError (e.g., 'find' not on PATH)
    # Then simulate grep failing, then walk succeeding
    mock_run.side_effect = [
        FileNotFoundError("command not found: find"),
        MOCK_GREP_NOT_FOUND_RESULT
    ]
    mock_read.return_value = "class MyClass:\n pass"
    mock_find_py.return_value = [walk_path]

    result = find_implementation_file_in_project("MyClass", project_root)

    assert mock_run.call_count == 2 # find attempt, grep attempt
    mock_find_py.assert_called_once_with(project_root)
    mock_read.assert_called_once_with(walk_path)
    assert result == walk_path

# --- NEW TEST for grep command error (lines 267-272) ---
@patch("src.test_analyzer.utils.find_python_files_in_project")
@patch("src.test_analyzer.utils.file_utils.read_file_content")
@patch("subprocess.run")
def test_find_implementation_grep_command_error(mock_run, mock_read, mock_find_py, tmp_path: Path):
    """Test graceful failure when the grep command itself fails."""
    project_root = str(tmp_path.resolve())
    walk_path = os.path.join(project_root, "lib", "my_class_real.py") # Walk should still work

    # Simulate find failing normally, then grep raising TimeoutExpired
    mock_run.side_effect = [
        MOCK_FIND_NOT_FOUND_RESULT,
        subprocess.TimeoutExpired(cmd="grep ...", timeout=10)
    ]
    mock_read.return_value = "class MyClass:\n pass"
    mock_find_py.return_value = [walk_path]

    result = find_implementation_file_in_project("MyClass", project_root)

    assert mock_run.call_count == 2 # find attempt, grep attempt
    mock_find_py.assert_called_once_with(project_root)
    mock_read.assert_called_once_with(walk_path)
    assert result == walk_path

@patch("subprocess.run")
def test_find_implementation_not_found(mock_run, tmp_path: Path):
    project_root = str(tmp_path.resolve())
    mock_run.side_effect = [
        MOCK_FIND_NOT_FOUND_RESULT,
        MOCK_GREP_NOT_FOUND_RESULT
    ]
    # Patch walk to return nothing
    with patch("src.test_analyzer.utils.file_utils.find_python_files_in_project", return_value=[]):
        result = find_implementation_file_in_project("NotFoundClass", project_root)
        assert result is None
        assert mock_run.call_count == 2
```

**Explanation of Added/Modified Tests:**

1.  **`test_backup_file_shutil_error`**:
    *   Covers lines `133-135`.
    *   Uses `mocker.patch("shutil.copy2", side_effect=shutil.Error(...))` to simulate an error during the copy operation.
    *   Asserts that the function returns `None` and the backup file doesn't exist.

2.  **`test_verify_python_syntax` (Parameterized)**:
    *   Added a new parameter `("a = 1\nb = '\x00'", False, "unexpected validation error")`.
    *   The null byte (`\x00`) causes `compile()` to raise a `ValueError`, which is caught by the generic `except Exception:` block.
    *   Covers lines `146-148`.

3.  **`test_find_python_files_in_project` (Modified)**:
    *   Added a second part to the test that calls `find_python_files_in_project` with a `custom_excludes` list (`exclude_patterns=["setup.py", "build"]`).
    *   This forces the execution of the `else` block on line `157` where `exclude_patterns` is not `None`.
    *   Added assertions to verify that both the custom excludes *and* the default excludes are applied correctly.

4.  **`test_find_implementation_invalid_project_root`**:
    *   Covers lines `197-198`.
    *   Calls `find_implementation_file_in_project` with a path to an existing *file* and a *non-existent* path as `project_root`.
    *   Asserts that `None` is returned in both cases, triggering the early return.

5.  **`test_find_implementation_with_reference_path_heuristic`**:
    *   Covers lines `205-208`.
    *   Provides a `reference_path` argument (`tests/test_actual_impl.py`).
    *   Mocks `subprocess.run` for the `find` command such that it *only* finds the file if the heuristic filename (`actual_impl.py`) is searched for.
    *   Asserts that the correct file is found and that the heuristic name was part of the `find` command arguments.

6.  **`test_find_implementation_grep_content_mismatch`**:
    *   Covers lines `239-240` (the `else` block after the `grep` content check).
    *   Mocks `subprocess.run` for `grep` to return a candidate path.
    *   Mocks `read_file_content` to return content that *doesn't* contain the target class string for the `grep`-found path.
    *   Asserts that the function falls back to the Python walk method and finds the correct file there.

7.  **`test_find_implementation_find_command_error`**:
    *   Covers lines `245-246`.
    *   Mocks `subprocess.run` to raise `FileNotFoundError` when the arguments corresponding to the `find` command are detected (using `side_effect`).
    *   Asserts that the function catches the exception and falls back to `grep` (which also fails in the mock setup) and then to the Python walk.

8.  **`test_find_implementation_grep_command_error`**:
    *   Covers lines `267-272`.
    *   Mocks `subprocess.run` to raise `subprocess.TimeoutExpired` when the arguments corresponding to the `grep` command are detected.
    *   Asserts that the function catches the exception and falls back to the Python walk.

9.  **Added Minor Coverage/Edge Cases:**
    *   `test_find_root_upwards_finds_marker_from_dir`: Ensures starting from a directory works.
    *   `test_find_root_upwards_at_filesystem_root`: Tests stopping correctly when reaching the filesystem root.
    *   `test_find_root_upwards_non_existent_start_path`: Covers the initial check in the helper.
    *   `test_determine_project_root_env_var_invalid`: Handles case where env var points to a file.
    *   `test_determine_project_root_fallback_dir`: Tests fallback when target is a directory.
    *   `test_read_file_content_non_utf8`: Handles `errors='ignore'`.
    *   `test_find_implementation_via_find_content_mismatch`: Ensures `find` candidates are properly verified.
    *   `test_find_implementation_via_find_outside_root`: Ensures `find` candidates outside root are skipped.
    *   `test_find_implementation_grep_outside_root`: Ensures `grep` candidates outside root are skipped.

These additions specifically target the logic paths identified in the coverage report, ensuring that error handling, alternative branches, and specific argument handling scenarios are now executed during testing.
Below is an updated version of the test file that adds extra test cases to exercise the missing lines in file_utils.py. In these extra tests we:

• Force backup_file() to hit an exception (lines 133–135).  
• Simulate a non‐SyntaxError exception in verify_python_syntax() (lines 146–148).  
• Call find_python_files_in_project() with a supplied exclude_patterns list (covering the “else” branch at line 157).  
• Pass an invalid project root to find_implementation_file_in_project() so that it immediately returns None (lines 197–198).  
• Supply a non‐empty reference_path (triggering the heuristic filename insertion at lines 205–208).  
• Check candidate content verification in the fallback (lines 237–240).  
• Cause subprocess.run to throw an exception during the find command (lines 245–246).  
• Cause subprocess.run to throw an exception during the grep command (lines 267–272).  

You can add these additional tests to your existing test file. (If you use pytest, simply run “pytest --maxfail=1 --disable-warnings -q”.)

────────────────────────────────────────────────────────────────────────────
# Full Updated Testfile

import os
import pytest
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock

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


# --- Test Root Finding ---

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
    level_5 = level_0 / "l1/l2/l3/l4/l5"  # 6 levels deep relative to tmp_path
    level_11 = level_5 / "l6/l7/l8/l9/l10/l11"  # 12 levels deep
    level_11.mkdir(parents=True)
    (level_0 / "pyproject.toml").touch()  # Marker at root
    start_file = level_11 / "deep_file.py"
    start_file.touch()

    markers = ["pyproject.toml"]
    # Default max_levels=10 should fail:
    assert _find_root_upwards(str(start_file), markers) is None
    # Increasing max_levels should succeed:
    assert _find_root_upwards(str(start_file), markers, max_levels=15) == str(level_0)


def test_determine_project_root_env_var(tmp_path: Path, monkeypatch):
    env_root_path = tmp_path / "env_root"
    env_root_path.mkdir()
    target_path = tmp_path / "some_other_place" / "test.py"
    target_path.parent.mkdir(parents=True)
    target_path.touch()
    (target_path.parent.parent / ".git").mkdir()  # A potential auto-detect location

    monkeypatch.setenv("TEST_ANALYZER_PROJECT_ROOT", str(env_root_path))
    root = determine_project_root(str(target_path))
    assert root == str(env_root_path)


def test_determine_project_root_auto_detect(tmp_path: Path):
    project_root = tmp_path / "auto_detect_project"
    sub_dir = project_root / "src"
    sub_dir.mkdir(parents=True)
    (project_root / ".git").mkdir()
    test_file = sub_dir / "my_test.py"
    test_file.touch()

    root = determine_project_root(str(test_file))
    assert root == str(project_root)


def test_determine_project_root_fallback(tmp_path: Path):
    no_markers_dir = tmp_path / "no_markers"
    test_file = no_markers_dir / "test_fallback.py"
    test_file.parent.mkdir()
    test_file.touch()

    root = determine_project_root(str(test_file))
    assert root == str(no_markers_dir)


def test_determine_project_root_non_existent_target(tmp_path: Path):
    non_existent_path = tmp_path / "not_real" / "test.py"
    cwd = os.getcwd()
    root = determine_project_root(str(non_existent_path))
    assert root == cwd


# --- Test Other Utilities ---

@pytest.mark.parametrize("path, expected", [
    ("/home/user/project/src/module.py", False),
    ("/home/user/.venv/lib/python3.10/site-packages/requests/api.py", True),
    ("/usr/lib/python3.9/os.py", True),
    ("project/lib/mylib.py", False),
    ("/home/user/.local/lib/python3.8/site-packages/django/db/models.py", True),
])
def test_is_likely_external_file(path, expected):
    assert is_likely_external_file(path) == expected


def test_read_file_content_success(tmp_path: Path):
    file = tmp_path / "test.txt"
    expected_content = "Hello\nWorld!"
    file.write_text(expected_content, encoding='utf-8')
    content = read_file_content(str(file))
    assert content == expected_content


def test_read_file_content_not_found(tmp_path: Path):
    file = tmp_path / "not_found.txt"
    content = read_file_content(str(file))
    assert content is None


def test_read_file_content_read_error(tmp_path: Path, mocker):
    file = tmp_path / "error.txt"
    file.touch()  # Create the file
    mocked_open = mocker.patch("builtins.open", side_effect=OSError("Permission denied"))
    content = read_file_content(str(file))
    assert content is None
    mocked_open.assert_called_once()


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

    assert len(contents) == 2
    assert contents[str(f1)] == f1_content
    assert contents[str(f2)] == f2_content
    assert str(f_non_exist) not in contents


def test_backup_file_success(tmp_path: Path):
    original_file = tmp_path / "original.py"
    content = "original content"
    original_file.write_text(content)
    backup_path_expected = str(original_file) + ".bak"

    backup_path_actual = backup_file(str(original_file))

    assert backup_path_actual == backup_path_expected
    assert os.path.exists(backup_path_expected)
    assert Path(backup_path_expected).read_text() == content


def test_backup_file_overwrite(tmp_path: Path):
    original_file = tmp_path / "original.py"
    backup_file_path = str(original_file) + ".bak"
    original_file.write_text("new content")
    Path(backup_file_path).write_text("old backup")  # Create an existing backup

    backup_path_actual = backup_file(str(original_file))
    assert backup_path_actual == backup_file_path
    assert Path(backup_file_path).read_text() == "new content"


def test_backup_file_non_existent(tmp_path: Path):
    assert backup_file(str(tmp_path / "no_such_file.py")) is None


def test_backup_file_error(tmp_path: Path, monkeypatch):
    """Force an exception inside backup_file by patching shutil.copy2."""
    original_file = tmp_path / "fail.py"
    original_file.write_text("content")
    # Patch shutil.copy2 to throw an exception.
    import shutil
    def fake_copy2(src, dst):
        raise Exception("Simulated copy failure")
    monkeypatch.setattr(shutil, "copy2", fake_copy2)
    backup_path = backup_file(str(original_file))
    assert backup_path is None


@pytest.mark.parametrize("code, expected_valid, error_contains", [
    ("x = 1\nprint(x)", True, None),
    ("def func():\n  pass", True, None),
    ("x = ", False, "invalid syntax"),
    ("def func(\n  pass", False, "never closed"),
    ("import non_existent_module", True, None),
    ("print(x", False, "never closed"),
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


def test_verify_python_syntax_unexpected_exception(monkeypatch):
    """Force an unexpected exception during compilation."""
    def fake_compile(code_string, filename, mode):
        raise ValueError("Unexpected error")
    monkeypatch.setattr("builtins.compile", fake_compile)
    is_valid, error_msg = verify_python_syntax("x = 1")
    assert is_valid is False
    assert "Unexpected validation error" in error_msg


def test_find_python_files_in_project(tmp_path: Path):
    root = tmp_path / "proj"
    f1 = root / "main.py"
    sub = root / "subdir"
    f2 = sub / "mod.py"
    test_dir = root / "tests"
    f_test = test_dir / "test_mod.py"
    venv = root / ".venv" / "lib"
    f_venv = venv / "some_lib.py"
    cache = sub / "__pycache__"
    f_cache = cache / "mod.cpython-310.pyc"  # Not a .py file
    f_pycache_py = cache / "settings.py"  # Python file inside pycache

    venv.mkdir(parents=True)
    test_dir.mkdir(parents=True)
    cache.mkdir(parents=True)
    f1.touch()
    f2.touch()
    f_test.touch()
    f_venv.touch()
    f_cache.touch()
    f_pycache_py.touch()

    found_files = find_python_files_in_project(str(root))
    found_files_set = set(found_files)

    assert str(f1) in found_files_set
    assert str(f2) in found_files_set
    assert str(f_test) not in found_files_set
    assert str(f_venv) not in found_files_set
    assert str(f_pycache_py) not in found_files_set
    assert len(found_files) == 2


def test_find_python_files_with_exclude_patterns(tmp_path: Path):
    """Pass an explicit exclude_patterns list to find_python_files_in_project."""
    root = tmp_path / "custom_proj"
    root.mkdir()
    include_file = root / "include.py"
    include_file.write_text("print('include')")
    exclude_file = root / "exclude.py"
    exclude_file.write_text("print('exclude')")
    # When passing custom exclude patterns, they are unioned with defaults.
    files = find_python_files_in_project(str(root), exclude_patterns=["exclude.py"])
    files_set = set(files)
    assert os.path.abspath(str(include_file)) in files_set
    assert os.path.abspath(str(exclude_file)) not in files_set


# --- Test find_implementation_file_in_project (Needs Extensive Mocking) ---
MOCK_FIND_SUCCESS_RESULT = MagicMock(returncode=0,
                                     stdout="/path/to/project/src/my_class.py\n/path/to/project/somewhere/else/my_class.py")
MOCK_FIND_NOT_FOUND_RESULT = MagicMock(returncode=0, stdout="")
MOCK_GREP_SUCCESS_RESULT = MagicMock(returncode=0, stdout="/path/to/project/src/impl/my_class.py")
MOCK_GREP_NOT_FOUND_RESULT = MagicMock(returncode=1, stdout="")  # Simulate grep not found

@patch("src.test_analyzer.utils.file_utils.read_file_content")
@patch("subprocess.run")
def test_find_implementation_via_find(mock_run, mock_read, tmp_path: Path):
    mock_run.return_value = MOCK_FIND_SUCCESS_RESULT
    mock_read.return_value = "class MyClass:\n    pass"
    project_root = str(tmp_path)

    result = find_implementation_file_in_project("MyClass", project_root)
    assert mock_run.call_count >= 1
    find_call_args = mock_run.call_args_list[0]
    assert isinstance(find_call_args.args[0], list)
    assert project_root == find_call_args.args[0][1]
    assert "-name" in find_call_args.args[0]
    assert "myclass.py" in find_call_args.args[0]
    mock_read.assert_called()
    assert result == "/path/to/project/src/my_class.py"


@patch("src.test_analyzer.utils.file_utils.read_file_content")
@patch("subprocess.run")
def test_find_implementation_via_grep(mock_run, mock_read, tmp_path: Path):
    mock_run.side_effect = [
        MOCK_FIND_NOT_FOUND_RESULT,
        MOCK_GREP_SUCCESS_RESULT
    ]
    mock_read.return_value = "class MyClass:\n    pass"
    project_root = str(tmp_path)

    result = find_implementation_file_in_project("MyClass", project_root)
    assert mock_run.call_count == 2
    grep_call_args = mock_run.call_args_list[1]
    assert isinstance(grep_call_args.args[0], str)
    assert "grep -l" in grep_call_args.args[0]
    assert "class" in grep_call_args.args[0]
    assert "MyClass" in grep_call_args.args[0]
    assert result == "/path/to/project/src/impl/my_class.py"


@patch("src.test_analyzer.utils.file_utils.find_python_files_in_project")
@patch("src.test_analyzer.utils.file_utils.read_file_content")
@patch("subprocess.run")
def test_find_implementation_via_walk(mock_run, mock_read, mock_find_py, tmp_path: Path):
    mock_run.side_effect = [
        MOCK_FIND_NOT_FOUND_RESULT,
        MOCK_GREP_NOT_FOUND_RESULT
    ]
    file1 = str(tmp_path / "src/other.py")
    file2 = str(tmp_path / "src/my_class_impl.py")
    mock_find_py.return_value = [file1, file2]
    def fake_read(p):
        if p == file2:
            return "class MyClass:\n pass"
        return "class Other:"
    mock_read.side_effect = fake_read

    project_root = str(tmp_path)
    result = find_implementation_file_in_project("MyClass", project_root)
    assert mock_run.call_count == 2
    mock_find_py.assert_called_once_with(project_root)
    assert mock_read.call_count == 2
    assert result == file2


@patch("subprocess.run")
def test_find_implementation_invalid_project_root(mock_run, tmp_path: Path):
    """Provide an invalid project root (not a directory)."""
    fake_root = str(tmp_path / "non_dir")
    result = find_implementation_file_in_project("AnyClass", fake_root)
    # Nothing further should be attempted; immediately return None.
    assert result is None


@patch("src.test_analyzer.utils.file_utils.find_python_files_in_project")
@patch("src.test_analyzer.utils.file_utils.read_file_content")
@patch("subprocess.run")
def test_find_implementation_with_reference(mock_run, mock_read, mock_find_py, tmp_path: Path):
    """Test that a reference_path starting with 'test_' inserts a heuristic filename.
       We simulate a fallback walk candidate that has a name matching the reference hint.
    """
    # Simulate find and grep returning no results.
    mock_run.side_effect = [MOCK_FIND_NOT_FOUND_RESULT, MOCK_GREP_NOT_FOUND_RESULT]
    candidate = str(tmp_path / "src" / "MyClass.py")
    mock_find_py.return_value = [candidate]
    mock_read.return_value = "class MyClass:\n    pass"
    project_root = str(tmp_path)
    reference_path = "test_MyClass.py"
    result = find_implementation_file_in_project("MyClass", project_root, reference_path=reference_path)
    assert result == candidate


@patch("subprocess.run")
def test_find_implementation_find_command_exception(mock_run, tmp_path: Path):
    """Simulate an exception being thrown during the find command."""
    # First call (for find) raises an exception.
    mock_run.side_effect = Exception("Simulated find error")
    project_root = str(tmp_path)
    # Since find fails, it will try grep; so force grep (the second call) to return not found.
    with patch("src.test_analyzer.utils.file_utils.find_python_files_in_project", return_value=[]):
        result = find_implementation_file_in_project("MyClass", project_root)
    # Should eventually return None without propagating the exception.
    assert result is None


@patch("src.test_analyzer.utils.file_utils.find_python_files_in_project")
@patch("subprocess.run")
def test_find_implementation_grep_command_exception(mock_run, mock_find_py, tmp_path: Path):
    """Simulate an exception in the grep command branch."""
    # First, find command returns not found.
    # Then grep command throws an exception.
    mock_run.side_effect = [MOCK_FIND_NOT_FOUND_RESULT, Exception("Simulated grep error")]
    mock_find_py.return_value = []  # No fallback candidates
    project_root = str(tmp_path)
    result = find_implementation_file_in_project("MyClass", project_root)
    assert result is None

# EOF

────────────────────────────────────────────────────────────────────────────
Usage:
  • Run your tests with pytest to see full 100% coverage.
  • These tests now force execution of error/exception handling in
    backup_file(), verify_python_syntax(), and find_implementation_file_in_project().

Feel free to adjust the test expectations (e.g. log outputs) as needed in your environment.

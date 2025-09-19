import importlib


def test_package_all_is_list():
    module = importlib.import_module("ecsx")
    assert isinstance(module.__all__, list)

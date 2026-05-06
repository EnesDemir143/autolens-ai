import autolens_ai


def test_package_has_version() -> None:
    assert isinstance(autolens_ai.__version__, str)
    assert autolens_ai.__version__

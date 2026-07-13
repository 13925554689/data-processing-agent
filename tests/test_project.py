"""Test project structure and basic imports."""


def test_version():
    """Verify package version is accessible."""
    from src import __version__

    assert __version__ == "0.1.0"


def test_all_packages_importable():
    """Verify all subpackages are importable."""
    modules = [
        "src.agents",
        "src.connectors",
        "src.layers",
        "src.quality",
        "src.lineage",
        "src.orchestration",
        "src.asset",
        "src.api",
        "src.knowledge",
        "src.skills",
    ]
    for mod in modules:
        __import__(mod)

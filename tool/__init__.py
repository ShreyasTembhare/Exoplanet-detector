"""Transit desk: one-port search tool for the TESS pipeline."""

__all__ = ["create_app"]


def create_app(data_dir: str = "var/desk"):
    from tool.server import create_app as _create

    return _create(data_dir)

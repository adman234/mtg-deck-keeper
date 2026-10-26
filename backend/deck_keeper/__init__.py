__version__ = "0.1.0"


def build_id() -> str:
    """Which build is running (the git commit in the Docker image), shown with test scores."""
    import os

    return os.environ.get("APP_BUILD", "dev")[:7]

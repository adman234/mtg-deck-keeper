__version__ = "0.1.0"


def build_id() -> str:
    """Which build is running (the git commit in the Docker image), shown with test scores."""
    import os

    return os.environ.get("APP_BUILD", "dev")[:7]


def build_date() -> str:
    """When the running build's commit was made (ISO 8601), or "" outside a Docker image."""
    import os

    return os.environ.get("APP_BUILD_DATE", "")

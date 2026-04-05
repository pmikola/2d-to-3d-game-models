"""Auto-install helper for optional pipeline dependencies.

Checks whether a Python package is importable and, if not, installs it
automatically (from PyPI or a Git repo) before retrying the import.

This keeps the main pipeline modules free of subprocess/pip logic.
"""

import importlib
import logging
import subprocess
import sys

logger = logging.getLogger(__name__)


def _build_install_command(
    import_name: str,
    pip_spec: str | None = None,
    git_url: str | None = None,
    editable: bool = False,
    no_deps: bool = False,
) -> list[str]:
    """Build a pip install command for an optional dependency."""
    cmd = [sys.executable, "-m", "pip", "install", "--quiet"]
    if no_deps:
        cmd.append("--no-deps")

    if git_url:
        url = git_url
        # Accept plain GitHub URLs and convert to pip-friendly form.
        if url.startswith("https://github.com/") and not url.startswith("git+"):
            url = f"git+{url}"
        if not url.endswith(".git"):
            url = f"{url}.git"
        if editable:
            cmd += ["-e", url]
        else:
            cmd.append(url)
    elif pip_spec:
        cmd.append(pip_spec)
    else:
        cmd.append(import_name)

    return cmd


def _summarize_installer_output(output: str, max_lines: int = 12) -> str:
    """Return the most relevant tail of pip/git output."""
    if not output:
        return ""
    lines = [line.rstrip() for line in output.splitlines() if line.strip()]
    if not lines:
        return ""
    return "\n".join(lines[-max_lines:])


def ensure_package(
    import_name: str,
    pip_spec: str | None = None,
    git_url: str | None = None,
    editable: bool = False,
    allow_no_deps_fallback: bool = False,
) -> None:
    """Ensure *import_name* is importable, installing it if necessary.

    Tries to import the package first.  If that fails, installs via pip
    from *pip_spec* (PyPI) or *git_url* (GitHub) and retries the import.

    Args:
        import_name: The top-level module name to ``import`` (e.g. ``"mvadapter"``).
        pip_spec: A PyPI install specifier (e.g. ``"mmgp>=0.9.0"``).
            Ignored when *git_url* is provided.
        git_url: A ``git+https://`` URL or plain ``https://`` GitHub URL.
            When a plain GitHub URL is given it is converted to the
            ``git+https://`` form automatically.
        editable: If True and *git_url* is used, install in editable mode
            (``pip install -e``).
        allow_no_deps_fallback: If True and a git install fails, retry with
            ``pip install --no-deps``. This is useful for packages like
            MV-Adapter whose metadata includes platform-specific extras not
            needed by this pipeline path.
    """
    # Fast path: already installed.
    try:
        importlib.import_module(import_name)
        return
    except ImportError:
        pass

    cmd = _build_install_command(
        import_name=import_name,
        pip_spec=pip_spec,
        git_url=git_url,
        editable=editable,
        no_deps=False,
    )

    logger.info(
        "Package '%s' not found. Installing: %s",
        import_name,
        " ".join(cmd),
    )

    result = subprocess.run(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )

    if result.returncode != 0 and allow_no_deps_fallback and git_url:
        fallback_cmd = _build_install_command(
            import_name=import_name,
            pip_spec=pip_spec,
            git_url=git_url,
            editable=editable,
            no_deps=True,
        )
        logger.warning(
            "Standard install for '%s' failed. Retrying without dependency "
            "resolution because this package declares optional platform-"
            "specific dependencies that may be unavailable.\n%s",
            import_name,
            _summarize_installer_output(result.stderr),
        )
        result = subprocess.run(
            fallback_cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        if result.returncode == 0:
            logger.info(
                "Package '%s' installed successfully via --no-deps fallback.",
                import_name,
            )
        else:
            error_output = _summarize_installer_output(result.stderr)
            raise RuntimeError(
                f"Failed to auto-install '{import_name}'. "
                f"Attempted commands:\n  {' '.join(cmd)}\n  {' '.join(fallback_cmd)}"
                + (f"\nInstaller output:\n{error_output}" if error_output else "")
            )
    elif result.returncode != 0:
        error_output = _summarize_installer_output(result.stderr)
        raise RuntimeError(
            f"Failed to auto-install '{import_name}'. "
            f"Please install it manually:\n  {' '.join(cmd)}"
            + (f"\nInstaller output:\n{error_output}" if error_output else "")
        )

    # Retry import.
    try:
        importlib.import_module(import_name)
        logger.info("Package '%s' installed and imported successfully.", import_name)
    except ImportError as exc:
        raise ImportError(
            f"Installed '{import_name}' but import still fails. "
            f"Try installing manually:\n  {' '.join(cmd)}"
        ) from exc

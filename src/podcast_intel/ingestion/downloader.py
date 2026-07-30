"""
Audio file downloader with progress tracking.

Streams a podcast audio file to disk with optional progress reporting. The
download goes to a ``.part`` sidecar and is moved into place only after the
whole body has been read, so a partial file is never mistaken for a complete
one. ``requests`` is a core dependency -- no extra is needed.

Not implemented: resumable (HTTP Range) downloads, bandwidth throttling and
retry/backoff. Callers that need them should wrap :func:`download_episode`.
"""

import logging
from collections.abc import Callable
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

#: Bytes read per iteration while streaming.
CHUNK_SIZE = 64 * 1024

#: Seconds to wait for the connection and for each chunk.
DEFAULT_TIMEOUT = 30

#: Leading bytes that identify a file as MP3 (ID3 tag or MPEG frame sync).
_MP3_MAGIC = (b"ID3", b"\xff\xfb", b"\xff\xf3", b"\xff\xf2", b"\xff\xfa")


def download_episode(
    url: str,
    output_path: Path,
    progress_callback: Callable[[int, int], None] | None = None,
    timeout: int = DEFAULT_TIMEOUT,
) -> bool:
    """
    Download a podcast audio file from a URL.

    Args:
        url: Audio file URL
        output_path: Local path to save file
        progress_callback: Optional callback for progress updates
            ``(bytes_downloaded, total_bytes)``. ``total_bytes`` is 0 when the
            server sends no ``Content-Length``.
        timeout: Per-request timeout in seconds

    Returns:
        True if the download completed, False on any network or write error.

    Example:
        >>> def on_progress(downloaded, total):
        ...     print(f"Progress: {downloaded}/{total} bytes")
        >>> download_episode(
        ...     "https://example.com/audio.mp3",
        ...     Path("data/audio/episode.mp3"),
        ...     progress_callback=on_progress,
        ... )                                            # doctest: +SKIP
    """
    output_path = Path(output_path)
    partial = output_path.with_suffix(output_path.suffix + ".part")

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with requests.get(url, stream=True, timeout=timeout) as response:
            response.raise_for_status()
            total = int(response.headers.get("Content-Length") or 0)

            downloaded = 0
            with open(partial, "wb") as handle:
                for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                    if not chunk:
                        continue
                    handle.write(chunk)
                    downloaded += len(chunk)
                    if progress_callback is not None:
                        progress_callback(downloaded, total)

        if total and downloaded != total:
            logger.error(
                "Truncated download for %s: got %d of %d bytes", url, downloaded, total
            )
            partial.unlink(missing_ok=True)
            return False

        partial.replace(output_path)
        return True

    except (requests.RequestException, OSError) as exc:
        logger.error("Download failed for %s: %s", url, exc)
        partial.unlink(missing_ok=True)
        return False


def validate_audio_file(file_path: Path) -> bool:
    """
    Check that a downloaded file exists, is non-empty and starts like an MP3.

    This is a cheap header check, not a decode: it catches the common failure
    where a feed serves an HTML error page with an ``.mp3`` URL. It does not
    verify that the whole stream is decodable.

    Args:
        file_path: Path to audio file

    Returns:
        True if the file looks like an MP3, False otherwise
    """
    path = Path(file_path)
    try:
        if not path.is_file() or path.stat().st_size == 0:
            return False
        with open(path, "rb") as handle:
            header = handle.read(3)
    except OSError as exc:
        logger.error("Could not read %s: %s", path, exc)
        return False

    return any(header.startswith(magic) for magic in _MP3_MAGIC)


def get_file_size(url: str, timeout: int = DEFAULT_TIMEOUT) -> int | None:
    """
    Get the expected file size from a URL without downloading the body.

    Args:
        url: Audio file URL
        timeout: Request timeout in seconds

    Returns:
        File size in bytes, or None if the server does not report one.
    """
    try:
        response = requests.head(url, timeout=timeout, allow_redirects=True)
        response.raise_for_status()
    except requests.RequestException as exc:
        logger.error("HEAD failed for %s: %s", url, exc)
        return None

    length = response.headers.get("Content-Length")
    if length is None:
        return None
    try:
        return int(length)
    except ValueError:
        return None

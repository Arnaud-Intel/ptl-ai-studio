"""Offline setup checks; an optional single, paced live probe when requested."""
from __future__ import annotations

import argparse
import importlib.metadata
import os
import re
import shutil
import subprocess


def check() -> list[tuple[bool, str]]:
    results = []
    for package in ('yt-dlp', 'yt-dlp-ejs'):
        try:
            results.append((True, f'{package}: {importlib.metadata.version(package)}'))
        except importlib.metadata.PackageNotFoundError:
            results.append((False, f'{package}: missing. Run uv sync --locked --extra openvino.'))
    runtime_ok = False
    for name, minimum in (('deno', (2, 3, 0)), ('node', (22, 0, 0))):
        executable = shutil.which(name)
        if not executable:
            continue
        try:
            result = subprocess.run([executable, '--version'], capture_output=True, text=True, timeout=5,
                                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
            match = re.search(r'(\d+)\.(\d+)\.(\d+)', result.stdout)
            if result.returncode == 0 and match and tuple(map(int, match.groups())) >= minimum:
                runtime_ok = True
                results.append((True, f'{name}: {match.group()}'))
            else:
                results.append((False, f'{name}: upgrade required (minimum {".".join(map(str, minimum))}).'))
        except (OSError, subprocess.TimeoutExpired):
            results.append((False, f'{name}: could not check version.'))
    if not runtime_ok:
        results.append((False, 'Install Deno 2.3+ (recommended) or Node 22+ and reopen the launcher. https://docs.deno.com/runtime/getting_started/installation/'))
    cookies = os.environ.get('PTL_YOUTUBE_COOKIES', '').strip()
    if cookies:
        results.append((os.path.isfile(cookies), 'YouTube cookies: configured (contents and path not displayed).'))
    else:
        results.append((True, 'YouTube cookies: not configured; public feeds use an anonymous session.'))
    return results


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', help='Optionally test ONE YouTube live URL. Default makes no network request.')
    args = parser.parse_args(argv)
    results = check()
    for ok, message in results:
        print(('OK: ' if ok else 'CHECK: ') + message)
    if not all(ok for ok, _ in results):
        return 1
    if args.url:
        from .sources import is_youtube, resolve_youtube_stream
        from .youtube import safe_message
        if not is_youtube(args.url):
            parser.error('--url must be a YouTube live page URL')
        try:
            resolve_youtube_stream(args.url)
            print('YouTube resolved an HLS stream. This does not test frame decoding or guarantee continued access.')
        except Exception as exc:
            print('CHECK: ' + safe_message(exc))
            return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

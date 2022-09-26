from pip._internal.commands.show import search_packages_info
import subprocess
import sys
import json
import pathlib
import math
import operator

TMPL = '{:<30} {:<30} {:<10} {:>9}'


def humanize(size):
    """Convert size in bytes to human readable output."""
    try:
        units = ['B', 'KB', 'MB', 'GB']
        i = math.floor(math.log(size) / math.log(1024))
        hsize = f"{round(size / 1024**i)}{units[i]}"
    except ValueError:
        hsize = '0B'
    return hsize


def pip_list_ext():
    """List installed packages with additional information."""
    # python -m pip list --format --json --verbose
    pip_list = subprocess.run([sys.executable, '-m', 'pip', 'list',
                               '--format', 'json', '--verbose'],
                              capture_output=True)

    if pip_list.returncode != 0:
        raise RuntimeError('...')

    query = [pkg['name'] for pkg in json.loads(pip_list.stdout)]
    packages = []
    for pkg in search_packages_info(query):
        print(pip_list.stdout)
        location = pathlib.Path(pkg['location'])
        pkg['installer'] = pkg.get('installer', '')
        pkg['files'] = pkg.get('files', [])
        size = sum((location / f).stat().st_size
                       for f in pkg['files'] if (location / f).is_file())
        pkg['size'] = size
        packages.append(pkg)

    return packages

if __name__ == '__main__':
    packages = pip_list_ext()

    print(TMPL.format('Package', 'Version', 'Installer', 'Size'))
    print(TMPL.format('-'*30, '-'*30, '-'*10, '-'*9))
    for pkg in packages:
        print(TMPL.format(pkg['name'], pkg['version'],
                          pkg['installer'], humanize(pkg['size'])))

    # TOP 10
    N = 10
    pkgs = sorted(packages, key=operator.itemgetter('size'), reverse=True)
    print(f"\nTOP {N}:")
    for i, pkg in enumerate(pkgs[:N], 1):
        print(f"{i:>2}. {pkg['name']} ({humanize(pkg['size'])})")

    # du -h -s $(conda info --base)/envs/tf
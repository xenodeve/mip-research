"""Drive one Colab account's A100 session. The account is named on every call (COLAB_ACCOUNT=a|b|c|d) and asserted, so a job cannot
land on another machine; results from different accounts are never compared (each is its own VM and vLLM install).

  a  C:\\Users\\xenod           session a1
  b  C:\\Users\\xenod\\colab-b   session b1
  c  C:\\Users\\xenod\\colab-c   session c1
  d  C:\\Users\\xenod\\colab-d   session d1 (a friend's account, added 2026-11-08)

A new account logs in once, in the developer's own PowerShell (copy-paste OAuth):
  $env:USERPROFILE = 'C:\\Users\\xenod\\colab-d'; colab usage

COLAB_ACCOUNT=a python vm.py new                     start the A100 session
COLAB_ACCOUNT=a python vm.py usage                   compute units left (prints `CU <balance>` last)
COLAB_ACCOUNT=a python vm.py probe                   hostname, GPU, /content
COLAB_ACCOUNT=a python vm.py up LOCAL REMOTE         upload a file (under 100 MB)
COLAB_ACCOUNT=a python vm.py get REMOTE LOCAL        download a file
COLAB_ACCOUNT=a python vm.py sh 'COMMAND' [timeout]  run a shell command on the VM, print its output
COLAB_ACCOUNT=a python vm.py bg 'COMMAND' LOG        start a command on the VM in the background
COLAB_ACCOUNT=a python vm.py stop                    stop the session (the compute units stop with it)"""
import os
import re
import subprocess
import sys
import tempfile

ACCOUNTS = {'a': (r'C:\Users\xenod', 'a1'), 'b': (r'C:\Users\xenod\colab-b', 'b1'), 'c': (r'C:\Users\xenod\colab-c', 'c1'),
            'd': (r'C:\Users\xenod\colab-d', 'd1')}
KEY = os.environ.get('COLAB_ACCOUNT', '')
if KEY not in ACCOUNTS:
    raise SystemExit('set COLAB_ACCOUNT to one of a, b, c, d')
ACCOUNT, SESSION = ACCOUNTS[KEY]
assert SESSION == KEY + '1'


def colab(*args, timeout=600):
    env = dict(os.environ, USERPROFILE=ACCOUNT, MSYS_NO_PATHCONV='1', PYTHONIOENCODING='utf-8', PYTHONUTF8='1')
    p = subprocess.run(['colab', *args], env=env, capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=timeout,
                       creationflags=0x08000000)
    return p.returncode, (p.stdout or '') + (p.stderr or '')


def py(code, timeout=120):
    with tempfile.NamedTemporaryFile('w', suffix='.py', delete=False, encoding='utf-8') as f:
        f.write(code)
        path = f.name
    try:
        return colab('exec', '-s', SESSION, '-f', path, '--timeout', str(timeout), timeout=timeout + 60)
    finally:
        os.unlink(path)


def main(argv):
    # Git Bash rewrites a `/content/...` argument into `C:/Program Files/Git/content/...` before Python starts: undo it
    argv = [re.sub(r'^[A-Za-z]:/Program Files/Git(?=/content)', '', a) for a in argv]
    cmd = argv[1] if len(argv) > 1 else ''
    if cmd == 'new':
        code, out = colab('new', '-s', SESSION, '--gpu', 'A100', '--high-mem', timeout=900)
    elif cmd == 'usage':
        code, out = colab('usage')
        m = re.search(r'balance:\s*([\d.]+)', out)
        out += f'\nCU {m.group(1) if m else "unknown"}'
    elif cmd == 'stop':
        code, out = colab('stop', '-s', SESSION, timeout=300)
    elif cmd == 'probe':
        code, out = py("import subprocess;print(subprocess.run('hostname; nvidia-smi --query-gpu=name,memory.total --format=csv; "
                       "free -g | head -2; df -h /content | tail -1; ls /content', shell=True, capture_output=True, text=True).stdout)")
    elif cmd == 'up':
        code, out = colab('upload', '-s', SESSION, argv[2], argv[3], timeout=1800)
    elif cmd == 'get':
        code, out = colab('download', '-s', SESSION, argv[2], argv[3], timeout=1800)
    elif cmd == 'sh':
        t = float(argv[3]) if len(argv) > 3 else 120
        code, out = py('import subprocess\np = subprocess.run(%r, shell=True, capture_output=True, text=True)\nprint(p.stdout[-20000:], p.stderr[-5000:])'
                       % argv[2], timeout=t)
    elif cmd == 'bg':
        # detached from the kernel's pipes, so `colab exec` returns at once and the job outlives it
        code, out = py('import subprocess\nsubprocess.Popen(["bash", "-c", %r], stdin=subprocess.DEVNULL, stdout=open(%r, "w"), '
                       'stderr=subprocess.STDOUT, start_new_session=True)\nprint("started")' % (argv[2], argv[3]))
    else:
        raise SystemExit(__doc__)
    print(out)
    return code


if __name__ == '__main__':
    sys.exit(main(sys.argv))

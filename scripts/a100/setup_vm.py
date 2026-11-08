"""Start the 4B setup in the background on a Colab VM: vLLM, the public Qwen3.5-4B (no token on the VM).
The stack is pinned to what rounds 1-8 ran (vllm 0.33.0, transformers 5.20.2, torch 2.14.0+cu130 by vLLM's own pin), so a round is not
silently run on a newer vLLM. Writes /content/t4b/setup.log, setup.freeze (pip freeze) and, last, setup.done with the versions;
run_vm9.sh refuses to start unless setup.done names the pinned vLLM."""
import subprocess
import textwrap

script = textwrap.dedent('''
    set -x
    mkdir -p /content/t4b && cd /content/t4b
    nvidia-smi --query-gpu=name,memory.total --format=csv
    pip install -q vllm==0.33.0 transformers==5.20.2 2>&1 | tail -3
    python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3.5-4B', local_dir='/content/q4b')" 2>&1 | tail -2
    du -sh /content/q4b
    pip freeze > /content/t4b/setup.freeze 2>&1
    python -c "import vllm, transformers, torch; print('vllm', vllm.__version__, 'transformers', transformers.__version__, 'torch', torch.__version__)" > /content/t4b/setup.done 2>&1
''')
open('/content/t4b/setup.sh', 'w').write(script)
subprocess.Popen('nohup bash /content/t4b/setup.sh > /content/t4b/setup.log 2>&1 &', shell=True)
print('started')

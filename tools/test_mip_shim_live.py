"""The llama.cpp shim, live: a patched llama-server (build-w4a8-mip) on the RTX 5060 Ti and a ChannelService in this process.
Skipped unless MIP_SHIM_TEST=1 (needs the build, the GGUF, the HF tokenizer and the card).
  MIP_SHIM_TEST=1 python -m unittest tools.test_mip_shim_live -v
"""
import json
import os
import subprocess
import time
import unittest
import urllib.request

SERVER = os.environ.get('MIP_SERVER', 'C:/AI/llama.cpp/build-w4a8-mip/bin/llama-server.exe')
GGUF = os.environ.get('MIP_GGUF', 'C:/Users/xenod/mip-work/gguf/qwen35-4b-nvfp4-axion.gguf')
HF = 'C:/Users/xenod/mip-work/q4b-hf'
PORT, CHANNEL = 8097, 8098
PROMPT = '<|im_start|>user\nWhat is 3*4?<|im_end|>\n<|im_start|>assistant\n<think>\n'


class FakeRollout:
    def __init__(self):
        self.out, self.phase, self.row, self.events = [], 'think', dict(question='3*4'), []


class SteerAtStart:
    """A planner that steers ` 999` at the first token of the thinking: text the model would not write by itself."""
    def __init__(self, pieces):
        from tools.igc.mip_planner import Plan
        from tools.igc.token_trie import ValueDag
        self.plan = Plan(' 999', ValueDag.build(' 999', pieces), 'test', [], '3*4', '999')

    def at_start(self, question):
        return self.plan

    def on_trigger(self, text, phase):
        return None


def post(body):
    req = urllib.request.Request(f'http://127.0.0.1:{PORT}/completion', data=json.dumps(body).encode(),
                                 headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())


@unittest.skipUnless(os.environ.get('MIP_SHIM_TEST'), 'live: set MIP_SHIM_TEST=1 (patched server, GGUF, RTX 5060 Ti)')
class ShimLive(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from transformers import AutoTokenizer
        from tools.igc.mip_channel import ChannelService
        from tools.igc.token_trie import vocab_pieces
        cls.tok = AutoTokenizer.from_pretrained(HF)
        cls.pieces = vocab_pieces(cls.tok)
        equals = [i for i in range(len(cls.tok)) if cls.pieces[i].strip() == '=']
        cls.svc = ChannelService(equals, cls.tok.convert_tokens_to_ids('</think>'), cls.pieces, cls.tok, port=CHANNEL)
        cls.svc.start()
        env = dict(os.environ, CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES='1', MIP_CHANNEL=f'127.0.0.1:{CHANNEL}')
        cuda = 'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.3/bin'
        env['PATH'] = f'{cuda}/x64;{cuda};' + env.get('PATH', '')
        cls.log = open(os.path.join(os.environ.get('TEMP', '.'), 'mip-shim-live.log'), 'w')
        cls.srv = subprocess.Popen([SERVER, '-m', GGUF, '-ngl', '999', '-c', '16384', '-np', '2', '-kvu', '-fa', 'on', '-ctk', 'q8_0',
                                    '-ctv', 'q8_0', '--no-cache-idle-slots', '--host', '127.0.0.1', '--port', str(PORT), '--no-webui'],
                                   env=env, stdout=cls.log, stderr=subprocess.STDOUT)
        for _ in range(120):
            if cls.srv.poll() is not None:
                raise RuntimeError(f'server exited with {cls.srv.returncode}')
            try:
                with urllib.request.urlopen(f'http://127.0.0.1:{PORT}/health', timeout=2) as r:
                    if b'ok' in r.read():
                        break
            except OSError:
                time.sleep(1)
        else:
            raise RuntimeError('server did not start')

    @classmethod
    def tearDownClass(cls):
        cls.srv.terminate()
        cls.srv.wait(30)
        cls.svc.stop()
        cls.log.close()

    def test_a_plan_steers_the_first_tokens(self):
        self.svc.register('t1', FakeRollout(), SteerAtStart(self.pieces))
        r = post(dict(prompt=PROMPT, n_predict=6, temperature=0, mip_id='t1', cache_prompt=False))
        self.assertTrue(r['content'].startswith(' 999'), r['content'])

    def test_no_mip_id_is_the_stock_path(self):
        r = post(dict(prompt=PROMPT, n_predict=6, temperature=0, cache_prompt=False))
        self.assertFalse(r['content'].startswith(' 999'), r['content'])

    def test_an_unregistered_id_steers_nothing_and_completes(self):
        r = post(dict(prompt=PROMPT, n_predict=6, temperature=0, mip_id='nobody', cache_prompt=False))
        self.assertFalse(r['content'].startswith(' 999'), r['content'])


    def test_z_service_down_still_answers(self):
        # runs last: the service stops, the shim must give up on it and the server keep answering, unsteered
        self.svc.stop()
        t = time.monotonic()
        r = post(dict(prompt=PROMPT, n_predict=6, temperature=0, mip_id='t1', cache_prompt=False))
        self.assertFalse(r['content'].startswith(' 999'), r['content'])
        self.assertLess(time.monotonic() - t, 30)


if __name__ == '__main__':
    unittest.main()

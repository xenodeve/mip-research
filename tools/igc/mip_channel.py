"""mip_channel.py: the MIP service the llama.cpp shim streams every token to (token channel design, 2026-11-12).

One loopback socket per server; one JSON object a line. On connect the service sends `{"t":"hello","triggers":[ids],"think_close":id}`.
The shim sends `open` (mip_id, slot, the request's prompt tokens), `tok` (each token accepted), and `ask` (slot, seq) after a trigger token
(seq >= 1) and once at the first sample of a request (seq 0); the service answers each ask with `{"t":"plan","slot","seq","dag","after_ban"}`
(`dag` null for nothing to steer). The driver registers each request's mip_id with its Rollout and Planner before posting it.

seq 0: the rest of a value a previous request of the same answer ended inside (read against the answer's text, not the shim's count), or
the fixed-question line when the answer has no text yet, else nothing. seq >= 1: Planner.on_trigger on the answer's text so far, in the
thinking only. Each plan is recorded on the Rollout as an event `steered value` (expr, value, at, done).

ChannelService(triggers, think_close, pieces, tokenizer, host='127.0.0.1', port=0).start() -> port; register(mip_id, rollout, planner);
stats() -> dict; stop()
"""
import json
import socket
import threading

from tools.igc.mip_planner import Plan
from tools.igc.token_trie import ValueDag


class _Answer:
    """What the service keeps for one answer across its requests: the plan being steered and where it started."""
    def __init__(self):
        self.plan, self.at, self.node, self.event = None, 0, 0, None
        self.prefix = []   # the answer's tokens before the value: a resume needs them unchanged


class ChannelService:
    def __init__(self, triggers, think_close, pieces, tokenizer, host='127.0.0.1', port=0):
        self.triggers, self.think_close = sorted(set(triggers)), think_close
        self.pieces, self.tok = pieces, tokenizer
        self.host, self.port = host, port
        self.lock = threading.Lock()
        self.ids = {}       # mip_id -> (rollout, planner)
        self.answers = {}   # id(rollout) -> _Answer
        self.counts = dict(asks=0, plans=0, steered=0, dropped=0)
        self.server = None
        self.conns = []     # live connections: stop() closes them, as a service that dies does

    # ---- the driver's side ----------------------------------------------------------------------------------------------------------
    def register(self, mip_id, rollout, planner):
        with self.lock:
            self.ids[mip_id] = (rollout, planner)
            self.answers.setdefault(id(rollout), _Answer())

    def stats(self):
        with self.lock:
            return dict(self.counts)

    def start(self):
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server.bind((self.host, self.port))
        self.server.listen(4)
        self.port = self.server.getsockname()[1]
        threading.Thread(target=self._accept, daemon=True).start()
        return self.port

    def stop(self):
        if self.server is not None:
            self.server.close()
            self.server = None
        for conn in self.conns:
            try:
                conn.shutdown(socket.SHUT_RDWR)
                conn.close()
            except OSError:
                pass
        self.conns = []

    # ---- the wire -------------------------------------------------------------------------------------------------------------------
    def _accept(self):
        while self.server is not None:
            try:
                conn, _ = self.server.accept()
            except OSError:
                return
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self.conns.append(conn)
            threading.Thread(target=self._serve, args=(conn,), daemon=True).start()

    def _serve(self, conn):
        f = conn.makefile('rw', encoding='utf-8', newline='\n')
        out_lock = threading.Lock()

        def send(m):
            with out_lock:
                f.write(json.dumps(m, separators=(',', ':')) + '\n')
                f.flush()
        send(dict(t='hello', triggers=self.triggers, think_close=self.think_close))
        slots = {}   # slot -> {'mip_id', 'new': [tokens]}
        try:
            for line in f:
                m = json.loads(line)
                t = m.get('t')
                if t == 'open':
                    slots[m['slot']] = dict(mip_id=m['mip_id'], slot=m['slot'], new=[], base=self._base(m['mip_id']))
                elif t == 'tok':
                    s = slots.get(m['slot'])
                    if s is not None:
                        s['new'].append(m['id'])
                        self._advance(s, m['id'])
                elif t == 'ask':
                    plan = self._ask(slots.get(m['slot']), m['seq'])
                    send(dict(t='plan', slot=m['slot'], seq=m['seq'], dag=plan.dag.to_json() if plan else None,
                              after_ban=plan.after_ban if plan else []))
        except (OSError, ValueError):
            pass
        finally:
            conn.close()

    # ---- the decisions --------------------------------------------------------------------------------------------------------------
    def _lookup(self, s):
        if s is None:
            return None, None, None
        hit = self.ids.get(s['mip_id'])
        if hit is None:
            return None, None, None
        rollout, planner = hit
        return rollout, planner, self.answers.setdefault(id(rollout), _Answer())

    def _base(self, mip_id):
        """How many tokens the answer had when this request opened: the driver may take the reply (and grow rollout.out) before the
        service reads the request's last ask, so a request's text is the answer up to here plus the tokens the shim sent since."""
        with self.lock:
            hit = self.ids.get(mip_id)
            return len(hit[0].out) if hit else 0

    def _ask(self, s, seq):
        with self.lock:
            self.counts['asks'] += 1
            rollout, planner, a = self._lookup(s)
            if rollout is None:
                return None
            if seq == 0:
                plan = self._resume(rollout, a)
                if plan is None and not rollout.out and rollout.phase == 'think':
                    plan = planner.at_start(rollout.row.get('question'))
                    if plan is not None:
                        self._begin(rollout, a, plan, [], s, 0)
                return plan
            new = s['new']
            phase = 'answer' if rollout.phase != 'think' or self.think_close in new else 'think'
            before = rollout.out[:s['base']]
            text = self.tok.decode(before + new)
            plan = planner.on_trigger(text, phase)
            if plan is not None:
                self._begin(rollout, a, plan, before + new, s, seq)
            return plan

    def _begin(self, rollout, a, plan, prefix, s, seq):
        """A plan starts after `prefix` (the answer's tokens up to the value); the event names the shim's slot and seq, so a plan
        the shim gave up waiting for (llama.log names it) is told from a value the model wrote itself."""
        if a.plan is not None:
            self._drop(a)   # a new plan replaces one not finished: the old one is counted dropped, not left open
        at = len(prefix)
        a.plan, a.at, a.node, a.prefix = plan, at, 0, list(prefix)
        a.event = dict(kind='steered value', expr=plan.expr[:200], value=plan.value, what=plan.kind, at=at, done=False,
                       slot=s.get('slot'), seq=seq)
        rollout.events.append(a.event)
        self.counts['plans'] += 1

    def _resume(self, rollout, a):
        """The rest of a plan a previous request of this answer ended inside, read against the answer's own text."""
        if a.plan is None:
            return None
        if rollout.phase != 'think' or len(rollout.out) < a.at or rollout.out[:a.at] != a.prefix:
            self._drop(a)   # the driver cut or closed the text since: the value's place is gone, never steer it elsewhere
            return None
        written = self.tok.decode(rollout.out[a.at:])
        full = a.plan.text
        if written.startswith(full):
            self._finish(a)
            return None
        if not full.startswith(written):
            self._drop(a)
            return None
        rest = full[len(written):]
        dag = ValueDag.build(rest, self.pieces)
        if dag is None:
            self._drop(a)
            return None
        a.node = 0
        a.plan = Plan(rest, dag, a.plan.kind, a.plan.after_ban, a.plan.expr, a.plan.value)
        a.at = len(rollout.out)
        return a.plan

    def _advance(self, s, token):
        with self.lock:
            rollout, _, a = self._lookup(s)
            if a is None or a.plan is None:
                return
            nxt = a.plan.dag.step(a.node, token)
            if nxt is None:
                self._drop(a)
            elif nxt == a.plan.dag.end:
                self._finish(a)
            else:
                a.node = nxt

    def _finish(self, a):
        if a.event is not None:
            a.event['done'] = True
        a.plan, a.event = None, None
        self.counts['steered'] += 1

    def _drop(self, a):
        if a.event is not None:
            a.event['dropped'] = True
        a.plan, a.event = None, None
        self.counts['dropped'] += 1

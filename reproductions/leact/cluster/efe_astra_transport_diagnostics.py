"""Opt-in observations of the teacher's existing Harbor transport.

Only the separately frozen teacher remote-worker entrypoint calls ``install``.
The original bridge, action loop, transport deadlines, and verifier run as-is.
Diagnostic writes are best effort and never enter the model context.
"""
from __future__ import annotations

import functools
import hashlib
import json
import os
from pathlib import Path
import re
import threading
import time
from uuid import uuid4


VERSION = 'astra-transport-lifecycle-v1'
_context = threading.local()
_SECRET = re.compile(r'\b(?:sk-|glpat-)[A-Za-z0-9_-]+|(?i:bearer)\s+\S+')


def _safe_text(value, limit=12000):
    return _SECRET.sub('[REDACTED]', str(value))[-limit:]


def _error(exc):
    return {'error_type': type(exc).__name__, 'error': _safe_text(exc, 1000),
            'errno': getattr(exc, 'errno', None)}


def _read_tail(path, limit=12000):
    with path.open('rb') as stream:
        stream.seek(0, os.SEEK_END)
        size = stream.tell()
        stream.seek(max(0, size-limit))
        text = stream.read(limit).decode(errors='replace')
    return {'path': str(path), 'bytes': size, 'tail': _safe_text(text, limit)}


class Lifecycle:
    def __init__(self, bridge, transport_folder=None):
        self.bridge = bridge
        self.transport_folder = Path(transport_folder) if transport_folder else None
        self.lock = threading.RLock()
        self.local = threading.local()
        self.started = time.monotonic()
        self.state = {'version': VERSION, 'episode_id': bridge.episode_id,
                      'scope': 'Teacher-only transport diagnostics; no reward or policy change.',
                      'events': 0, 'diagnostic_write_errors': 0}
        if self.transport_folder:
            self.state['transport_episode_id'] = self.transport_folder.name
        self.record('bridge_created')

    def record(self, event, **details):
        """A full disk or unavailable share must not change an episode outcome."""
        try:
            with self.lock:
                row = {'event': event, 'unix': time.time(),
                       'elapsed_seconds': time.monotonic()-self.started,
                       'native_call': self.bridge.calls, **details}
                request_id = getattr(self.local, 'request_id', None)
                if request_id:
                    row['request_id'] = request_id
                self.state.update(last_event=row, events=self.state['events']+1)
                if event == 'http_request_started':
                    self.state['last_request'] = row
                elif event == 'response_write_failed':
                    self.state['last_response_failure'] = row
                elif event == 'http_handler_finished':
                    self.state['last_response'] = row
                elif event.startswith('bridge_') and event.endswith('_exception'):
                    self.state['last_bridge_exception'] = row
                path = self.bridge.root/'teacher-transport-events.jsonl'
                descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
                with os.fdopen(descriptor, 'a') as stream:
                    stream.write(json.dumps(row, allow_nan=False)+'\n')
                self._persist()
        except Exception:
            self.state['diagnostic_write_errors'] += 1

    def _persist(self):
        path = self.bridge.root/'teacher-transport-diagnostics.json'
        temporary = path.with_name(path.name+'.'+uuid4().hex+'.tmp')
        try:
            descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(descriptor, 'w') as stream:
                json.dump(self.state, stream, allow_nan=False)
                stream.write('\n')
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)

    def snapshot(self, phase):
        try:
            with self.lock:
                process = self.bridge.process
                value = {'phase': phase, 'unix': time.time(),
                         'harbor_returncode': process.poll() if process else None,
                         'harbor_pid': getattr(process, 'pid', None),
                         'bridge_closed': self.bridge.closed,
                         'observation_waiting': self.bridge.current is not None,
                         'action_waiting': not self.bridge.actions.empty()}
                trial_root = self.bridge.root/'jobs'/self.bridge.episode_id
                results = sorted(trial_root.glob('*/result.json'))
                value['native_trial_result_paths'] = [str(p) for p in results]
                value['job_summary_exists'] = (trial_root/'result.json').exists()
                exceptions = []
                for path in trial_root.glob('*/exception.txt'):
                    try:
                        exceptions.append((path.stat().st_mtime, path))
                    except OSError:
                        continue
                exceptions.sort()
                value['last_trial_exception'] = None
                if exceptions:
                    try:
                        value['last_trial_exception'] = _read_tail(exceptions[-1][1])
                    except OSError as exc:
                        value['trial_exception_read_error'] = _error(exc)
                if self.transport_folder:
                    signals = {}
                    for name in ['cancel.json', 'infrastructure-error.json']:
                        path = self.transport_folder/name
                        if not path.exists():
                            continue
                        try:
                            if path.stat().st_size > 128*1024:
                                raise ValueError('Transport signal exceeds bounded diagnostic size')
                            raw = path.read_bytes()
                            item = json.loads(raw)
                        except (OSError, ValueError) as exc:
                            signals[name] = {'read_error': _error(exc)}
                            continue
                        keys = ['closed_unix', 'native_worker_stopping', 'native_child_exit',
                                'reconciled_unix', 'lost_native_worker', 'error']
                        signals[name] = {'sha256': hashlib.sha256(raw).hexdigest(),
                                         'fields': {k: _safe_text(item[k], 1000)
                                                    if isinstance(item[k], str) else item[k]
                                                    for k in keys if k in item}}
                    value['transport_signals'] = signals
                    value['cancellation_reason'] = (
                        'native_worker_stopping' if signals.get('cancel.json', {}).get(
                            'fields', {}).get('native_worker_stopping') else
                        'client_closed' if 'closed_unix' in signals.get('cancel.json', {}).get('fields', {}) else
                        'transport_cancel_signal' if 'cancel.json' in signals else
                        'no_cancel_signal_observed')
                self.state[phase] = value
                self.record('lifecycle_snapshot', phase=phase)
        except Exception:
            self.state['diagnostic_write_errors'] += 1


class _ObservedQueue:
    def __init__(self, original, lifecycle, name):
        self.original, self.lifecycle, self.name = original, lifecycle, name

    def __getattr__(self, name):
        return getattr(self.original, name)

    def put(self, item, *args, **kwargs):
        result = self.original.put(item, *args, **kwargs)
        self.lifecycle.record(self.name+'_put', item_is_exception=isinstance(item, Exception))
        return result

    def put_nowait(self, item):
        return self.put(item, block=False)

    def get(self, *args, **kwargs):
        value = self.original.get(*args, **kwargs)
        self.lifecycle.record(self.name+'_get', item_is_exception=isinstance(value, Exception))
        return value


class _ObservedWriter:
    def __init__(self, original, lifecycle):
        self.original, self.lifecycle = original, lifecycle

    def __getattr__(self, name):
        return getattr(self.original, name)

    def write(self, data):
        before = time.monotonic()
        try:
            result = self.original.write(data)
        except Exception as exc:
            self.lifecycle.record('response_write_failed', bytes_attempted=len(data),
                                  write_seconds=time.monotonic()-before, **_error(exc))
            raise
        self.lifecycle.record('response_write_completed', bytes_attempted=len(data),
                              write_seconds=time.monotonic()-before)
        return result


def _handler(original, lifecycle):
    class ObservedHandler(original):
        def do_POST(self):
            lifecycle.local.request_id = uuid4().hex
            before = time.monotonic()
            self._diagnostic_status = None
            self.wfile = _ObservedWriter(self.wfile, lifecycle)
            lifecycle.record('http_request_started',
                              content_length=_safe_text(self.headers.get('Content-Length'), 60))
            try:
                return super().do_POST()
            except Exception as exc:
                lifecycle.record('http_handler_exception', **_error(exc))
                raise
            finally:
                lifecycle.record('http_handler_finished', http_status=self._diagnostic_status,
                                  request_seconds=time.monotonic()-before)
                lifecycle.local.request_id = None

        def send_response(self, code, message=None):
            self._diagnostic_status = code
            lifecycle.record('http_response_headers', http_status=code)
            return super().send_response(code, message)

    return ObservedHandler


def install(bridge_module=None, rpc_module=None):
    """Instrument this explicit teacher process only; never patch stdlib globals."""
    if bridge_module is None:
        import efe_pear_bridge as bridge_module
    if rpc_module is None:
        import efe_pear_file_rpc as rpc_module
    if getattr(bridge_module, '_astra_transport_diagnostics', None) == VERSION:
        return
    original_bridge = bridge_module.HarborEpisode
    original_server = bridge_module.ThreadingHTTPServer
    original_serve = rpc_module.serve_episode

    class ObservedServer(original_server):
        def __init__(self, address, handler, *args, **kwargs):
            lifecycle = getattr(_context, 'lifecycle', None)
            super().__init__(address, _handler(handler, lifecycle) if lifecycle else handler,
                             *args, **kwargs)

    class ObservedBridge(original_bridge):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.teacher_lifecycle = Lifecycle(self, getattr(_context, 'transport_folder', None))
            self.requests = _ObservedQueue(self.requests, self.teacher_lifecycle, 'requests')
            self.actions = _ObservedQueue(self.actions, self.teacher_lifecycle, 'actions')

        def start(self):
            previous = getattr(_context, 'lifecycle', None)
            _context.lifecycle = self.teacher_lifecycle
            self.teacher_lifecycle.record('harbor_start_requested')
            try:
                return super().start()
            except Exception as exc:
                self.teacher_lifecycle.record('bridge_start_exception', **_error(exc))
                self.teacher_lifecycle.snapshot('start_exception')
                raise
            finally:
                _context.lifecycle = previous

        def next(self):
            try:
                return super().next()
            except Exception as exc:
                self.teacher_lifecycle.record('bridge_next_exception', **_error(exc))
                self.teacher_lifecycle.snapshot('next_exception')
                raise

        def finish(self):
            self.teacher_lifecycle.snapshot('before_finish')
            try:
                return super().finish()
            except Exception as exc:
                self.teacher_lifecycle.record('bridge_finish_exception', **_error(exc))
                raise
            finally:
                self.teacher_lifecycle.snapshot('after_finish')

        def close(self):
            self.teacher_lifecycle.snapshot('before_close')
            try:
                return super().close()
            finally:
                # NativeEfeEpisode archives/prunes only after bridge.close().
                self.teacher_lifecycle.snapshot('after_close')

    @functools.wraps(original_serve)
    def observed_serve(folder, *args, **kwargs):
        previous = getattr(_context, 'transport_folder', None)
        _context.transport_folder = Path(folder)
        try:
            return original_serve(folder, *args, **kwargs)
        finally:
            _context.transport_folder = previous

    bridge_module.ThreadingHTTPServer = ObservedServer
    bridge_module.HarborEpisode = ObservedBridge
    bridge_module._astra_transport_diagnostics = VERSION
    rpc_module.serve_episode = observed_serve

"""Bounded Responses transport for the EFE research pilot.

The generation budget is capped; an optional shared image-history bound replaces
the oldest image inputs with explicit omission markers. Text and tools pass through.
The provider key stays outside the agent container. Logs retain public inputs,
public outputs and usage, never encrypted/provider-private reasoning fields.
This adapter is a recorded research-harness change, not an official EFE run.
"""
from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import time
import uuid

from aiohttp import ClientSession, ClientTimeout, web


def public_value(value, secrets=()):
    if isinstance(value, dict):
        if value.get('type') == 'reasoning':
            return None
        return {k: public_value(v, secrets) for k, v in value.items()
                if k not in {'encrypted_content', 'reasoning_content', 'reasoning_details'}
                and not (isinstance(v, dict) and v.get('type') == 'reasoning')}
    if isinstance(value, list):
        return [public_value(v, secrets) for v in value
                if not (isinstance(v, dict) and v.get('type') == 'reasoning')]
    if isinstance(value, str):
        for secret in secrets:
            if secret:
                value = value.replace(secret, '[REDACTED]')
    return value


def capped_request(body, model, cap):
    if body.get('model') != model:
        raise ValueError('Unexpected request model')
    old = body.get('max_output_tokens')
    if old is not None and (not isinstance(old, int) or isinstance(old, bool) or old <= 0):
        raise ValueError('Invalid max_output_tokens')
    result = dict(body)
    result['max_output_tokens'] = min(old, cap) if old is not None else cap
    return result


def image_count(value):
    if isinstance(value,dict):
        if value.get('type')=='input_image':return 1
        return sum(image_count(v) for v in value.values())
    if isinstance(value,list):return sum(image_count(v) for v in value)
    return 0


def bounded_images(body, maximum):
    """Preserve recent images and all text, without mutating the source request."""
    total=image_count(body.get('input',[]))
    remaining=max(0,total-maximum) if maximum else 0
    dropped=remaining
    def visit(value):
        nonlocal remaining
        if isinstance(value,dict):
            if value.get('type')=='input_image':
                if remaining:
                    remaining-=1
                    return {'type':'input_text','text':f'[Earlier image omitted by the shared {maximum}-image history limit.]'}
                return value
            return {k:visit(v) for k,v in value.items()}
        if isinstance(value,list):return [visit(v) for v in value]
        return value
    result={**body,'input':visit(body.get('input',[]))} if dropped else body
    return result,{'original_images':total,'forwarded_images':total-dropped,'omitted_images':dropped,'max_images':maximum or None}


def flatten_namespaces(body):
    """Preserve native tools/history while supplying vLLM's required metadata."""
    mapping = {}
    tools = []
    names = {tool['name'] for tool in body.get('tools', []) if tool.get('type') == 'function'}
    for tool in body.get('tools', []):
        if tool.get('type') != 'namespace':
            tools.append(tool)
            continue
        namespace = tool['name']
        for child in tool['tools']:
            if child.get('type') != 'function':
                raise ValueError('Namespace contains a non-function tool')
            name = namespace + '__' + child['name']
            if name in names or len(name) > 64:
                raise ValueError('Ambiguous or oversized flattened tool name')
            names.add(name)
            mapping[name] = {'name': child['name'], 'namespace': namespace}
            description = '\n'.join(x for x in (tool.get('description'), child.get('description')) if x)
            tools.append({**child, 'name': name, 'description': description})
    reverse = {(v['namespace'], v['name']): k for k, v in mapping.items()}
    def visit(value):
        if isinstance(value, list):
            return [visit(x) for x in value]
        if isinstance(value, dict):
            result = {k: visit(v) for k, v in value.items()}
            if value.get('type') == 'input_image':
                # vLLM validates the SDK TypedDict, where the API default is required.
                result.setdefault('detail', 'auto')
            if value.get('type') == 'message' and value.get('role') == 'assistant':
                # Codex sends completed history without status; vLLM's output-message
                # TypedDict requires it when the content consists of output_text.
                result.setdefault('status', 'completed')
            if value.get('type') == 'output_text':
                result.setdefault('annotations', [])
            if value.get('type') in {'function', 'function_call'} and value.get('namespace'):
                key = (value['namespace'], value['name'])
                if key not in reverse:
                    raise ValueError('Historical namespace function absent from current tool schema')
                result['name'] = reverse[key]
                result.pop('namespace')
            return result
        return value
    return {**body, 'tools': tools, 'input': visit(body.get('input', [])),
            **({'tool_choice': visit(body['tool_choice'])} if 'tool_choice' in body else {})}, mapping


def restore_namespaces(value, mapping):
    if isinstance(value, list):
        return [restore_namespaces(x, mapping) for x in value]
    if isinstance(value, dict):
        result = {k: restore_namespaces(v, mapping) for k, v in value.items()}
        if value.get('type') == 'function_call' and value.get('name') in mapping:
            result.update(mapping[value['name']])
        return result
    return value


def aligned_instructions(body):
    """Named student condition: align the generic Codex fallback with its tools.

    Task/developer messages, tool schemas and observations are unchanged. This
    is frozen across the base and trained students, never applied to Astra.
    """
    original = body.get('instructions')
    if not isinstance(original, str) or not original:
        return body, None
    names = [t['name'] for t in body.get('tools', []) if t.get('type') == 'function']
    executable = next((n for n in names if n.rsplit('__', 1)[-1] == 'exec_command'), None)
    if not executable:
        return body, None
    patch_available = any(n.rsplit('__', 1)[-1] == 'apply_patch' for n in names)
    lines = original.splitlines()
    kept, removed = [], []
    for line in lines:
        if ('apply_patch' in line and not patch_available) or line.startswith("If there's something that you think you could help with as a logical next step") or line.startswith('If there\u2019s something that you think you could help with as a logical next step'):
            removed.append(line)
        else:
            kept.append(line)
    addition = ('\n\n## Executable tool contract for this non-interactive robot trial\n'
                'Use only the functions declared in this request. Available function names: '
                + ', '.join(names) + '.\n'
                f'Use {executable} to create/edit files with shell commands and to run your controller. '
                'The task requires acting in the live episode: execute the controller you write, '
                'inspect its observations and continue until the task is complete or its budget ends. '
                'Do not stop after writing code or ask permission to execute the requested task. '
                'Follow all task rules, resource limits and the no-reset restriction.\n')
    revised = '\n'.join(kept) + addition
    receipt = {'condition': 'native-tool-aligned-v1', 'removed_lines': removed,
               'source_sha256': hashlib.sha256(original.encode()).hexdigest(),
               'forwarded_sha256': hashlib.sha256(revised.encode()).hexdigest(),
               'available_functions': names}
    return {**body, 'instructions': revised}, receipt


def adapt_developer_roles(body):
    """Represent developer text in Qwen's system preamble, retaining provenance.

    Qwen3.6's pinned template accepts system/user/assistant/tool roles only.
    This opt-in adapter keeps every developer text block in order, labels its
    lower priority relative to the original system instructions, and leaves
    user observations and tool histories unchanged. It is a recorded research
    serving condition, not a claim of tokenizer-level parity across models.
    """
    items = body.get('input')
    if not isinstance(items, list):
        return body, None
    retained, developer = [], []
    for index, item in enumerate(items):
        if not isinstance(item, dict) or item.get('role') != 'developer':
            retained.append(item)
            continue
        content = item.get('content')
        if isinstance(content, str):
            blocks = [content]
        elif isinstance(content, list) and all(isinstance(x, dict) and
                x.get('type') in {'input_text', 'text'} and isinstance(x.get('text'), str)
                for x in content):
            blocks = [x['text'] for x in content]
        else:
            raise ValueError('Developer-role adapter supports text instructions only')
        developer.append({'source_index': index, 'blocks': blocks})
    if not developer:
        return body, None
    original = body.get('instructions', '')
    if not isinstance(original, str):
        raise ValueError('Developer-role adapter requires string system instructions')
    addition = ('\n\n## Developer instructions\n'
                'The following developer instructions are subordinate to the system '
                'instructions above and take precedence over user and tool content. '
                'Their original order is retained.\n')
    for number, row in enumerate(developer, 1):
        addition += f'\n### Developer message {number}\n' + '\n'.join(row['blocks']) + '\n'
    revised = original + addition
    receipt = {'condition': 'developer-to-system-preamble-v1',
        'message_count': len(developer),
        'source_indices': [r['source_index'] for r in developer],
        'text_block_sha256': [[hashlib.sha256(b.encode()).hexdigest() for b in r['blocks']]
                             for r in developer],
        'source_instructions_sha256': hashlib.sha256(original.encode()).hexdigest(),
        'forwarded_instructions_sha256': hashlib.sha256(revised.encode()).hexdigest(),
        'history_note': 'Developer instructions collected into the system preamble in original order; other input items retain order and content.'}
    return {**body, 'input': retained, 'instructions': revised}, receipt


def completed_output_events(result):
    """Emit native tool events from vLLM's parsed final output, without changing it.

    vLLM 0.16 streams tool syntax as text, then parses tools only in the final
    response. Codex executes output_item.done calls, so use that parsed output.
    """
    for index, item in enumerate(result.get('output', [])):
        ident = item['id']
        pending = {**item, 'status': 'in_progress'}
        if item['type'] == 'function_call':
            pending['arguments'] = ''
        elif item['type'] == 'message':
            pending['content'] = []
        yield {'type': 'response.output_item.added', 'output_index': index, 'item': pending}
        if item['type'] == 'function_call':
            common = {'item_id': ident, 'output_index': index}
            yield {'type': 'response.function_call_arguments.delta', **common, 'delta': item['arguments']}
            yield {'type': 'response.function_call_arguments.done', **common, 'arguments': item['arguments']}
        elif item['type'] == 'message':
            for content_index, part in enumerate(item.get('content', [])):
                if part['type'] != 'output_text':
                    continue
                common = {'item_id': ident, 'output_index': index, 'content_index': content_index}
                yield {'type': 'response.content_part.added', **common, 'part': {**part, 'text': ''}}
                yield {'type': 'response.output_text.delta', **common, 'delta': part['text']}
                yield {'type': 'response.output_text.done', **common, 'text': part['text']}
                yield {'type': 'response.content_part.done', **common, 'part': part}
        yield {'type': 'response.output_item.done', 'output_index': index, 'item': item}


class Gateway:
    def __init__(self, root, upstream, model, cap, key, token, max_images=0, namespace_adapter=False,
                 terminal_output_events=False, align_tool_instructions=False, instruction_suffix='',
                 developer_role_adapter=False):
        self.root, self.upstream, self.model, self.cap = Path(root), upstream.rstrip('/'), model, cap
        self.key, self.token = key, token
        self.max_images=max_images
        self.namespace_adapter = namespace_adapter
        self.terminal_output_events = terminal_output_events
        self.align_tool_instructions = align_tool_instructions
        self.instruction_suffix = instruction_suffix
        self.developer_role_adapter = developer_role_adapter
        self.root.mkdir(parents=True, exist_ok=True)
        self.session = None

    async def start(self, app):
        self.session = ClientSession(timeout=ClientTimeout(total=None, connect=30, sock_read=600),
                                     auto_decompress=False)

    async def stop(self, app):
        await self.session.close()

    def write(self, name, value, compressed=False):
        value = public_value(value, (self.key, self.token))
        data = (json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n').encode()
        path = self.root / name
        if compressed:
            data = gzip.compress(data)
        temp = path.with_suffix(path.suffix + '.tmp')
        temp.write_bytes(data)
        temp.replace(path)

    async def handle(self, request):
        if request.headers.get('Authorization') != 'Bearer ' + self.token:
            raise web.HTTPUnauthorized()
        if request.method != 'POST' or request.path != '/v1/responses' or request.query_string:
            raise web.HTTPNotFound()
        try:
            raw = await request.read()
            body = json.loads(raw)
            forwarded = capped_request(body, self.model, self.cap)
            forwarded, image_receipt = bounded_images(forwarded,self.max_images)
            namespace_mapping = {}
            if self.namespace_adapter:
                forwarded, namespace_mapping = flatten_namespaces(forwarded)
            instruction_alignment = None
            if self.align_tool_instructions:
                forwarded, instruction_alignment = aligned_instructions(forwarded)
            if self.instruction_suffix:
                if not isinstance(forwarded.get('instructions', ''), str):
                    raise ValueError('Method protocol requires string instructions')
                forwarded['instructions'] = forwarded.get('instructions', '') + '\n\n' + self.instruction_suffix
            role_adaptation = None
            if self.developer_role_adapter:
                forwarded, role_adaptation = adapt_developer_roles(forwarded)
        except (ValueError, TypeError, json.JSONDecodeError):
            raise web.HTTPBadRequest(text='Invalid bounded Responses request')
        ident = uuid.uuid4().hex
        start = time.monotonic()
        record = {'id': ident, 'started_utc': datetime.now(timezone.utc).isoformat(),
                  'requested_model': self.model, 'max_output_tokens': forwarded['max_output_tokens'],
                  'original_max_output_tokens': body.get('max_output_tokens'),
                  'request_sha256': hashlib.sha256(raw).hexdigest(),
                  'request_keys': sorted(body), 'stream': bool(body.get('stream')),
                  'status': 'pending', 'public_output': [], 'image_history':image_receipt,
                  'namespace_mapping': namespace_mapping,
                  'instruction_alignment': instruction_alignment,
                  'developer_role_adaptation': role_adaptation,
                  'instruction_suffix_sha256': hashlib.sha256(self.instruction_suffix.encode()).hexdigest() if self.instruction_suffix else None,
                  'terminal_output_events': self.terminal_output_events}
        self.write(ident + '.request.json.gz', forwarded, compressed=True)
        if role_adaptation:
            self.write(ident + '.source-request.json.gz', body, compressed=True)
        self.write(ident + '.json', record)
        headers = {'Authorization': 'Bearer ' + self.key, 'Content-Type': 'application/json',
                   'Accept-Encoding': 'identity'}
        try:
            async with self.session.post(self.upstream + '/responses', json=forwarded, headers=headers) as response:
                record['http_status'] = response.status
                out = web.StreamResponse(status=response.status,
                                         headers={'Content-Type': response.headers.get('Content-Type', 'application/json'),
                                                  'Cache-Control': 'no-cache'})
                await out.prepare(request)
                buffer = b''
                body_parts = []
                sequence_number = 0
                async def emit(event):
                    nonlocal sequence_number
                    event = {**event, 'sequence_number': sequence_number}
                    sequence_number += 1
                    await out.write(b'data: ' + json.dumps(event).encode() + b'\n\n')
                rewrite = bool(namespace_mapping) or self.terminal_output_events
                async for chunk in response.content.iter_any():
                    record.setdefault('first_upstream_byte_seconds', time.monotonic() - start)
                    if not rewrite:
                        await out.write(chunk)
                    if forwarded.get('stream') and response.status == 200:
                        buffer += chunk
                        while b'\n' in buffer:
                            line, buffer = buffer.split(b'\n', 1)
                            if rewrite and not self.terminal_output_events:
                                if line.startswith(b'data:'):
                                    try:
                                        mapped = restore_namespaces(json.loads(line[5:].strip()), namespace_mapping)
                                        line = b'data: ' + json.dumps(mapped).encode()
                                    except (ValueError, TypeError):
                                        pass
                                await out.write(line + b'\n')
                            if not line.startswith(b'data:'):
                                continue
                            try:
                                event = restore_namespaces(json.loads(line[5:].strip()), namespace_mapping)
                            except (ValueError, TypeError):
                                continue
                            kind = event.get('type', '')
                            if self.terminal_output_events:
                                if kind in {'response.created', 'response.in_progress', 'error'}:
                                    await emit(event)
                                elif kind in {'response.completed', 'response.incomplete', 'response.failed'}:
                                    for output_event in completed_output_events(event.get('response', {})):
                                        await emit(output_event)
                                    await emit(event)
                            if kind in {'response.completed', 'response.incomplete', 'response.failed'}:
                                result = event.get('response', {})
                                record.update(status=result.get('status', kind), usage=result.get('usage'),
                                              returned_model=result.get('model'),
                                              public_output=public_value(result.get('output', [])),
                                              error=public_value(result.get('error')),
                                              incomplete_details=result.get('incomplete_details'))
                            elif kind == 'error':
                                record['error'] = public_value(event)
                    else:
                        body_parts.append(chunk)
                if body_parts:
                    try:
                        result = json.loads(b''.join(body_parts))
                        if rewrite:
                            result = restore_namespaces(result, namespace_mapping)
                            await out.write(json.dumps(result).encode())
                        record.update(status=result.get('status', 'http_error' if response.status >= 400 else 'completed'),
                                      usage=result.get('usage'), returned_model=result.get('model'),
                                      public_output=public_value(result.get('output', [])),
                                      error=public_value(result.get('error')))
                    except (ValueError, TypeError):
                        record['status'] = 'non_json_response'
                        if rewrite:
                            await out.write(b''.join(body_parts))
                if rewrite and buffer and not self.terminal_output_events:
                    await out.write(buffer)
                await out.write_eof()
                return out
        except (asyncio.CancelledError, Exception) as exc:
            record['status'] = 'transport_error'
            record['error_type'] = type(exc).__name__
            raise
        finally:
            record['elapsed_seconds'] = time.monotonic() - start
            self.write(ident + '.json', record)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--max-output-tokens', type=int, default=16384)
    parser.add_argument('--upstream', default='https://inference-api.nvidia.com/v1')
    parser.add_argument('--max-images',type=int,default=0)
    parser.add_argument('--flatten-namespaces', action='store_true')
    parser.add_argument('--terminal-output-events', action='store_true')
    parser.add_argument('--align-tool-instructions', action='store_true')
    parser.add_argument('--instruction-suffix-file', type=Path)
    parser.add_argument('--developer-role-adapter', action='store_true')
    args = parser.parse_args()
    if args.max_output_tokens <= 0 or args.max_images<0:
        parser.error('max output tokens must be positive and max images nonnegative')
    gateway = Gateway(args.root, args.upstream, args.model, args.max_output_tokens,
                      os.environ['NVIDIA_INFERENCE_API_KEY'], os.environ['EFE_GATEWAY_TOKEN'],args.max_images,
                      args.flatten_namespaces,args.terminal_output_events,args.align_tool_instructions,
                      args.instruction_suffix_file.read_text() if args.instruction_suffix_file else '',
                      args.developer_role_adapter)
    app = web.Application(client_max_size=128 * 1024 * 1024)
    app.router.add_route('*', '/{tail:.*}', gateway.handle)
    app.on_startup.append(gateway.start)
    app.on_cleanup.append(gateway.stop)
    web.run_app(app, host='0.0.0.0', port=args.port, print=None, access_log=None)


if __name__ == '__main__':
    main()

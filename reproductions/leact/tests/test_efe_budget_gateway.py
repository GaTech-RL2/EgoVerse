import gzip
import json
from pathlib import Path
import tempfile
import unittest

from aiohttp import ClientSession, web

from reproductions.leact.cluster.efe_budget_gateway import Gateway, capped_request, bounded_images, flatten_namespaces, aligned_instructions, adapt_developer_roles


class GatewayTest(unittest.IsolatedAsyncioTestCase):
    async def test_developer_adapter_retains_instructions_and_auditable_original_request(self):
        self.gateway.developer_role_adapter = True
        body = {'model': 'astra-test', 'stream': True, 'instructions': 'System: obey task limits.',
                'input': [{'role': 'developer', 'content': [{'type': 'input_text', 'text': 'No simulator reset.'}]},
                          {'role': 'user', 'content': 'Move the gripper.'},
                          {'type': 'function_call_output', 'call_id': 'c1', 'output': 'position=[0,0,0]'},
                          {'role': 'developer', 'content': 'Use only the declared tools.'}]}
        original = json.dumps(body)
        async with self.client.post(self.url + '/v1/responses', json=body,
                                    headers={'Authorization': 'Bearer ' + self.local_token}) as response:
            self.assertEqual(response.status, 200)
            await response.read()
        forwarded = self.seen[0][0]
        self.assertEqual(forwarded['input'], body['input'][1:3])
        self.assertTrue(forwarded['instructions'].startswith(body['instructions']))
        self.assertLess(forwarded['instructions'].index('No simulator reset.'),
                        forwarded['instructions'].index('Use only the declared tools.'))
        self.assertIn('subordinate to the system', forwarded['instructions'])
        self.assertEqual(json.dumps(body), original)
        records = list(Path(self.temp.name).glob('*.json'))
        receipt = json.loads(records[0].read_text())
        self.assertEqual(receipt['developer_role_adaptation']['source_indices'], [0, 3])
        source = next(Path(self.temp.name).glob('*.source-request.json.gz'))
        self.assertEqual(json.loads(gzip.decompress(source.read_bytes())), body)

    def test_developer_adapter_rejects_nontext_without_silent_loss(self):
        with self.assertRaises(ValueError):
            adapt_developer_roles({'input': [{'role': 'developer', 'content': [
                {'type': 'input_image', 'image_url': 'data:image/png;base64,AAAA'}]}]})
        body = {'instructions': 'system', 'input': [{'role': 'user', 'content': 'task'}]}
        self.assertEqual(adapt_developer_roles(body), (body, None))

    def test_student_alignment_preserves_task_and_uses_only_available_functions(self):
        body = {'instructions': 'Respect task limits.\nUse the `apply_patch` tool to edit files.\nKeep observations private.',
                'tools': [{'type': 'function', 'name': 'functions__exec_command'}],
                'input': [{'role': 'user', 'content': 'Move the object. No reset.'}]}
        before = json.dumps(body)
        result, receipt = aligned_instructions(body)
        self.assertNotIn('apply_patch', result['instructions'])
        self.assertIn('functions__exec_command', result['instructions'])
        self.assertIn('Respect task limits.', result['instructions'])
        self.assertEqual(result['input'], body['input'])
        self.assertEqual(result['tools'], body['tools'])
        self.assertEqual(json.dumps(body), before)
        self.assertEqual(len(receipt['removed_lines']), 1)
        self.assertEqual(aligned_instructions({'input': []}), ({'input': []}, None))

    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.seen = []
        self.provider_key = 'provider-test-secret-12345678'
        self.local_token = 'local-test-secret-12345678'
        self.wire = ('data: ' + json.dumps({'type': 'response.completed', 'response': {
            'status': 'completed', 'model': 'astra-test',
            'output': [{'type': 'reasoning', 'encrypted_content': 'private-test-payload'},
                       {'type': 'function_call', 'name': 'shell', 'arguments': '{"command":"pwd"}'}],
            'usage': {'output_tokens': 12}}}) + '\n\n').encode()

        async def upstream(request):
            self.seen.append((await request.json(), request.headers.get('Authorization')))
            return web.Response(body=self.wire, content_type='text/event-stream')

        app = web.Application()
        app.router.add_post('/v1/responses', upstream)
        self.up_runner, self.up_url = await self.serve(app)
        self.gateway = Gateway(self.temp.name, self.up_url + '/v1', 'astra-test', 16384,
                               self.provider_key, self.local_token)
        app = web.Application()
        app.router.add_route('*', '/{tail:.*}', self.gateway.handle)
        app.on_startup.append(self.gateway.start)
        app.on_cleanup.append(self.gateway.stop)
        self.runner, self.url = await self.serve(app)
        self.client = ClientSession()

    async def serve(self, app):
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, '127.0.0.1', 0)
        await site.start()
        return runner, 'http://127.0.0.1:' + str(site._server.sockets[0].getsockname()[1])

    async def asyncTearDown(self):
        await self.client.close()
        await self.runner.cleanup()
        await self.up_runner.cleanup()
        self.temp.cleanup()

    async def test_stream_preserved_only_budget_changed_and_private_fields_not_exported(self):
        body = {'model': 'astra-test', 'stream': True, 'input': [
            {'type': 'message', 'role': 'user', 'content': 'task'},
            {'type': 'reasoning', 'encrypted_content': 'private-input'}],
            'tools': [{'type': 'function', 'name': 'shell', 'parameters': {'type': 'object'}}]}
        async with self.client.post(self.url + '/v1/responses', json=body,
                                    headers={'Authorization': 'Bearer ' + self.local_token}) as response:
            self.assertEqual(response.status, 200)
            self.assertEqual(await response.read(), self.wire)
        self.assertEqual(self.seen, [({**body, 'max_output_tokens': 16384}, 'Bearer ' + self.provider_key)])
        saved = list(Path(self.temp.name).glob('*.json'))
        self.assertEqual(len(saved), 1)
        summary = json.loads(saved[0].read_text())
        self.assertEqual(summary['status'], 'completed')
        self.assertEqual(summary['public_output'][0]['type'], 'function_call')
        data = saved[0].read_bytes() + gzip.decompress(next(Path(self.temp.name).glob('*.gz')).read_bytes())
        for secret in (self.provider_key, self.local_token, 'private-test-payload', 'private-input'):
            self.assertNotIn(secret.encode(), data)

    async def test_wrong_route_auth_or_model_never_reaches_upstream(self):
        for path, token, model, expected in [('/v1/responses', 'wrong', 'astra-test', 401),
                                            ('/v1/files', self.local_token, 'astra-test', 404),
                                            ('/v1/responses', self.local_token, 'wrong', 400)]:
            async with self.client.post(self.url + path, json={'model': model},
                                        headers={'Authorization': 'Bearer ' + token}) as response:
                self.assertEqual(response.status, expected)
        self.assertEqual(self.seen, [])

    async def test_namespaced_tool_and_result_roundtrip_preserves_native_call_identity(self):
        self.gateway.namespace_adapter = True
        native_call = {'type':'function_call','name':'inspect','namespace':'robot','call_id':'call-1','arguments':'{}'}
        body = {'model':'astra-test','stream':True,'input':[native_call,
                {'type':'function_call_output','call_id':'call-1','output':'green'}],
                'tools':[{'type':'namespace','name':'robot','description':'Robot actions','tools':[
                    {'type':'function','name':'inspect','description':'Inspect scene','parameters':{'type':'object'}}]}]}
        self.wire = ('data: '+json.dumps({'type':'response.output_item.added','item':{
            **native_call,'name':'robot__inspect','namespace':None}})+'\n\n'+'data: '+json.dumps({
            'type':'response.completed','response':{'status':'completed','output':[{
                **native_call,'name':'robot__inspect','namespace':None}]}})+'\n\n').encode()
        async with self.client.post(self.url+'/v1/responses',json=body,
                                    headers={'Authorization':'Bearer '+self.local_token}) as response:
            self.assertEqual(response.status,200)
            events=[json.loads(line[6:]) for line in (await response.text()).splitlines() if line.startswith('data: ')]
        self.assertEqual(events[0]['item'],native_call)
        self.assertEqual(events[1]['response']['output'],[native_call])
        forwarded=self.seen[0][0]
        self.assertEqual(forwarded['tools'][0]['name'],'robot__inspect')
        self.assertEqual(forwarded['input'][0]['name'],'robot__inspect')
        self.assertNotIn('namespace',forwarded['input'][0])
        self.assertEqual(forwarded['input'][1],body['input'][1])
        self.assertEqual(body['input'][0],native_call)

    def test_namespace_name_collision_is_rejected(self):
        with self.assertRaises(ValueError):
            flatten_namespaces({'tools':[{'type':'function','name':'robot__inspect'},
                {'type':'namespace','name':'robot','tools':[{'type':'function','name':'inspect'}]}]})

    def test_codex_assistant_history_gets_required_sdk_metadata_without_text_changes(self):
        original={'type':'message','id':'msg_history','role':'assistant',
                  'content':[{'type':'output_text','text':'Checking the observation.'}]}
        before=json.dumps(original)
        forwarded,_=flatten_namespaces({'input':[original]})
        self.assertEqual(forwarded['input'][0],{**original,'status':'completed',
            'content':[{'type':'output_text','text':'Checking the observation.','annotations':[]}]})
        self.assertEqual(json.dumps(original),before)

    async def test_student_stream_emits_executable_tool_calls_before_completion(self):
        self.gateway.terminal_output_events=True
        call={'type':'function_call','id':'fc1','call_id':'call1','name':'shell','arguments':'{"cmd":"pwd"}','status':'completed'}
        # This runtime streams literal tool syntax but parses it in the terminal response.
        self.wire=('data: '+json.dumps({'type':'response.output_text.delta','delta':'<tool_call>...broken partial...'})+'\n\n'
                   +'data: '+json.dumps({'type':'response.completed','response':{'status':'completed','output':[call]}})+'\n\n').encode()
        async with self.client.post(self.url+'/v1/responses',json={'model':'astra-test','stream':True,'input':[]},
                                    headers={'Authorization':'Bearer '+self.local_token}) as response:
            events=[json.loads(line[6:]) for line in (await response.text()).splitlines() if line.startswith('data: ')]
        self.assertEqual(events[-1]['type'],'response.completed')
        self.assertEqual([e['item'] for e in events if e['type']=='response.output_item.done'],[call])
        self.assertNotIn('response.output_text.delta',[e['type'] for e in events])
        self.assertEqual([e['sequence_number'] for e in events],list(range(len(events))))

    def test_cap_never_increases_a_lower_requested_budget(self):
        self.assertEqual(capped_request({'model': 'x', 'max_output_tokens': 512}, 'x', 16384)['max_output_tokens'], 512)
        self.assertEqual(capped_request({'model': 'x', 'max_output_tokens': 32768}, 'x', 16384)['max_output_tokens'], 16384)
        with self.assertRaises(ValueError):
            capped_request({'model': 'x', 'max_output_tokens': True}, 'x', 16384)

    def test_history_bound_retains_recent_images_and_all_tool_text(self):
        body={'model':'x','input':[{'role':'user','content':[{'type':'input_text','text':'task'}]},
              {'type':'function_call_output','call_id':'call_1','output':[
                 {'type':'input_image','image_url':f'image-{i}'} for i in range(53)]},
              {'role':'assistant','content':'do not alter this code'}]}
        before=json.dumps(body)
        result,receipt=bounded_images(body,40)
        self.assertEqual(json.dumps(body),before)
        output=result['input'][1]['output']
        self.assertEqual([x['image_url'] for x in output if x['type']=='input_image'],[f'image-{i}' for i in range(13,53)])
        self.assertTrue(all(x['type']=='input_text' for x in output[:13]))
        self.assertEqual(result['input'][0],body['input'][0]);self.assertEqual(result['input'][2],body['input'][2])
        self.assertEqual(receipt,{'original_images':53,'forwarded_images':40,'omitted_images':13,'max_images':40})

    def test_short_image_history_is_unchanged(self):
        body={'input':[{'type':'input_image','image_url':'one'}]}
        result,receipt=bounded_images(body,40)
        self.assertIs(result,body);self.assertEqual(receipt['omitted_images'],0)

"""Check the streamed image/action interface without screening out bad policies."""
import base64
import io
import json

from efe_action_loop import SYSTEM,parse_action


def check(spec,token,response):
    from PIL import Image
    contract={'action_types':{'ee_delta':{'dim':4,'layout':['dx','dy','dz','grip'],
        'lower':[-1.]*4,'upper':[1.]*4,'dtype':'float64'}},'max_repeat':25,
        'control_dt':.0125,'cameras':['synthetic'],'output_keys':['action_type','action','repeat','summary']}
    trials=[]
    for color,rgb,direction in [('green',(0,230,0),1),('blue',(0,0,230),-1)]:
        image=Image.new('RGB',(96,96),rgb);buffer=io.BytesIO();image.save(buffer,format='PNG')
        visible={'mode':'movement','action_contract':contract,'observation':{'synthetic':True},
            'recent_actions_and_feedback':[],
            'objective':'Synthetic interface check; no robot is connected. If the image is green, request dx=0.25; if blue, dx=-0.25. Set dy=dz=grip=0 and repeat=1. Name the observed color in summary.'}
        body={'model':spec['request_model'],'instructions':SYSTEM,
            'input':[{'role':'user','content':[{'type':'input_text','text':json.dumps(visible)},
                {'type':'input_image','image_url':'data:image/png;base64,'+base64.b64encode(buffer.getvalue()).decode(),'detail':'auto'}]}],
            'stream':True,'store':False,'max_output_tokens':4096,'reasoning':{'effort':'high'}}
        result=response(spec['upstream'],body,token)
        # HTTP/stream/incomplete failures still raise in the shared response reader.
        # A wrong answer or invalid command is policy behavior, recorded and then
        # evaluated by the native tasks rather than used to select passing seeds.
        if result.get('status')!='completed':raise RuntimeError('Constrained interface probe did not complete')
        text=''.join(c.get('text','') for item in result.get('output',[]) if item.get('type')=='message'
            for c in item.get('content',[]) if c.get('type')=='output_text')
        trial={'expected_color':color,'public_output':text,'usage':result.get('usage'),
            'returned_model':result.get('model'),'valid_action':False,'semantic_correct':False}
        try:
            command=parse_action(text,contract=contract,allow_stop=False)
            trial.update(valid_action=True,command=command,
                semantic_correct=command['action']==[direction*.25,0,0,0] and command['repeat']==1
                    and color in command['summary'].lower())
        except (ValueError,TypeError) as error:trial['parse_error']=str(error)
        trials.append(trial)
    return {'status':'passed' if all(t['semantic_correct'] for t in trials) else 'semantic_failure',
        'transport_status':'passed','benchmark_execution_allowed':True,
        'semantics_policy':'record_and_evaluate','protocol':'constrained-image-action-preflight-v1',
        'trials':trials,'transport_scope':'Two completed streamed image-to-text Responses requests through the native gateway. Command and color errors are recorded, not used to select seeds. This is not a robot success score or proof of internal image grounding.'}

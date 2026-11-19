"""Describe the public robot interface without inventing simulator conventions."""
import hashlib
import json


VERSION = 'pear-public-interface-language-v1'
DEFAULT_PROMPT_BUDGET = 16384


def context_policy():
    return {'version':VERSION,'enabled':True,'same_for_both_arms':True,'per_observation':True,
        'format':'Natural-language guide alongside the unchanged native schema and sensor values.',
        'action_fields':['component_order','dimensions','bounds','units','frames','axis_directions',
                         'gripper_and_hand_semantics','action_duration'],
        'observation_fields':['delivered_sensor_names_and_order','RGB_resolution','image_axis_convention',
            'declared_camera_mount_motion_and_calibration','proprioception_layout','public_observation_template'],
        'metadata_rule':'Use public adapter declarations only; omitted conventions remain explicitly unspecified. Never infer world/base/camera axes from each other.',
        'provenance':['interface_context_version','interface_context_sha256'],
        'implementation_scope':'Shared EFE native PEAR tool. Additional YAM/reasoning adapters must supply their own verified public metadata through the same context contract.'}


def compact(value):
    return value if isinstance(value,str) else json.dumps(value,ensure_ascii=False,allow_nan=False)


def template(value,depth=0):
    """Show sensor field structure, never current values or private task labels."""
    if isinstance(value,dict):
        if depth>=3:return '<object fields: '+', '.join(value)+'>'
        return {key:template(item,depth+1)for key,item in value.items()}
    if isinstance(value,list):
        item=compact(template(value[0],depth+1)) if value and depth<3 else 'value'
        return f'<array length={len(value)}; item={item}>'
    if value is None:return '<optional value; currently absent>'
    if isinstance(value,bool):return '<boolean>'
    if isinstance(value,(int,float)):return '<number>'
    return '<text>'


def interface_context(visible,*,image_sizes=None):
    """The same deterministic, versioned language context serves both PEAR arms.

    Coordinate statements are copied from public native declarations. In
    particular, a world-frame convention never becomes a base-frame convention,
    and advertised depth support never becomes an attached depth observation.
    """
    spec=visible.get('spec',{});contract=visible['action_contract']
    observation=visible.get('observation',{})
    if not isinstance(spec,dict)or not isinstance(observation,dict):
        raise ValueError('Expected public robot specification and observation objects')
    cameras=contract.get('cameras',[])
    if not isinstance(cameras,list)or not cameras or len(set(cameras))!=len(cameras):
        raise ValueError('A unique ordered list of delivered RGB sensors is required')
    if image_sizes is not None and len(image_sizes)!=len(cameras):
        raise ValueError('Delivered RGB images do not match the declared sensor order')
    lines=['Action and observation guide',
        'Use these declared robot coordinates and sensor fields when reasoning about your next action.',
        'Action space:',
        'Return reasoning followed by the required JSON action. Its action_type selects the native vector; '
        'action is the numeric vector in the exact component order below; repeat is an integer from 1 to '
        f"{contract['max_repeat']}; summary explains the chosen action.",
        f"One control step lasts {contract['control_dt']} seconds. The controller applies the same vector "
        'once per step for repeat steps; requested duration is repeat times the control-step duration.',
        'Declared world/robot coordinate conventions: '+compact(spec.get('frames',
            spec.get('world','Not declared.')if contract.get('control_interface')=='pear-eef-native-v1'else'Not declared.')),
        'For Cartesian deltas, positive values follow the declared positive axis and negative values '
        'go in the opposite direction. Absolute position or joint targets are targets, not deltas. '
        'Image left/right/up/down are not automatically world or robot-base directions. '
        'Use native frame and rotation conventions; any convention omitted below is unspecified.']
    for name,entry in contract['action_types'].items():
        layout=entry['layout']
        if len(layout)!=entry['dim']or any(len(entry[key])!=entry['dim']for key in ['lower','upper']):
            raise ValueError('Action layout or bounds do not match its dimension')
        lines.extend([f"- {name}: {entry['dim']} components, in this exact order: "+compact(layout),
            '  Component-wise lower bounds: '+compact(entry['lower'])+'; upper bounds: '+compact(entry['upper'])])
        # Preserve asymmetric arm/finger order, mixed frames, normalized scales,
        # quaternion order and gripper sign descriptions exactly as published.
        fields=['frame','frames','axis_directions','units','dtype','kinds','parts',
                'semantics','grip','gripper','rotation_convention','quaternion_order','hybrid_mode']
        if contract.get('control_interface')=='pear-eef-native-v1':
            fields+=['tool_frame','note','quaternion_groups']
        for key in fields:
            if key in entry:lines.append('  '+key+': '+compact(entry[key]))
        if not any(key in entry for key in ['frame','frames']):
            lines.append('  Action-specific frame: not separately declared; consult the native semantics above and global conventions without assuming they coincide.')
        if 'units'not in entry:lines.append('  Units: not declared.')
    lines.extend(['Observation space:',
        'The public metadata JSON contains objective (task instruction), spec (robot/sensor declarations), '
        'action_contract (executable action schema), observation (current public sensor values), and '
        'recent_actions_and_feedback (previous commands and execution feedback).',
        'RGB images follow their named "Current camera: <name>" labels in this order: '+compact(cameras)+'.',
        'Attached images are RGB only. Depth is not attached, even when a native camera advertises depth support.',
        'For these image arrays, row 0 is at the top and column 0 is at the left; increasing columns '
        'moves right and increasing rows moves down. This image indexing does not define action axes.'])
    declared=spec.get('cameras',{})
    for index,name in enumerate(cameras):
        camera=declared.get(name,{})
        line='- RGB sensor '+name
        if image_sizes is not None:
            width,height=image_sizes[index];line+=f': delivered size {width} pixels wide by {height} high'
        if 'size'in camera:line+='; native size declaration '+compact(camera['size'])
        if 'moving'in camera:line+='; moving='+compact(camera['moving'])
        lines.append(line+'.')
        for key in ['frame','mount','pose','intrinsics','extrinsics','update_hz','hz','fps']:
            if key in camera:lines.append('  '+key+': '+compact(camera[key]))
        keys=list(observation.get('cameras',{}).get(name,{}))
        if keys:lines.append('  Public calibration/sensor metadata is at observation.cameras['+json.dumps(name)+'], with fields '+compact(keys)+'.')
    if spec.get('camera_calibration'):
        lines.append('Camera calibration convention: '+compact(spec['camera_calibration']))
    proprio=observation.get('proprio',{})
    if isinstance(proprio,dict)and ('layout'in proprio or 'now'in proprio):
        layout=proprio.get('layout',spec.get('proprio_layout'))
        now=proprio.get('now')
        if layout is not None and (not isinstance(layout,list)or isinstance(now,list)and len(now)!=len(layout)):
            raise ValueError('Public proprioception values do not match their declared layout')
        if now is not None and not isinstance(now,list):
            raise ValueError('Public proprioception now must be a vector when present')
        if layout is not None:
            lines.append('Proprioception: when present, observation.proprio.now[i] is the current value of observation.proprio.layout[i]. Exact field order: '+compact(layout)+'.')
        else:lines.append('Proprioception is present but its component names are not declared.')
        if now is None:lines.append('No current proprio.now sensor vector is supplied in this observation.')
        for key in ['units','frame','frames','update_hz','hz']:
            if key in proprio:lines.append('Proprioception '+key+': '+compact(proprio[key]))
    elif spec.get('proprio_layout'):
        lines.append('Native proprioception layout: '+compact(spec['proprio_layout'])+'. No current proprio.now vector is present in this observation.')
    for name,sensor in spec.get('sensors',{}).items():
        if name in observation and isinstance(sensor,dict):
            details={key:sensor[key]for key in ['type','layout','shape','units','frame','update_hz','hz']if key in sensor}
            lines.append('Additional delivered sensor observation.'+name+': '+compact(details))
    lines.append('Current public observation template (placeholders show field types and array layout, not example sensor values): '+compact(template(observation)))
    text='\n'.join(lines)
    return {'version':VERSION,'text':text,'sha256':hashlib.sha256(text.encode()).hexdigest()}

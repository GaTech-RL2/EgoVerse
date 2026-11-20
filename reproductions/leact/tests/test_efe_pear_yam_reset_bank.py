from reproductions.leact.cluster.efe_pear_yam_reset_bank import independent_spec,pose_fingerprint


def test_training_scaffold_never_uses_saved_evaluation_scene_poses_or_appearance():
    authored={'cup':{'pos':[.5,0,.75],'quat_xyzw':[0,0,0,1]}}
    reference={'schema_version':3,'task':'place','translation_only_actors':[],
        'base_object_init_poses':authored,'placement_geometry':{},'initial_constraints':[],
        'inventory_kind':'canonical_reference',
        'scenes':[{'object_init_poses':{'cup':'HELD_OUT_POSE'},'appearance':'HELD_OUT_COLOUR',
            'distribution':{'groups':[{'actors':['cup'],'xy_m':.025}],'yaw_degrees':0}}],
        'entries':[{'id':'canonical-001'}]}
    result=independent_spec(reference,authored,123)
    assert result['scenes'][0]['object_init_poses']==authored
    assert 'HELD_OUT' not in str(result)and 'canonical-001'not in str(result)
    assert 'inventory_kind'not in result and 'entries'not in result


def test_physical_fingerprint_ignores_dict_order_and_machine_precision_drift():
    a={'cup':{'pos':[.5,0,.75],'quat_xyzw':[0,0,0,1]}}
    b={'cup':{'quat_xyzw':[0.,0.,0.,1.],'pos':[.5+1e-10,0.,.75]}}
    assert pose_fingerprint(a)==pose_fingerprint(b)
    b['cup']['pos'][0]+=.001
    assert pose_fingerprint(a)!=pose_fingerprint(b)

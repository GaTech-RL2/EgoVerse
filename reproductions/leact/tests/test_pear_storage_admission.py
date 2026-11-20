from reproductions.leact.cluster.monitor_efe_pear_l40s01 import verified_readiness


def inputs():
    return ({'passed':True,'transport_roundtrip_passed':True,'transport_sha256':'current',
        'assets_root':'/private/assets','executed_control_steps':1,'cameras':[{'std':20}],
        'native_termination_terms':{'success':False},'model_side_filesystem':'AMLFS07',
        'simulator_side_filesystem':'LFS'},
        {'passed':True,'assets_root':'/private/assets'}, {'passed':True},
        {name:{'passed':True}for name in ['robodojo','robolab','yamlab']})


def test_pool_migration_requires_the_native_round_trip_and_all_three_renderer_smokes():
    probe,assets,sources,rtx=inputs()
    assert verified_readiness(probe,assets,sources,rtx,'current')['passed']
    del rtx['yamlab']
    assert verified_readiness(probe,assets,sources,rtx,'current')is None


def test_stale_transport_or_unverified_inputs_cannot_open_primary_admission():
    probe,assets,sources,rtx=inputs()
    assert verified_readiness(probe,assets,sources,rtx,'changed')is None
    sources['passed']=False
    assert verified_readiness(probe,assets,sources,rtx,'current')is None

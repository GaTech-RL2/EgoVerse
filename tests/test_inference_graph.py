from egomimic.eval.inference_graph import (
    ActionCacheState,
    InferenceGraph,
    KeyedNode,
    Subgraph,
)


def _node(fn, inputs, outputs):
    return KeyedNode(fn, **{"in": inputs, "out": outputs})


def test_cache_hit_exits_before_preprocess_and_model():
    calls = []
    graph = InferenceGraph(
        check_cache=_node(lambda obs: calls.append("check") or 7,
                          {"obs": "obs"}, {"action": "policy.action"}),
        inference_preprocess=_node(
            lambda obs: calls.append("pre") or obs,
            {"obs": "obs"}, {"request": "request"}),
        model=_node(lambda request: calls.append("model") or request,
                    {"request": "request"}, {"plan": "plan"}),
        update_cache=_node(lambda plan: calls.append("update") or plan,
                           {"plan": "plan"},
                           {"action": "policy.action"}),
    )
    assert graph(obs=3) == 7
    assert calls == ["check"]


def test_cache_miss_runs_preprocess_model_update_in_order():
    calls = []
    graph = InferenceGraph(
        check_cache=_node(lambda obs: calls.append("check"),
                          {"obs": "rollout.obs"},
                          {"action": "policy.action"}),
        inference_preprocess=_node(
            lambda obs: calls.append("pre") or obs + 1,
            {"obs": "rollout.obs"}, {"request": "model.request"}),
        model=_node(lambda request: calls.append("model") or request * 2,
                    {"request": "model.request"}, {"plan": "model.plan"}),
        update_cache=_node(
            lambda plan: calls.append("update") or plan + 3,
            {"plan": "model.plan"}, {"action": "policy.action"}),
    )
    assert graph(**{"rollout.obs": 4}) == 13
    assert calls == ["check", "pre", "model", "update"]


def test_subgraph_has_single_entry_and_endpoint():
    subgraph = Subgraph(
        [
            _node(lambda x: x + 2, {"x": "x"}, {"y": "y"}),
            _node(lambda y: y * 3, {"y": "y"}, {"result": "result"}),
        ],
        **{"in": {"x": "outer.x"}, "out": {"result": "outer.y"}},
    )
    context = {"outer.x": 5}
    subgraph(context)
    assert context["outer.y"] == 21


def test_action_cache_is_instance_scoped_and_resettable():
    left, right = ActionCacheState(), ActionCacheState()
    left.replace([1, 2])
    right.replace([9])
    assert left.pop() == 1
    assert right.pop() == 9
    left.reset()
    assert not left and not right

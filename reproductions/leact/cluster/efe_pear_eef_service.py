"""Bound existing native pose controllers by a truthful control-step budget.

Installed only in an explicitly EEF-configured simulator process. It intercepts
the public RCB service at registration, retaining the original controller,
step function, termination checks, recording and native grader.
"""
import functools


def bound_service(service):
    original = service.submit
    namespace = original.__globals__
    if 'run_ee_pose' not in namespace or 'ctrl_step' not in namespace:
        raise ValueError('Native pose service has no bounded-step integration point')
    step = namespace['ctrl_step']

    @functools.wraps(original)
    def submit(meta, actions):
        if meta.get('type') != 'ee_pose':
            return original(meta, actions)
        # One identical target row per requested step is the wire convention.
        # Never turn repeat rows into repeat unbounded native waypoint plans.
        import numpy as np
        budget = len(actions)
        if not 1 <= budget <= 25 or not np.array_equal(actions, np.repeat(actions[:1], budget, axis=0)):
            raise ValueError('EEF submits need 1..25 identical rows declaring the native step budget')
        count = 0

        def bounded_step(*args, **kwargs):
            nonlocal count
            if count >= budget:
                return True
            stopped = step(*args, **kwargs)
            count += 1
            return bool(stopped or count >= budget)

        namespace['ctrl_step'] = bounded_step
        try:
            output, parts = original(meta, actions[:1])
        finally:
            namespace['ctrl_step'] = step
        actual = output.get('control_steps_chunk')
        if type(actual) is not int or actual != count:
            raise ValueError('EEF controller and native control-step accounting disagree')
        output['eef_execution'] = {'requested_control_steps': budget,
            'executed_control_steps': actual, 'step_budget_exhausted': count == budget,
            'native_plan': output.get('plan', [])}
        return output, parts

    service.submit = submit
    return service


def install(limited):
    original = limited.serve
    def serve(service, *args, **kwargs):
        bound_service(service)
        return original(service, *args, **kwargs)
    limited.serve = serve

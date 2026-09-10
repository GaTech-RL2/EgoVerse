"""Optional broad-phase shortcuts for headless contact-tool candidate search.

These shortcuts never run in the acceptance replay. Every saved episode must
reproduce its complete state trajectory in the untouched simulator first.
"""

from __future__ import annotations

from types import MethodType

import pymunk


def enable_fast_queries(env):
    assert env._skip_obs_render and env.pusher_shape in (
        "scoop",
        "triangle",
        "flipper",
        "spring",
    )
    pusher_shapes = set(env.agent.physics_shapes(env))
    static_bounds = [
        shape.cache_bb()
        for shape in env._space.shapes
        if shape.body.body_type == pymunk.Body.STATIC
    ]
    radii = {}
    bounded = [(env._object_body, env._object_shapes)]
    if env.pusher_shape in ("scoop", "triangle"):
        assert all(shape.body is env._pusher_body for shape in pusher_shapes)
        bounded.append((env._pusher_body, pusher_shapes))
    for body, shapes in bounded:
        radius = 0.0
        for shape in shapes:
            if isinstance(shape, pymunk.Segment):
                points = [shape.a, shape.b]
            elif isinstance(shape, pymunk.Poly):
                points = shape.get_vertices()
            else:
                points = [shape.offset]
            radius = max(
                radius,
                max(
                    (shape.body.local_to_world(p) - body.position).length
                    for p in points
                )
                + shape.radius,
            )
        radii[body] = radius + 1e-6

    def static_clear(body, shapes):
        radius = radii.get(body)
        if radius is not None:
            x, y = body.position
            boxes = [pymunk.BB(x - radius, y - radius, x + radius, y + radius)]
        else:
            boxes = [shape.cache_bb() for shape in shapes]
        return all(not box.intersects(bb) for box in boxes for bb in static_bounds)

    original_static = env._shapes_static_penetration_depth
    original_resolve = env._clamp_pusher_to_static

    def static_depth(self, body, shapes):
        if static_clear(body, shapes):
            return 0.0
        return original_static(body, shapes)

    def resolve(self):
        if static_clear(self._pusher_body, self.agent.physics_shapes(self)):
            return
        original_resolve()

    def pusher_depth(self):
        self._space.reindex_shapes_for_body(self._object_body)
        self._space.reindex_shapes_for_body(self._pusher_body)
        depth = 0.0
        # The T has two convex shapes; the scoop has many arc segments.
        for shape in self._object_shapes:
            for query in self._space.shape_query(shape):
                if query.shape in pusher_shapes:
                    for point in query.contact_point_set.points:
                        depth = max(depth, abs(float(point.distance)))
        return depth

    env._shapes_static_penetration_depth = MethodType(static_depth, env)
    env._clamp_pusher_to_static = MethodType(resolve, env)
    if env.pusher_shape == "scoop":
        env._pusher_object_penetration_depth = MethodType(pusher_depth, env)

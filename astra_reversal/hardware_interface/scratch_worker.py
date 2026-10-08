"""Copied alone into the scratch jail; has no evaluator imports or credentials."""

import json
import sys


def rpc(tool, arguments):
    sys.stdout.write(
        json.dumps({"tool": tool, "arguments": arguments}, allow_nan=False) + "\n"
    )
    sys.stdout.flush()
    return json.loads(sys.stdin.readline())


def observe(keys):
    return rpc("observe", {"keys": keys})


def step(action, repeat_steps=1, observation_step=None):
    return rpc(
        "step",
        {
            "action": action,
            "repeat_steps": repeat_steps,
            "observation_step": observation_step,
        },
    )


def read(channel, max_age_ms=1000):
    return rpc("read", {"channel": channel, "max_age_ms": max_age_ms})


def act(envelope):
    return rpc("act", {"envelope": envelope})


def main():
    payload = json.loads(sys.stdin.readline())
    context = {"__builtins__": __builtins__, "result": None}
    if payload["condition"] == "F":
        context.update(read=read, act=act)
    else:
        context.update(observe=observe, step=step)
    try:
        exec(compile(payload["code"], "scratch.py", "exec"), context)
        sys.stdout.write(
            json.dumps({"result": context.get("result")}, allow_nan=False) + "\n"
        )
    except BaseException as error:
        sys.stdout.write(json.dumps({"error": type(error).__name__}) + "\n")
    sys.stdout.flush()


if __name__ == "__main__":
    main()

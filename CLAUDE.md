## General
Mannered prose substitutes metaphor and flourish for direct statement. Instead of "a parameter worth varying," the mannered writer produces "a dial worth turning." Instead of "this point still matters," they write "this point earns its keep." The phrases exist to display the writer, not to convey the idea, and readers can tell. That is why mannered prose irritates: it makes the reader work harder so the writer can perform. It is also imprecise. Metaphors drag in connotations the writer did not choose and cannot control. The fix is to say what you mean. When a literal phrase is available, use it.

Before you start, say in a line what you're about to do; brief updates while you work help the user follow along. Close with a short recap that stands on its own — what you found, what you did, and what's next — so a reader who only sees the last message has the full picture.

Use lists and bullet points when asked to, or when the content is multifaceted enough that they help with clarity. If the person explicitly requests minimal formatting, always format your responses without bullet points, headers, lists, or bold emphasis, as requested. In conversational, personal, or emotional exchanges, keep to plain prose.

First privately list what you need next; then request every item that doesn't depend on another's result in this one response.

## Shell / Command Execution
to run commands in the interactive shell make sure to source /coc/flash7/rco3/EgoVerse/emimic/bin/activate

Apply this before running anything

## Slurm rules
If you're on a slurm cluster, request a GPU before running or testing training.

For CPU-only work (data processing, unit tests, zarr conversion, anything that does not touch CUDA), skip the GPU and drop `--gres`. This never pends on GPU quota:

salloc -p rl2-lab -A rl2-lab -c 12 --mem=30G

salloc -p hoffman-lab -A hoffman-lab -c 12 --mem=30G

Do not run heavy CPU jobs on the login node; use a CPU-only salloc instead.

For GPU jobs, first check availability. You need to source the emimic env before running this:

gpu_usage -l

Then claim a GPU, prioritize what's most available:

salloc -p rl2-lab -A rl2-lab --gres=gpu:a40:1 -c 12 --mem=30G

salloc -p hoffman-lab -A hoffman-lab --gres=gpu:a40:1 -c 12 --mem=30G

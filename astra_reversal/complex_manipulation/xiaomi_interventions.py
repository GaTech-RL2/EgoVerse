"""Reversible hooks around the released Xiaomi processor, VLM and flow field.

Native/neutral execution calls the original model.forward unchanged. Text and
vision edits never write robot state or action normalization parameters.
"""

from contextlib import contextmanager

import numpy as np
import torch
from PIL import Image

CAMERAS = {"left": "video.robot0_agentview_left", "right": "video.robot0_agentview_right",
           "wrist": "video.robot0_eye_in_hand"}


def perturb_images(images, settings, previous=None):
    output = {k: np.array(v, copy=True) for k, v in images.items()}
    key = CAMERAS[settings["camera"]]
    x0, y0, x1, y1 = settings["roi"]
    for i, image in enumerate(output[key]):
        height, width = image.shape[:2]
        box = (int(x0*width), int(y0*height), max(int(x1*width), int(x0*width)+1),
               max(int(y1*height), int(y0*height)+1))
        operation = settings["image_operation"]
        if operation == "occlude":
            image[box[1]:box[3], box[0]:box[2]] = 128
        elif operation == "crop":
            output[key][i] = np.asarray(Image.fromarray(image).crop(box).resize(
                (width, height), Image.Resampling.BILINEAR))
        elif operation == "previous":
            # At the first review the source is the first available live frame.
            output[key][i] = previous[key] if previous is not None else images[key][0]
        else:
            raise ValueError("Unknown image source")
    return output


def blend_images(original, source, alpha):
    if alpha == 0:
        return original
    return {k: np.rint((1-alpha)*v.astype(np.float32)+alpha*source[k].astype(np.float32)).clip(0,255).astype(np.uint8)
            for k, v in original.items()}


def text_slots(tokenizer, ids, instruction):
    values = ids[0].detach().cpu().tolist()
    decoded = tokenizer.decode(values, skip_special_tokens=False)
    marker = "Generate robot actions for the task:\n"
    start = decoded.rindex(marker) + len(marker)
    end = start + len(instruction)
    if decoded[start:end] != instruction or not decoded[end:].startswith(" /no_cot"):
        raise ValueError("Instruction boundary differs from native chat template")
    needle = tokenizer.encode(instruction, add_special_tokens=False)
    matches = [i for i in range(len(values)-len(needle)+1) if values[i:i+len(needle)] == needle
               and tokenizer.decode(values[:i], skip_special_tokens=False).endswith(marker)
               and tokenizer.decode(values[i+len(needle):], skip_special_tokens=False).startswith(" /no_cot")]
    if len(matches) != 1:
        raise ValueError("Cannot verify an instruction-only native token span")
    slots = list(range(matches[0], matches[0]+len(needle)))
    if not slots:
        raise ValueError("No verified instruction tokens")
    return torch.tensor(slots, dtype=torch.long, device=ids.device)


def aligned(source, length):
    result = source.new_zeros((length, source.shape[-1]))
    count = min(length, len(source))
    result[:count] = source[:count]
    return result


def add_at_slots(hidden, slots, delta, alpha):
    result = hidden.clone()
    result[:, slots] = result[:, slots] + alpha * delta
    return result


@contextmanager
def registered(handle):
    try:
        yield
    finally:
        handle.remove()


def flow_integrate(velocity, x, *, reverse=False, steps=5):
    dt = 1.0/steps
    for index in range(steps):
        value = 1-index*dt if reverse else index*dt
        t = torch.full((x.shape[0], 1, 1), value, device=x.device, dtype=x.dtype)
        v = velocity(x, t)
        x = x-v*dt if reverse else x+v*dt
    return x


class NativeEngine:
    def __init__(self, model, processor, message_builder):
        self.model, self.processor, self.messages = model, processor, message_builder
        self.language = model.vlm.model.language_model
        self.last_audit = {}

    def inputs(self, states, images, instruction):
        state = np.zeros((1, len(states), 60), np.float32)
        state[0, :, :14] = states
        inputs = self.processor.apply_chat_template(self.messages(images, instruction), tokenize=True,
            return_dict=True, return_tensors="pt", do_resize=False, state=state, robot_type="robocasa365")
        result = {k: (v.to(device=self.model.device, dtype=self.model.dtype) if v.is_floating_point()
                    else v.to(device=self.model.device)) if isinstance(v, torch.Tensor) else v
                for k, v in dict(inputs).items()}
        result["task_id"] = "robocasa365"
        return result

    def _vlm(self, data):
        return self.model.vlm(**{k:v for k,v in data.items() if k not in ("state", "action_mask")}, use_cache=True)

    def _capture(self, data, slots, layer):
        captured = []
        if layer is None:
            def read(module, args, kwargs):
                captured.append(kwargs["inputs_embeds"][0, slots].detach().clone())
            handle = self.language.register_forward_pre_hook(read, with_kwargs=True)
        else:
            def read(module, args, kwargs):
                captured.append(args[0][0, slots].detach().clone())
            handle = self.language.layers[layer].register_forward_pre_hook(read, with_kwargs=True)
        with registered(handle):
            self._vlm(data)
        if len(captured) != 1:
            raise ValueError("Unexpected native VLM hook count")
        return captured[0]

    def _representations(self, data, source, instruction, subgoal, settings):
        method, alpha = settings["method"], settings["alpha"]
        text = method in ("tei", "tli")
        if text:
            target_slots = text_slots(self.processor.tokenizer, data["input_ids"], instruction)
            source_slots = text_slots(self.processor.tokenizer, source["input_ids"], subgoal)
        else:
            if not torch.equal(data["input_ids"], source["input_ids"]):
                raise ValueError("Vision intervention changed token layout")
            vision_id = self.model.config.vlm_config.video_token_id
            target_slots = torch.nonzero(data["input_ids"][0] == vision_id, as_tuple=False).flatten()
            source_slots = target_slots
            if not len(target_slots):
                raise ValueError("No native video tokens")
        layer = settings["layer"] if method in ("tli", "vli") else None
        if method == "tei":
            embedding = self.model.vlm.get_input_embeddings()
            original = embedding(data["input_ids"])[0, target_slots].detach()
            donor = embedding(source["input_ids"])[0, source_slots].detach()
        else:
            original = self._capture(data, target_slots, layer)
            donor = self._capture(source, source_slots, layer)
        delta = aligned(donor, len(target_slots))-original
        edits = []
        if method == "tei":
            def edit(module, args, output):
                edits.append(True)
                return add_at_slots(output, target_slots, delta, alpha)
            handle = embedding.register_forward_hook(edit)
        elif layer is None:
            def edit(module, args, kwargs):
                edits.append(True)
                kwargs = dict(kwargs)
                kwargs["inputs_embeds"] = add_at_slots(kwargs["inputs_embeds"], target_slots, delta, alpha)
                return args, kwargs
            handle = self.language.register_forward_pre_hook(edit, with_kwargs=True)
        else:
            def edit(module, args, kwargs):
                edits.append(True)
                return (add_at_slots(args[0], target_slots, delta, alpha), *args[1:]), kwargs
            handle = self.language.layers[layer].register_forward_pre_hook(edit, with_kwargs=True)
        with registered(handle):
            output = self.model(**data).actions
        if len(edits) != 1:
            raise ValueError("Unexpected edit count")
        self.last_audit.update(edited_slots=len(target_slots), layer=layer,
                               representation_delta_rms=float(delta.float().square().mean().sqrt()))
        return output

    def _frs(self, data, settings, seed):
        original = self.model.dit_forward
        calls, captured = [], {}
        def record(**kwargs):
            if not calls:
                captured["initial_noise"] = kwargs["noisy_action"].detach().clone()
            calls.append(float(kwargs["t"][0,0,0]))
            captured["context"] = {k:v for k,v in kwargs.items() if k not in ("noisy_action", "t")}
            return original(**kwargs)
        self.model.dit_forward = record
        try:
            native = self.model(**data).actions
        finally:
            self.model.dit_forward = original
        if len(calls) != 5:
            raise ValueError("FRS requires the registered five-step native field")
        config = self.processor.action_config["robocasa365"]
        std, mean = config["std"].to(native.device), config["mean"].to(native.device)
        decoded = native.float()*std+mean
        target = decoded.clone()
        bias = torch.tensor(settings["translation_bias"],device=native.device,dtype=torch.float32)
        if torch.any(bias != 0):
            target[:, :16, :3] = (target[:, :16, :3]+bias).clamp(-1,1)
        if settings["gripper_target"] != -1:
            target[:, :16, 6] = settings["gripper_target"]
        edited = (native.float()+(target-decoded)/std.clamp_min(1e-5)*data["action_mask"]).to(native.dtype)
        velocity = lambda x,t: original(noisy_action=x,t=t,**captured["context"])
        noise = flow_integrate(velocity, edited, reverse=True)
        active = data["action_mask"].bool()
        recovered_rms = float(noise[active].float().square().mean().sqrt())
        sigma = settings["noise_sigma"]
        if sigma:
            generator = torch.Generator(device=noise.device).manual_seed(seed+90000001)
            jitter = torch.randn(noise.shape, device=noise.device, dtype=noise.dtype, generator=generator)
            noise = noise+sigma*jitter*data["action_mask"]
        result = flow_integrate(velocity, noise)
        self.last_audit.update(flow_evaluations=15, reverse_steps=5, forward_steps=5,
            recovered_noise_rms=recovered_rms,
            target_delta_rms=float((edited-native)[active].float().square().mean().sqrt()),
            returned_action_delta_rms=float((result-native)[active].float().square().mean().sqrt()))
        return result

    @torch.no_grad()
    def normalized(self, data, *, settings=None, source=None, instruction="", seed=7):
        torch.manual_seed(seed)
        self.last_audit = {"method": settings["method"] if settings else "native", "noise_seed": seed}
        if settings is None or settings["method"] == "native" or (
                settings["method"] != "phase_prompt" and settings["alpha"] == 0):
            result = self.model(**data).actions
        elif settings["method"] in ("tei", "tli", "vei", "vli"):
            result = self._representations(data, source, instruction, settings["subgoal"], settings)
        elif settings["method"] == "frs":
            result = self._frs(data, settings, seed)
        elif settings["method"] in ("image", "phase_prompt"):
            result = self.model(**source).actions
        else:
            raise ValueError("Unknown native intervention")
        if not torch.isfinite(result).all():
            raise ValueError("Nonfinite policy prediction")
        return result

    def infer(self, states, images, instruction, *, settings=None, previous=None, seed=7):
        data = self.inputs(states, images, instruction)
        source = None
        if settings and settings["method"] in ("phase_prompt", "tei", "tli"):
            prompt = (instruction+" Current phase: "+settings["subgoal"]
                      if settings["method"] == "phase_prompt" else settings["subgoal"])
            source = self.inputs(states, images, prompt)
        elif settings and settings["method"] in ("image", "vei", "vli"):
            transformed = perturb_images(images, settings, previous)
            if settings["method"] == "image":
                transformed = blend_images(images, transformed, settings["alpha"])
            source = self.inputs(states, transformed, instruction)
        protected = {k: data[k].clone() for k in ("state", "action_mask")}
        output = self.normalized(data, settings=settings, source=source, instruction=instruction, seed=seed)
        if any(not torch.equal(data[k], v) for k,v in protected.items()):
            raise ValueError("Intervention changed protected native inputs")
        # Match the released socket client: transfer normalized predictions to
        # CPU before applying its native float32 action normalization.
        actions = self.processor.decode_action(output.cpu(), robot_type="robocasa365")[0, :, :12].float().numpy()
        return np.asarray(actions, dtype=np.float32)

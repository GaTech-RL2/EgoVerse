# EgoVerse Contribution Guidelines

The EgoVerse project aims to capture a **diverse set of economically useful work performed by a proficient demonstrator**. Below are some heuristics to determine if a given dataset meets this bar, but should not be treated as an exhaustive list.

# Diversity

In aggregate, the EgoVerse project aims for the following mix along 2 axes: 

| **Dimension** | **Target Mix** | **Notes/Examples** |
| --- | --- | --- |
| Task Type | ~15% mobile manipulation · ~85% manipulation | **Navigation:** walking, running, biking<br/>**Mobile Manipulation:** moving dishes from dining room to kitchen |
| Dexterity | ~60% doable with a parallel-jaw gripper · ~40% doable by either a parallel-jaw gripper or a dexterous hand | **Parallel Gripper:** pick and place, laundry folding<br/>**Dexterous Hand:** screwing on/off a bottle cap|

> **NOTE**: While annotations are optional, having episode level annotations for task type, dexterity, and environment increase the likelihood of acceptance


# Quality

We measure per-episode quality across operator, task, and technical proficiency. Some guidelines for each are enumerated below.

## Operator Proficiency

The operator completing the task should abide by the following behavior, regardless of the task, environment, or collection device:

- Hands must be visible in 90% of frames
- No aggressive head movements
- No unnecessarily aggressive hand movements
- No idle time

## Task Proficiency

The episode should depict a task as if it were done by an expert. This means the **task goal is completed without mistakes, wasted actions, idle time, or repetition.** Here are some examples of proficient and non-proficient executions of the X task:   

- Proficient:
- Not Proficient:

Additionally, the task must work towards completing a **visually obvious goal**. For example, if the task is “Mop stain from floor”, there should be an actual stain on the floor that the operator cleans.

## Technical Proficiency

The episode should not have any bugs or glitches. Some examples of this include:

- Dropped frames
- Damaged sensors
- Hand tracking errors
- Misaligned timestamps across sensors

This is highly variable depending on the specific hardware setup.
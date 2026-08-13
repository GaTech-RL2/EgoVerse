# EgoVerse Annotation Guide

Language annotations are **optional but strongly encouraged**. They are stored as a span-based structure: each annotation covers a contiguous range of frames.

Good quality annotations are both adequately granular across an episode and descriptive of each target subtask. Some guidelines for granularity and description quality are outlined below. These are not exhaustive.

# Granularity

Every meaningful hand-object interaction should be accounted for within the episode. This includes every pick, place, fold etc.

As a rule of thumb, all annotation spans should be between 1-10 seconds long. The hand-object rule and the 1-10s rule may disagree in certain cases. Here are some common ones and how to resolve them

| **Action** | **Preferred Span(s)** |
| --- | --- |
| An object is picked and placed, but the pick is very quick (<1s) | One span where the pick and place are described |
| The same repeated action over the length of the episode (chopping the same vegetable) | One continuous span. The annotation should mention that the action is being done repeatedly |
| The same “complex” action over the length of the episode (drying and stacking plates)  | Each “cycle” of the action should get its own span. For each of these cycles, use the heuristic from the first scenario |

# Description Quality

Annotation text should be **true and descriptive** of the action taking place during the span, and should use specific language for the verb and object. In particular, words like “adjust” or “manipulate” should not be used.

Additionally, they must be written in English and use the imperative or present-continuous form ("pick up the shirt", "folding the left sleeve", etc).

# Common Failure Modes

- A single span detailing a complex action with a compound sentence: “Pick up the cucumber, wash it, and place in the bowl”.
    - **FIX:** Each action (pick, wash, place) should get its own annotation span
- Plural/singular error: “Pick up bottle”, when two or more bottles were actually picked up
    - **FIX:** Make sure the annotation is **true** by updating to: “Pick up two bottles”, “Pick up the bottles”, etc
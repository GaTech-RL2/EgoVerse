# EgoVerse Annotation Rubric

- Language annotations are optional but strongly encouraged.
- An episode with no language annotations is **not evaluated** by this rubric
- When annotations are provided, every submitted span is evaluated.

## Passing Rule

An annotated episode passes when 90% of its spans pass ALL criteria in the required rubric:

```text
pass = (correct_spans / total_evaluated_spans) > 0.90
```

A span is correct only when all rubric criteria (`R1`-`R5`) pass. The non-failing criteria (`Q1`-`Q3`) are scored and reported separately. They do not affect the pass/fail result.

## Required Span-Level Rubric

| ID | Criterion | Pass | Fail |
| --- | --- | --- | --- |
| `R1` | Truth and action accuracy | The described action occurs during the span, and the verb, object, and object count match what is visible. | The text describes the wrong action or object, or uses an incorrect singular/plural count. |
| `R2` | Description specificity | The annotation uses a specific verb and names the relevant object. | The annotation is vague, omits the object, or uses nonspecific verbs such as "adjust" or "manipulate." |
| `R3` | Action granularity | The span describes one meaningful action, except for an allowed combined action described below. Sequential actions are split into separate spans. | A span combines multiple sequential actions in a compound description, such as picking up, washing, and placing an object. |
| `R4` | Span duration | The span is between 1 and 10 seconds long, or it follows one of the documented exceptions below. | The span is shorter than 1 second or longer than 10 seconds without a documented exception. |
| `R5` | Language and grammatical form | The annotation is written in English and uses either the imperative or present-continuous form. | The annotation is not in English or uses another grammatical form. |

### Granularity and Duration Exceptions

Use these rules when `R3` (Action granularity) and `R4` (Span duration) conflict:

| Scenario | Required treatment |
| --- | --- |
| A pick is very quick (less than 1 second) and is followed by a placement. | Use one span that describes both the pick and the placement. |
| The same simple action is repeated over a long interval, such as repeatedly chopping the same vegetable. | Use one continuous span, even if it is longer than 10 seconds, and state that the action is repeated. |
| The same complex action is repeated in cycles, such as drying and stacking plates. | Use a separate span for each cycle. Within each cycle, combine a sub-second pick with its corresponding placement when needed. |
| Several distinct actions occur in sequence, such as picking up, washing, and placing a cucumber. | Use a separate span for each action rather than one compound annotation. |

## Non-failing Rubric

These are other attributes that are nice to have and graded, but not included in the failure calculation 

| ID | Criterion | Pass | Fail |
| --- | --- | --- | --- |
| `Q1` | Start-boundary precision | The span begins when the described action begins. | The span begins noticeably before or after the action begins. |
| `Q2` | End-boundary precision | The span ends when the action finishes; when contact defines the action, it ends when contact is broken. | The span ends noticeably before or after the action finishes or contact is broken. |
| `Q3` | Boundary tightness | The span contains only the frames needed to cover the described action. | The span includes avoidable idle time or unrelated activity before or after the action. |
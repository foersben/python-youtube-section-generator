# Git Commit Message Instructions

Follow the Conventional Commits specification for clear, searchable commit history.

## Commit Message Format

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

## Types

- **feat**: A new feature
- **fix**: A bug fix
- **docs**: Documentation only changes
- **style**: Code style changes (formatting, missing semicolons, etc)
- **refactor**: Code change that neither fixes a bug nor adds a feature
- **perf**: Performance improvement
- **test**: Adding or modifying tests
- **build**: Changes to build system or dependencies
- **ci**: Changes to CI configuration files and scripts
- **chore**: Other changes that don't modify src or test files
- **revert**: Reverts a previous commit

## Scopes (Optional)

Use scopes to specify the area of change:

- **model**: Changes to simulation model, events, state management
- **view**: Changes to GUI, visualization, UI components
- **controller**: Changes to controllers
- **config**: Changes to configuration classes
- **events**: Changes to event system
- **network**: Changes to network state or graph utilities
- **energy**: Changes to energy management
- **tests**: Changes to test files
- **docs**: Changes to documentation
- **deps**: Dependency updates

## Description

- Use imperative mood: "add feature" not "added feature"
- Don't capitalize first letter
- No period at the end
- Keep under 72 characters
- Be specific and descriptive

## Body (Optional)

- Separate from description with blank line
- Wrap at 72 characters
- Explain **what** and **why**, not **how**
- Include motivation for the change
- Contrast with previous behavior

## Footer (Optional)

- Reference issues: `Fixes #123`, `Closes #456`
- Note breaking changes: `BREAKING CHANGE: description`
- Co-authors: `Co-authored-by: Name <email>`

## Examples

### Simple Feature
```
feat(model): add energy normalization fallback for ring rendering

Ensures energy rings are always visible even when normalization 
function is not provided by using energy/initial_energy ratio.
```

### Bug Fix
```
fix(view): restore missing energy rings and buffer indicators

Energy rings and buffer points were not being rendered because
_create_sensor_node was not calling the creation methods.

Added calls to _create_energy_ring and _create_buffer_point to
ensure all node indicators are properly displayed.

Fixes #42
```

### Documentation
```
docs(readme): add installation instructions for Poetry

Include step-by-step guide for installing dependencies using
Poetry and running the application.
```

### Refactoring
```
refactor(model): replace Any with precise type hints

- Use dict[str, object] for event data
- Replace callable with Callable[[float], float]
- Use built-in generics (list, dict, tuple)
- Add type guards where needed

Improves type safety and IDE autocomplete.
```

### Breaking Change
```
feat(events)!: change event data structure to typed dict

BREAKING CHANGE: Event handlers now receive dict[str, object]
instead of Any. Update custom handlers to use type annotations
and type: ignore comments for extraction.

Migration guide:
```python
# Old
def handle(self, time: int, data: Any) -> None:
    path = data['path']

# New
def handle(self, time: int, data: dict[str, object]) -> None:
    path: list[int] = data['path']  # type: ignore[assignment]
```
```

### Test Addition
```
test(events): add test for packet reception with full buffer

Verifies that packets are dropped when node buffer is full
and appropriate error logging occurs.
```

### Dependency Update
```
build(deps): update networkx to 3.2.1

Includes bug fixes for graph traversal and improved performance
for shortest path calculations.
```

### Multiple Changes (use body)
```
feat(view): enhance node visualization

- Add energy rings with green-to-red gradient
- Add buffer level indicators at node center
- Improve z-ordering for proper layering
- Use precise Callable types for normalizers

All sensor nodes now clearly show energy and buffer status
with color-coded visual indicators.
```

## Project-Specific Guidelines

### Module Changes
When changing specific modules, use appropriate scope:
- `model/events/*` → scope: `events`
- `model/config/*` → scope: `config`
- `view/components/*` → scope: `view`
- `controller/*` → scope: `controller`

### Type Hint Improvements
```
refactor(model): improve type hints in event handlers

Replace broad types with precise built-in generics and Callables
for better static analysis and IDE support.
```

### GUI Changes
```
feat(view): add tooltip support for network nodes

Displays node ID, energy level, buffer fill, and packet counts
when hovering over nodes in the network visualization.
```

### Performance Improvements
```
perf(model): optimize state snapshot compression

Reduce memory usage by 40% using delta compression for
consecutive simulation states.
```

## Anti-Patterns (What NOT to Do)

❌ **Too vague**
```
fix: fix bug
chore: update stuff
feat: improvements
```

❌ **Too technical (should be in body)**
```
fix: change line 42 in node_renderer.py to use plt.get_cmap
```

❌ **Multiple unrelated changes**
```
feat: add tooltips, fix energy rings, update docs, refactor tests
```
(Split into separate commits)

❌ **Wrong tense**
```
feat: added new feature
fix: fixing the bug
```

✅ **Good examples**
```
feat(view): add node hover tooltips
fix(model): prevent division by zero in buffer calculation
docs(api): document EventHandler base class
refactor(events): extract packet validation to helper method
test(network): add edge case tests for isolated nodes
```

## When to Commit

- Commit logical units of work
- Commit working code (tests should pass)
- Commit before and after refactoring
- Commit before switching context
- Don't commit commented-out code
- Don't commit debug print statements

## Branch Naming (if using)

```
feature/add-energy-visualization
fix/buffer-overflow-crash
docs/api-reference
refactor/type-hints
test/event-handlers
```

Use descriptive names with type prefix and kebab-case.


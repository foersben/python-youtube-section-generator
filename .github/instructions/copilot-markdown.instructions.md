---
applyTo:
  - "**/*.md"
exclude:
  - ".venv/**"
  - "node_modules/**"
---

# Markdown Documentation Instructions

## Documentation Framework
- Use **MkDocs** for documentation generation
- Follow project structure in `docs/` directory
- Link to code with proper relative paths

## Markdown Style Guide

### Headers
```markdown
# Top Level (H1) - Only one per file, matches file purpose

## Section Headers (H2) - Major sections

### Subsection Headers (H3) - Detailed topics

#### Minor Headers (H4) - Use sparingly
```

### Code Blocks
Always specify the language:
````markdown
```python
def example():
    """Google-style docstring."""
    return "value"
```

```bash
poetry install
poetry run pytest
```
````

### Links
```markdown
# Internal links (relative)
[Architecture Overview](../introduction/software_design/software_architecture.md)

# External links
[NetworkX Documentation](https://networkx.org/)

# Code references
See [`SimulationModel`](../src/model/simulation_model.py) for implementation.
```

### Lists
```markdown
# Unordered lists (use - not *)
- First item
- Second item
  - Nested item
  - Another nested item

# Ordered lists
1. First step
2. Second step
3. Third step
```

### Tables
```markdown
| Feature | Status | Notes |
|---------|--------|-------|
| Event System | ✅ Complete | Observer pattern |
| GUI | ✅ Complete | Tkinter + matplotlib |
| Replay | ✅ Complete | State snapshots |
```

### Admonitions (for MkDocs)
```markdown
!!! note
    This is a note with important information.

!!! warning
    This is a warning about potential issues.

!!! tip
    This is a helpful tip for users.

!!! example
    Code example or usage demonstration.
```

## Documentation Types

### API Documentation
```markdown
## `ClassName`

Brief description of the class.

### Attributes
- `attribute_name` (type): Description of attribute
- `another_attr` (type): Description

### Methods

#### `method_name(param: type) -> return_type`

Brief description of what the method does.

**Parameters:**
- `param` (type): Description of parameter

**Returns:**
- type: Description of return value

**Raises:**
- `ExceptionType`: When and why

**Example:**
\```python
obj = ClassName()
result = obj.method_name("value")
\```
```

### Tutorials
```markdown
# Tutorial: Creating a Custom Event Handler

This tutorial shows you how to create a custom event handler for the simulator.

## Prerequisites
- Basic Python knowledge
- Understanding of the Observer pattern
- Familiarity with the event system

## Step 1: Create Handler Class

Create a new file in `src/model/events/event_handlers/`:

\```python
"""Custom packet handler for special routing logic."""

from typing import override
from model.events.event_handler import EventHandler

class CustomPacketHandler(EventHandler):
    """Handles packets with custom routing logic.

    This handler implements special routing rules for...
    """

    @override
    def handle(self, time: int, data: dict[str, object]) -> None:
        """Process packet with custom logic."""
        # Implementation
        pass
\```

## Step 2: Register Handler

Register your handler in `SimulationModel._register_event_handlers()`:

\```python
self.event_manager.register_handler(
    Event.CUSTOM_EVENT,
    CustomPacketHandler(...)
)
\```

## Testing

Create tests in `tests/test_custom_handler.py`:

\```python
def test_custom_handler_processes_packet():
    """Test custom handler processes packets correctly."""
    # Test implementation
    pass
\```
```

### Feature Documentation
```markdown
# Energy Management

The simulator tracks energy consumption at each node to model realistic sensor network behavior.

## Overview

Each sensor node has a finite energy budget that depletes as it:
- Transmits packets
- Receives packets
- Generates packets

## Configuration

Energy parameters are configured in `EnergyConfig`:

\```python
energy_config = EnergyConfig(
    e_tx=0.1,        # Energy per transmission
    e_rx=0.05,       # Energy per reception
    e_packet_gen=0.02,  # Energy per packet generation
    base_tx_cost=0.01,  # Base transmission cost
    distance_scale=10   # Distance scaling factor
)
\```

## Energy Calculation

Transmission energy is calculated as:

```
E_tx = base_tx_cost + (distance / distance_scale) * e_tx
```

## Monitoring Energy Levels

Energy levels are visualized in the GUI:
- **Green ring**: Full energy
- **Yellow ring**: Medium energy
- **Red ring**: Low energy

Nodes are marked as depleted when energy reaches zero.
```

## Best Practices

### Be Concise
- Use short, clear sentences
- Break up long paragraphs
- Use bullet points for lists

### Use Examples
- Include code examples for APIs
- Show before/after for changes
- Provide complete, runnable examples

### Keep Updated
- Update docs when code changes
- Mark deprecated features clearly
- Include version information where relevant

### Cross-Reference
- Link to related documentation
- Reference code files
- Point to examples

## Project-Specific Conventions

### File Organization
```
docs/
├── introduction/          # Overview, features, design
├── quickstart/           # Getting started guides
├── planned_concepts/     # Future features (design docs)
├── literature/           # References and citations
└── figures/              # Images and diagrams
```

### Citation Format
Use BibTeX references in `references.bib`:
```markdown
Network routing is based on shortest path [@dijkstra1959note].

## References
See [References](../literature/references.md) for full citations.
```

### Code Examples in Documentation
- Always use full, working examples
- Include imports
- Show expected output
- Test examples before documenting

## What NOT to Do

- ❌ Don't use generic "click here" links (use descriptive text)
- ❌ Don't leave broken links
- ❌ Don't copy/paste code without testing it
- ❌ Don't use absolute paths (use relative)
- ❌ Don't forget to update docs when changing code
- ❌ Don't use screenshots for code (use code blocks)
- ❌ Don't duplicate information (link instead)

## Templates

### New Feature Documentation Template
```markdown
# Feature Name

Brief one-sentence description.

## Overview
What is this feature and why does it exist?

## Usage
How to use this feature with examples.

## Configuration
Any configuration options.

## API Reference
Links to relevant classes/functions.

## Examples
Complete working examples.

## See Also
Related features or documentation.
```

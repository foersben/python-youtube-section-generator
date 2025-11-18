# Type Hints Conversion

## Task
Convert existing code to use precise, modern Python type hints following project standards.

## Requirements

### 1. Use Built-in Generics (Python 3.9+)

```python
# ❌ Old style (don't use)
from typing import List, Dict, Tuple, Optional

def process(items: List[str]) -> Dict[str, int]:
    results: Optional[Tuple[int, int]] = None
    return {}

# ✅ New style (use this)
def process(items: list[str]) -> dict[str, int]:
    results: tuple[int, int] | None = None
    return {}
```

### 2. Prefer Precise Types Over Broad Ones

```python
# ❌ Avoid
from typing import Any

def handle(data: Any) -> Any:
    return data['value']

# ✅ Better
def handle(data: dict[str, object]) -> object:
    return data['value']

# ✅ Best (when you know the structure)
def handle(data: dict[str, int]) -> int:
    return data['value']
```

### 3. Callable Types

```python
# ❌ Too vague
def process(callback: callable) -> None:
    pass

# ✅ Precise
from typing import Callable

def process(callback: Callable[[str, int], bool]) -> None:
    """Process with callback.

    Args:
        callback: Function taking (str, int) and returning bool.
    """
    pass
```

### 4. Union Types

```python
# ❌ Old style
from typing import Union, Optional

def get_value() -> Union[int, str]:
    pass

def find_item() -> Optional[int]:
    pass

# ✅ New style (Python 3.10+)
def get_value() -> int | str:
    pass

def find_item() -> int | None:
    pass
```

### 5. Type Guards and Narrowing

```python
def process(value: object) -> str:
    """Process value with type narrowing."""
    # Guard against None
    if value is None:
        raise ValueError("Value cannot be None")

    # Assert for type checker
    assert isinstance(value, str), "Expected string"

    # Now value is known to be str
    return value.upper()
```

### 6. Generic Collections

```python
# ❌ Not specific enough
def process_data(data: list) -> dict:
    pass

# ✅ Specific types
def process_data(data: list[tuple[str, int]]) -> dict[str, list[int]]:
    """Process data into grouped results.

    Args:
        data: List of (name, value) tuples.

    Returns:
        Dictionary mapping names to lists of values.
    """
    pass
```

## Common Conversions

### Event Handler Data

```python
# Before
from typing import Any

def handle(self, time: int, data: Any) -> None:
    path = data['path']
    packet = data['packet']

# After
def handle(self, time: int, data: dict[str, object]) -> None:
    # Extract with type annotations and type: ignore
    path: list[int] = data['path']  # type: ignore[assignment]
    packet: Packet = data['packet']  # type: ignore[assignment]
```

### Optional Parameters

```python
# Before
from typing import Optional

def create_node(logger: Optional[Logger] = None) -> None:
    pass

# After
def create_node(logger: Logger | None = None) -> None:
    pass
```

### Complex Nested Types

```python
# Before
from typing import Dict, List, Tuple

def analyze(
    data: Dict[str, List[Tuple[int, str]]]
) -> List[Dict[str, int]]:
    pass

# After
def analyze(
    data: dict[str, list[tuple[int, str]]]
) -> list[dict[str, int]]:
    pass
```

### Method Return Types

```python
class NetworkState:
    # Before
    def get_neighbors(self, node):
        return list(self.graph.neighbors(node))

    # After
    def get_neighbors(self, node: int) -> list[int]:
        """Get neighbor node IDs.

        Args:
            node: Node ID to query.

        Returns:
            List of neighboring node IDs.
        """
        return list(self.graph.neighbors(node))
```

## Matplotlib Type Hints

```python
# Before
def create_plot(ax, data):
    pass

# After
from matplotlib.axes import Axes

def create_plot(ax: Axes, data: list[float]) -> None:
    """Create plot on given axes.

    Args:
        ax: Matplotlib axes to plot on.
        data: Data points to plot.
    """
    pass
```

## NetworkX Type Hints

```python
from networkx import Graph

# Node/edge attributes
def update_node(graph: Graph, node: int, energy: float) -> None:
    """Update node energy level."""
    graph.nodes[node]['energy'] = energy

def get_edge_data(graph: Graph, u: int, v: int) -> dict[str, object]:
    """Get edge attributes."""
    return graph.edges[u, v]
```

## Type Ignore Comments

Use `# type: ignore[specific-error]` when necessary:

```python
# For extracting from untyped dict
value: MyType = data['key']  # type: ignore[assignment]

# For Tkinter pack (false positive)
frame.pack(side=tk.LEFT)  # type: ignore[arg-type]

# For matplotlib colormap access
artist.remove()  # type: ignore[attr-defined]
```

## Checklist

- [ ] Replace `List`, `Dict`, `Tuple` with `list`, `dict`, `tuple`
- [ ] Replace `Optional[T]` with `T | None`
- [ ] Replace `Union[A, B]` with `A | B`
- [ ] Replace `Any` with `object` or specific type
- [ ] Replace bare `callable` with `Callable[[Args], Return]`
- [ ] Add return type annotations to all functions
- [ ] Add parameter type annotations to all functions
- [ ] Add type narrowing guards where needed
- [ ] Use `# type: ignore[specific]` instead of bare `# type: ignore`
- [ ] Run static type checker (`mypy` or IDE checker)
- [ ] Ensure tests still pass

## Testing After Conversion

```bash
# Run tests
poetry run pytest

# Check with mypy (if configured)
poetry run mypy src/

# Or use IDE's built-in checker
```

## Migration Strategy

1. **Start with interfaces/base classes**
   - Observable, Observer, EventHandler
   - Configuration classes

2. **Move to data structures**
   - Packets, State classes
   - Config dataclasses

3. **Then implementation classes**
   - Event handlers
   - Network state
   - Resource managers

4. **Finally UI components**
   - Panels, widgets
   - Renderers

5. **Test after each module**
   - Run affected tests
   - Check for type errors
   - Verify runtime behavior

## Common Pitfalls

❌ **Over-specifying internal types**
```python
# Don't need to type every variable
def process(items: list[int]) -> int:
    # Fine without annotation (inferred)
    total = 0
    for item in items:
        total += item
    return total
```

❌ **Forgetting to import Callable**
```python
# Missing import
def register(callback: Callable[[int], None]) -> None:
    pass

# Should be
from typing import Callable

def register(callback: Callable[[int], None]) -> None:
    pass
```

❌ **Using wrong type ignore**
```python
# Too broad
value = data['key']  # type: ignore

# Specific (better)
value: MyType = data['key']  # type: ignore[assignment]
```

## Resources

- [PEP 585](https://peps.python.org/pep-0585/) - Built-in generics
- [PEP 604](https://peps.python.org/pep-0604/) - Union operator `|`
- [typing module docs](https://docs.python.org/3/library/typing.html)

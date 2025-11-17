# Refactoring Guide

## Task
Safely refactor code to improve structure, readability, or performance while maintaining correctness.

## Pre-Refactoring Checklist

- [ ] Run full test suite and document baseline (all tests should pass)
- [ ] Understand the code's current behavior completely
- [ ] Identify specific improvement goals
- [ ] Plan refactoring in small, testable steps
- [ ] Commit current working state

## Refactoring Workflow

### 1. Make One Change at a Time

```python
# ❌ Don't do multiple refactors at once
# - Rename variables
# - Extract methods  
# - Change algorithm
# - Update types

# ✅ Do one focused change
# Step 1: Extract method (test)
# Step 2: Rename for clarity (test)
# Step 3: Update types (test)
```

### 2. Test After Each Change

```bash
# After each refactoring step
poetry run pytest

# For specific module
poetry run pytest tests/test_module.py

# With coverage
poetry run pytest --cov=src/module
```

### 3. Commit Frequently

```bash
# After each successful refactoring step
git add .
git commit -m "refactor(module): extract validation logic to helper method"
```

## Common Refactoring Patterns

### Extract Method

```python
# Before: Long method doing multiple things
def process_packet(self, node: int, packet: Packet, time: int) -> None:
    """Process packet at node."""
    # Validate node has energy
    if self.network.graph.nodes[node]['energy'] < self.energy.calculate_rx_energy():
        self.logger.debug(f"Node {node} has no energy")
        self.network.graph.nodes[node]['energy'] = 0
        return
    
    # Validate buffer capacity
    if self.network.graph.nodes[node]['tx_rx_buffer'] >= self.config.tx_rx_buffer_size:
        self.logger.debug(f"Node {node} buffer full")
        return
    
    # Process packet
    self.network.graph.nodes[node]['energy'] -= self.energy.calculate_rx_energy()
    self.network.graph.nodes[node]['tx_rx_buffer'] += 1

# After: Extracted validation methods
def process_packet(self, node: int, packet: Packet, time: int) -> None:
    """Process packet at node."""
    if not self._has_sufficient_energy(node):
        return
    
    if not self._has_buffer_space(node):
        return
    
    self._consume_energy(node)
    self._add_to_buffer(node)

def _has_sufficient_energy(self, node: int) -> bool:
    """Check if node has energy for reception."""
    energy_required = self.energy.calculate_rx_energy()
    if self.network.graph.nodes[node]['energy'] < energy_required:
        self.logger.debug(f"Node {node} has no energy")
        self.network.graph.nodes[node]['energy'] = 0
        return False
    return True

def _has_buffer_space(self, node: int) -> bool:
    """Check if node has buffer space."""
    if self.network.graph.nodes[node]['tx_rx_buffer'] >= self.config.tx_rx_buffer_size:
        self.logger.debug(f"Node {node} buffer full")
        return False
    return True
```

### Rename for Clarity

```python
# Before: Vague names
def proc(n, p, t):
    d = self.g.nodes[n]
    e = d['e']
    # ...

# After: Clear names
def process_packet(node_id: int, packet: Packet, time: int) -> None:
    """Process packet reception at node."""
    node_data = self.network.graph.nodes[node_id]
    current_energy = node_data['energy']
    # ...
```

### Simplify Conditionals

```python
# Before: Complex nested conditions
def can_transmit(self, node: int) -> bool:
    if self.graph.nodes[node]['energy'] > 0:
        if self.graph.nodes[node]['tx_count'] > 0:
            if self.graph.nodes[node]['tx_rx_buffer'] > 0:
                return True
    return False

# After: Early returns
def can_transmit(self, node: int) -> bool:
    """Check if node can transmit packet."""
    if self.graph.nodes[node]['energy'] <= 0:
        return False
    
    if self.graph.nodes[node]['tx_count'] <= 0:
        return False
    
    if self.graph.nodes[node]['tx_rx_buffer'] <= 0:
        return False
    
    return True

# Or: Single expression
def can_transmit(self, node: int) -> bool:
    """Check if node can transmit packet."""
    node_data = self.graph.nodes[node]
    return (
        node_data['energy'] > 0
        and node_data['tx_count'] > 0
        and node_data['tx_rx_buffer'] > 0
    )
```

### Replace Magic Numbers

```python
# Before: Magic numbers
def calculate_energy(self, distance: float) -> float:
    return 0.01 + (distance / 10) * 0.1

# After: Named constants
class EnergyManager:
    BASE_TX_COST = 0.01
    DISTANCE_SCALE = 10
    TX_ENERGY_PER_UNIT = 0.1
    
    def calculate_energy(self, distance: float) -> float:
        """Calculate transmission energy based on distance."""
        return self.BASE_TX_COST + (distance / self.DISTANCE_SCALE) * self.TX_ENERGY_PER_UNIT
```

### Consolidate Duplicate Code

```python
# Before: Duplicated logic
def transmit_to_neighbor(self, node: int, neighbor: int) -> None:
    if self.graph.nodes[node]['energy'] < self.energy.calculate_tx_energy():
        self.logger.debug(f"Node {node} has no energy")
        self.graph.nodes[node]['energy'] = 0
        return
    # ... transmit logic

def receive_from_neighbor(self, node: int, sender: int) -> None:
    if self.graph.nodes[node]['energy'] < self.energy.calculate_rx_energy():
        self.logger.debug(f"Node {node} has no energy")
        self.graph.nodes[node]['energy'] = 0
        return
    # ... receive logic

# After: Shared helper
def _check_and_deplete_if_insufficient(
    self,
    node: int,
    required_energy: float
) -> bool:
    """Check if node has energy, deplete if not.
    
    Returns:
        True if node has sufficient energy, False otherwise.
    """
    if self.graph.nodes[node]['energy'] < required_energy:
        self.logger.debug(f"Node {node} has no energy")
        self.graph.nodes[node]['energy'] = 0
        return False
    return True

def transmit_to_neighbor(self, node: int, neighbor: int) -> None:
    if not self._check_and_deplete_if_insufficient(
        node, self.energy.calculate_tx_energy()
    ):
        return
    # ... transmit logic
```

### Split Large Class

```python
# Before: God class doing everything
class NetworkManager:
    def create_graph(self): ...
    def initialize_nodes(self): ...
    def calculate_paths(self): ...
    def render_network(self): ...
    def handle_events(self): ...
    def manage_energy(self): ...

# After: Separate responsibilities
class NetworkState:
    """Manages network graph and node state."""
    def create_graph(self): ...
    def initialize_nodes(self): ...

class PathCalculator:
    """Calculates routing paths."""
    def calculate_paths(self): ...

class NetworkRenderer:
    """Renders network visualization."""
    def render_network(self): ...

class EnergyManager:
    """Manages energy consumption."""
    def calculate_energy(self): ...
    def deduct_energy(self): ...
```

## Testing Strategy

### Before Refactoring
```python
# Add characterization tests if coverage is low
def test_current_behavior_baseline():
    """Document current behavior before refactoring."""
    result = function_to_refactor(input_data)
    # This test locks in current behavior
    assert result == expected_current_output
```

### During Refactoring
```python
# Tests should pass after each step
# Run: poetry run pytest -v

# If tests fail:
# 1. Is the failure expected (bug fix)?
# 2. Did I break something?
# 3. Revert and try smaller step
```

### After Refactoring
```python
# Add tests for new extracted methods
def test_extracted_helper_method():
    """Test new helper method in isolation."""
    result = obj._new_helper_method(input)
    assert result == expected
```

## Performance Refactoring

### Profile First
```python
import cProfile
import pstats

# Profile the code
profiler = cProfile.Profile()
profiler.enable()

# Run code to profile
slow_function()

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(10)  # Top 10 slow functions
```

### Common Optimizations

```python
# Before: Repeated graph lookups
for node in nodes:
    energy = self.graph.nodes[node]['energy']
    buffer = self.graph.nodes[node]['tx_rx_buffer']
    # ... multiple more lookups

# After: Cache node data
for node in nodes:
    node_data = self.graph.nodes[node]
    energy = node_data['energy']
    buffer = node_data['tx_rx_buffer']
    # ... use node_data

# Before: List comprehension with filter
result = [process(x) for x in items if is_valid(x)]

# After: Generator for large lists
result = (process(x) for x in items if is_valid(x))
```

## Code Smells to Fix

### Long Parameter Lists
```python
# Before
def create_node(
    self, x, y, energy, buffer_size, tx_max, rx_max, types, is_base
):
    pass

# After: Use config object
def create_node(self, position: tuple[float, float], config: NodeConfig):
    pass
```

### Feature Envy
```python
# Before: Method uses another object's data extensively
class PacketHandler:
    def process(self, packet):
        sender_energy = packet.sender.graph.nodes[packet.sender_id]['energy']
        sender_buffer = packet.sender.graph.nodes[packet.sender_id]['buffer']
        # More sender operations...

# After: Move method to the envied class
class NetworkState:
    def get_node_status(self, node: int) -> NodeStatus:
        """Get comprehensive node status."""
        return NodeStatus(
            energy=self.graph.nodes[node]['energy'],
            buffer=self.graph.nodes[node]['buffer'],
            # ...
        )
```

## Refactoring Checklist

- [ ] Tests pass before starting
- [ ] One logical change per commit
- [ ] Tests pass after each change
- [ ] Code is more readable after change
- [ ] No functionality change (unless bug fix)
- [ ] Performance not degraded (profile if critical)
- [ ] Type hints updated/added
- [ ] Docstrings updated
- [ ] No dead code left behind

## When NOT to Refactor

- ❌ Code works fine and isn't being modified
- ❌ No tests exist (write tests first)
- ❌ Under time pressure (creates risk)
- ❌ Don't understand the code (study first)
- ❌ Just to use a "cool" pattern

## Safe Refactoring Order

1. **Rename** (safest, IDE-supported)
2. **Extract method** (testable)
3. **Move method** (structural)
4. **Change interface** (riskier, needs careful testing)
5. **Algorithmic change** (riskiest, needs extensive testing)

Start safe, test frequently, commit often.


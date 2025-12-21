# Plugin Marketplace: Plugin Info

Display detailed information about a specific installed plugin.

## Task

When invoked with a plugin name as argument (e.g., `/plugin-info prism-ve-swarm`):

1. Look up the plugin in `.claude/marketplace.json`
2. Display full details including:
   - Name, version, description
   - Installation date
   - Source path
   - Tags
   - List of all components (agents, commands, skills)

If no argument provided, list all available plugins to choose from.

## Output Format

```
Plugin: [name]
Version: [version]
Description: [description]
Installed: [date]
Source: [path]
Tags: [tag1, tag2, ...]

Components:
  Agents (N):
    - agent-1
    - agent-2
  Commands (M):
    - /command-1
    - /command-2
  Skills (K):
    - skill-1
```

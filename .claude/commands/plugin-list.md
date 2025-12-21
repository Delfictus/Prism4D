# Plugin Marketplace: List Installed Plugins

List all installed plugins from the PRISM-4D marketplace.

## Task

Read the marketplace registry at `.claude/marketplace.json` and display:

1. **Installed Plugins**: Show name, version, description, and components for each installed plugin
2. **Available Plugins**: Show any plugins available for installation
3. **Summary**: Total counts of agents, commands, and skills installed

## Output Format

```
PRISM-4D Plugin Marketplace
===========================

Installed Plugins:
- [plugin-name] v[version]
  Description: [description]
  Components: [N agents, M commands, K skills]

Available for Installation:
- (none)

Summary:
- Total Agents: N
- Total Commands: M
- Total Skills: K
```

Read `.claude/marketplace.json` and format the output as shown above.

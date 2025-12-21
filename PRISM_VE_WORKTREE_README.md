# PRISM-VE Development Worktree

## Purpose

This worktree is for developing PRISM-VE (Viral Evolution predictor) without breaking the publication-ready PRISM-Viral system.

## Worktree Strategy

**Main Working Directory:** `/mnt/c/Users/Predator/Desktop/PRISM`
- Branch: `prism-viral-escape`
- Status: **PUBLICATION-READY** (Nature Methods)
- DO NOT BREAK THIS!

**PRISM-VE Worktree:** `/mnt/c/Users/Predator/Desktop/prism-ve`
- Branch: `prism-ve-development`
- Status: **DEVELOPMENT** (safe to experiment)
- Will merge back when ready

## What's in Each

### PRISM (Main) - PRISM-Viral (Escape Only)
```
✅ Escape prediction working
✅ 3/3 viruses beat EVEscape
✅ Nature Methods manuscript
✅ READY TO SUBMIT
```

### PRISM-VE (Worktree) - Full Platform
```
⏳ Escape module (copy from main)
⏳ Fitness module (ADD HERE)
⏳ Cycle module (ADD HERE)
⏳ Integration layer
```

## Development Plan

**Week 1-2: Fitness Module**
- Add ΔΔG prediction
- Validate on DMS functional data
- Commit: `prism-ve-fitness-module-complete`

**Week 3-5: Cycle Module**
- GISAID integration
- Phase detection (6 phases)
- Temporal prediction
- Commit: `prism-ve-cycle-module-complete`

**Week 6: Integration**
- Combine all modules
- End-to-end testing
- Merge to main: `prism-ve-v1.0-complete`

## Switching Between

**Work on PRISM-Viral (publication):**
```bash
cd /mnt/c/Users/Predator/Desktop/PRISM
# Safe, stable, publication-ready
```

**Work on PRISM-VE (development):**
```bash
cd /mnt/c/Users/Predator/Desktop/prism-ve
# Experimental, can break things
```

**Merge PRISM-VE back when ready:**
```bash
cd /mnt/c/Users/Predator/Desktop/PRISM
git merge prism-ve-development
```

## Benefits

✅ **Safety:** Can't accidentally break publication-ready system
✅ **Parallel:** Can submit paper while developing PRISM-VE
✅ **Isolation:** Experimental features don't affect stable version
✅ **Easy rollback:** Just delete worktree if needed
✅ **Clean history:** Clear separation of stable vs development

## Current Status

**PRISM-Viral (Main):**
- Ready for Nature Methods submission
- All tests passing
- Complete manuscript
- Tagged: `publication-ready-nature-methods`

**PRISM-VE (This Worktree):**
- Starting point: Copy of PRISM-Viral
- Will add: Fitness + Cycle modules
- Target: Nature paper (Phase II)

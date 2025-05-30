 Agent Workflow Maximization Template

## Agent Preferences
- Auto-commit and push after every change: YES
- Summarize every commit: YES
- Ask before deleting files: YES
- Use batch mode for experiments: NO
- Trust all agent commits and changes: YES 

---

## Blog Integration Workflow
- After each important experiment or analysis, generate a markdown blog post draft in the blog's _posts/ directory.
- Use the following template for consistency:

```
---
title: "[Experiment/Idea Title]"
date: YYYY-MM-DD
categories: [transformers, deep learning, experiments]
---

## Summary
[Short summary of the experiment]

## Background
[Why you did this experiment]

## Experiment
- Model: [description]
- Data: [description]
- Method: [description]

## Results
- [Key findings, plots, tables]

## Insights & Next Steps
- [What you learned, what's next]

## Resources
- [Links to code, notebooks, results]
```

- Save plots from results/ to the blog's assets/ or images/ directory and reference them in the markdown.
- Not every experiment needs a post—focus on the most interesting or insightful runs.
- Use the blog as a lab notebook and a way to communicate work to others.
- After running an experiment, auto-generate a summary and blog post draft, save relevant plots, and remind to review/publish.
- Track blog post ideas and drafts in this workflow file if needed. 
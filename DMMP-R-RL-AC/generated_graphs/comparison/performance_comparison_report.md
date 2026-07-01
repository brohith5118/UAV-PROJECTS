# DMMP-R-RL-AC Comparison Report

| Metric | TSA local search off | Proposed DMMP-R-RL-AC |
| :--- | :---: | :---: |
| Overall completion rate | 72.500% | 72.500% |
| High-priority completion rate | 100.000% | 100.000% |
| Backup pool size | 5 | 5 |
| Overloaded UAV count | 0 | 0 |
| Total travel distance | 61101.1 m | 61101.1 m |
| Mean energy utilisation | 49.696% | 49.696% |
| Mean compute utilisation | 61.111% | 61.111% |
| Jain fairness index | 0.909 | 0.909 |
| Runtime | 8.12 s | 8.24 s |
| Runtime ratio | 1.00x | 0.99x |

The comparison keeps the D-module and PR-module identical and toggles only the
TSA route-refinement flag. This avoids mixing architectural changes with route
planning effects.

Bitonic Sorter
===============

These are examples of bitonic sorters implemented with [legion](https://github.com/StanfordLegion/legion).

**Bitonic sorter**: https://en.wikipedia.org/wiki/Bitonic_sorter

## simple task

Directory `simple_task` is an implementation using simple legion tasks. Values are passed to sub-tasks through `TaskArgument`, `Future`, and returned as serializable structs.
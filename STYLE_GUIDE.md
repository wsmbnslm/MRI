# Code Style Guide

Coding conventions for this repository (exercises, solutions, exams). The goal is one
boring, uniform style so code review is about content, not formatting.

## Formatter: Black

All code — `.py` files and code cells inside `.ipynb` notebooks — is formatted with
[Black](https://black.readthedocs.io/), using its defaults (88-character line length,
double quotes). No project-specific Black config; using the defaults everywhere is the
whole point.

Formatting notebooks in place requires the `jupyter` extra:

```bash
pip install "black[jupyter]"
black .                # format .py and .ipynb files in place
black --check .        # verify formatting without changing anything
```

Run Black before committing. It is necessary but not sufficient — see below.

## Imports

Standard library, then third-party, then local (`mri.*`), each group separated by a
blank line. `isort` (already installed, configured via `pyproject.toml` in this repo)
enforces this automatically:

```bash
isort .
```

## Naming

- `snake_case` for variables and functions, `PascalCase` for classes.
- English identifiers throughout — matches the existing exercise/solution notebooks
  and keeps identifiers
  greppable regardless of who's reading.
- Short, math-notation names (`phi`, `gamma`, `nx`, `ny`, `dx`, `dy`) are fine *only*
  when they mirror a symbol defined in the lecture/exam text right next to the code.
  Everything else gets a full descriptive name (`det_index`, not `di`).
- Suffix, don't abbreviate, for a broadcast/derived version of an existing variable:
  `ray_pos_x` (per-ray, 1-D) vs. `pos_x = ray_pos_x[:, None]` (broadcast column), not
  `rpx` / `rpxb`.

## Comments and docstrings

Comments, prose, identifiers, and error messages are English throughout (matching
the course materials). Keep it that way — don't mix languages within a single comment.

Functions are documented with a **numpydoc docstring**: one-line summary, then
`Parameters` (with types) and `Returns` sections. This is the one convention for
all Python code — `src/`, `scripts/`, `tests/`, and notebook code cells that define
functions. No `#`-comment blocks as function documentation.

## Type hints

Function signatures get type hints (`def f(x: torch.Tensor, n: int) -> torch.Tensor:`).
Local variables generally don't need them — infer from the assignment.

## The readability rule Black can't enforce for you

Black fixes line length, nothing else. Given a dense expression that mixes broadcasting
indexing with several operators, it just wraps the *same* expression across more lines:

```python
# technically under 88 chars, still hard to read
coords_a_x = (
    ray_pos_x[:, None]
    + (x_edges[None, :] - ray_pos_y[:, None])
    / safe_dir_y[:, None]
    * ray_dir_x[:, None]
)
```

**Rule of thumb:** if an expression combines broadcasting indexing (`[:, None]`,
`[None, :]`) *and* more than one arithmetic/comparison operator, break it into named
intermediate steps instead of letting Black wrap it:

```python
pos_x, pos_y = ray_pos_x[:, None], ray_pos_y[:, None]
dir_x, dir_y = ray_dir_x[:, None], ray_dir_y[:, None]

t_a = (x_edges[None, :] - pos_y) / safe_dir_y
coords_a_x = pos_x + t_a * dir_x
```

Two things make this easy to apply consistently:

1. **Broadcast once, name it, reuse it.** If `tensor[:, None]` appears more than once
   for the same tensor in a function, assign it to a name at the top instead of
   repeating the indexing at every use site.
2. **Name an intermediate when it has physical/mathematical meaning**, not just to
   shorten a line. `t_a` above is "the ray parameter at which it crosses this edge" —
   naming it documents a derivation step that a wrapped one-liner hides.

This tends to remove duplicate computation for free, too: caching `cos_phi = torch.cos(phi)`
once instead of calling `torch.cos(phi)` at every use site is simultaneously more
readable and cheaper.

## Notebooks

- One logical step per cell; a markdown cell explaining *why* precedes the code that
  does it — this is the existing exam/exercise convention, keep following it.
- Before committing a solution notebook, re-run it top to bottom so committed outputs
  match the current code (outputs are intentionally kept in solution notebooks, since
  reproducing them requires the private dataset).

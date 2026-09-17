# Code Review Agent -- Style & Review Guide

## Formatting & naming
- Functions and variables: snake_case. Classes: PascalCase.
- Module-level constants: ALL_CAPS.
- No single-letter variable names, except loop indices `i`, `j`, `k`.
- Every public function/method needs a one-line docstring.
- Prefer f-strings over `.format()` or `%` for string interpolation.
- Never `from module import *`.

## Correctness & safety policies (not just cosmetics)
- Never build SQL by concatenating or interpolating a variable into the
  query string. Always use parameterized queries.
- Check for `None` before dereferencing an argument that can legitimately
  be `None` (per its docstring or obvious calling convention).
- Don't catch `Exception` (or bare `except:`) and silently discard it --
  either narrow the exception type, or log and re-raise.
- Don't hardcode credentials, API keys, or secrets -- read them from
  configuration or environment variables.
- Prefer `with` for file handles and other closable resources over a
  manual open/close.
- Don't use `pickle` (or similarly unsafe deserializers) on data that
  didn't originate from this process.

The `braces@3.0.3.patch` bounds brace nesting during parsing and recursive AST traversal
during compilation, expansion, and stringification. It mitigates
[GHSA-vfj7-8cjw-p6xm](https://github.com/advisories/GHSA-vfj7-8cjw-p6xm), for which no
upstream patched version is available. It affects development and release globbing; it is
not part of the codec or the npm runtime package. The upstream license is retained in
`braces.LICENSE.txt`.

Remove the patch when an upstream release provides equivalent protection. Before changing
it, compare normal nested globs and deeply nested balanced and unclosed patterns through
the public `parse`, `compile`, `expand`, and `stringify` APIs. Oversized nesting must fail
with a deliberate `SyntaxError` rather than exhaust the call stack.

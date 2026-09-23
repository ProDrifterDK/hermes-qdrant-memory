"""Executable inventory of the routes that can delete or replace a payload.

Four independent reviews of this closure found the same defect in four different
places: `qdrant_memory_store`, `qdrant_memory_extraction_approve`, restore and the
off-mode reindex each decided "is this payload protected?" from a local, narrower list
of fields. Every one of them was found by reading code, none by a failing test, and the
route table in ``docs/SAFETY.md`` claimed coverage that did not exist.

The fix for the four sites is the shared predicate. The fix for the *pattern* is this
file: the mutating surface is derived from the code — tool dispatch, CLI commands and
provider hook overrides — every derived entry has to be classified, and a route declared
protected has to name a test that actually exists. Adding a tool, a command or a hook
now fails until someone decides what it does to a protected payload, and the SAFETY
route table cannot claim a pin that has been renamed or deleted.

Classification is by hand because "does this command mutate Qdrant?" is not derivable
from the parser. ``uncovered`` exists so that the honest answer has a place to go: a
route whose refusal was reasoned from code but never executed stays visible as debt
instead of being silently counted as covered.

The classification is itself pinned: the exact protection set and the exact pin list of
every protection row are frozen below, and the detectors that consume them are tested
against deliberately unclassified and deliberately broken inputs. A hand-written
classification that no test defends is the same claim surface that failed three times.
"""
from __future__ import annotations

import argparse
import ast
import importlib.util
import sys
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent

# --- the surface, derived from the code ------------------------------------


def _plugin_module():
    """This plugin's ``__init__.py``, loaded by path.

    ``import qdrant_memory`` would resolve to whatever is installed; the derivations must
    read the tree under test.
    """
    spec = importlib.util.spec_from_file_location("_route_inventory_plugin", REPO / "__init__.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _comparison_strings(node: ast.AST) -> set[str]:
    """Every string constant any comparison inside ``node`` puts on either side.

    The dispatch used to be read as ``if <x> == "<name>"``, taking
    ``comparators[0]`` for the name. That saw only that one shape, so
    ``if tool_name in ("qdrant_memory_wipe_all",):`` and
    ``if "qdrant_memory_wipe_all" == tool_name:`` were invisible: a new destructive route
    could ship unclassified with the size pin and the completeness detector both saying
    fine. Read both sides of the comparison, expand container literals, and keep the
    prefix filter only as a hint — the schema cross-check below is what makes an
    unprefixed name a failure rather than a silent miss.
    """
    found: set[str] = set()

    def collect(expr: ast.AST) -> None:
        if isinstance(expr, ast.Constant) and isinstance(expr.value, str):
            found.add(expr.value)
        elif isinstance(expr, (ast.Tuple, ast.List, ast.Set)):
            for element in expr.elts:
                collect(element)
        elif isinstance(expr, ast.BinOp):
            collect(expr.left)
            collect(expr.right)
        elif isinstance(expr, ast.JoinedStr):
            for value in expr.values:
                collect(value)

    for sub in ast.walk(node):
        if isinstance(sub, ast.Compare):
            collect(sub.left)
            for comparator in sub.comparators:
                collect(comparator)
    return found


def _dispatched_tool_names() -> set[str]:
    """Every string constant ``handle_tool_call`` compares against.

    No prefix filter: a name this dispatch answers to is part of the surface whether or
    not it is spelled like one. Filtering by ``qdrant_memory_``/``qdrant_learning_`` left
    a dispatch branch on an unprefixed name invisible to the derivation, the size pin, the
    detector and the schema cross-check at once — the same "the check looks at the shape
    it expects" defect as the comparison form itself. Every constant is now reported, and
    a non-tool comparison here fails loudly instead of being silently skipped, which is a
    decision a human makes in this file rather than one the parser makes for them.
    """
    tree = ast.parse((REPO / "__init__.py").read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "handle_tool_call":
            names |= _comparison_strings(node)
            unreadable = [
                ast.unparse(sub)
                for sub in ast.walk(node)
                if isinstance(sub, ast.Compare)
                and any(isinstance(part, ast.Name) and part.id == "tool_name" for part in ast.walk(sub))
                and not _comparison_strings(sub)
            ]
            assert not unreadable, (
                "handle_tool_call compares tool_name against something that is not a string "
                "constant, so the name this route answers to cannot be read from the source: "
                f"{unreadable}"
            )
    assert names, "handle_tool_call dispatch could not be parsed"
    return names


# The shape a tool name has to match to be seen at all. Digits are in the character class
# because a name the host can register with a digit in it (`qdrant_memory_retrieve_v2`) was
# invisible to all four derivations: (1) reads comparisons inside one function, (3) and (4)
# filter by this pattern, and (2) is the only one that would have caught it. Every declared
# name is now asserted to match this pattern, so a name outside it fails loudly instead of
# being skipped by three of the four reads.
_TOOL_NAME_SHAPE = re.compile(r"^(?:qdrant_memory_|qdrant_learning_)[a-z0-9_]+$")

# A ``_tool_*`` implementation the dispatch does not reach from ``handle_tool_call``, in two
# classes that are checked against the source rather than described in prose. A reason string
# is a claim, and this file shipped one that was false: ``retrieve_learning`` was excused as
# "a direct-Python-caller route only, no branch in handle_tool_call", but ``_tool_retrieve``
# dispatches ``collection="learning"`` straight to it, so ``qdrant_memory_retrieve`` reaches
# it (review 8, N1). Both classes now carry a testable form.
#
# - ``_HELPER_TOOL_METHODS``: reached from a *dispatched* handler instead of from
#   ``handle_tool_call``. It is a real route — the host reaches it by calling the tool whose
#   handler calls it. Each entry names the reaching route, the call site inside that route's
#   handler, and the code evidence for its read-only claim; all three are asserted, and the
#   handler is read from the dispatch branch rather than derived from the route's name.
#   Review 9 falsified the first version of this entry twice, both times by mutating what the
#   check was *about* rather than what it read: the prose mutation survived because a docstring
#   sentence and the statement it describes render to the same text, and then, with the read
#   moved to the AST, two mutations of the real call survived because the keyword was collected
#   from the whole function instead of from the call it is about. Every claim here is now read
#   from the tree and bound to the exact node it describes: the branch for the handler, the
#   call for the argument.
# - ``_UNREACHED_TOOL_METHODS``: no dispatch name, no schema entry, and no call site in the
#   module. Currently empty, and it cannot be parked in while a handler calls the method,
#   because the call-site check runs for every entry.
_HELPER_TOOL_METHODS = {
    "retrieve_learning": {
        "reached_from": "qdrant_memory_retrieve",
        "call_site": "self._tool_retrieve_learning",
        "read_only_evidence": {
            "kind": "keyword_false",
            "keyword": "update_access",
            "callee": "search",
        },
        "reason": (
            "reachable through a declared tool: the dispatch branch that tests "
            "``qdrant_memory_retrieve`` calls ``_tool_retrieve``, which dispatches "
            "``collection=\"learning\"`` to it. It has no ``TOOL_SCHEMAS`` entry and no "
            "dispatch name of its own, which is exactly what makes it invisible to a name "
            "read. Read-only by contract — every call to ``store.search`` in its code passes "
            "``update_access=False``, so it does not bump ``last_accessed``/``access_count`` "
            "either. The handler is read from the branch and the keyword from the call: "
            "neither claim is satisfied by the route's name or by a literal elsewhere in the "
            "body."
        ),
    },
}
_UNREACHED_TOOL_METHODS: dict[str, str] = {}


def _literal_tool_names() -> set[str]:
    """Every tool-name-shaped string literal anywhere in the plugin module.

    The dispatch read asks one question about one function and one syntax: which names
    does ``handle_tool_call`` compare against. This reads the file instead. A name the
    module spells out at all — a module-level constant, a dict key, a ``match``/``case``
    pattern, a second dispatch function, a ``startswith`` prefix — is part of the surface
    whether or not any single ``Compare`` mentions it. Five shapes that passed the
    previous derivations are caught here, each duplicated as a harness mutant.
    """
    tree = ast.parse((REPO / "__init__.py").read_text(encoding="utf-8"))
    return {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and _TOOL_NAME_SHAPE.match(node.value)
    }


def _module_source() -> str:
    return (REPO / "__init__.py").read_text(encoding="utf-8")


def _module_tree() -> ast.Module:
    return ast.parse(_module_source())


def _module_classes(tree: ast.Module) -> dict[str, ast.ClassDef]:
    return {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}


def _module_functions(tree: ast.Module) -> dict[str, ast.FunctionDef]:
    return {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}


def _class_names_in(node: ast.AST) -> set[str]:
    """Names of classes the type is defined from, from its module-local bases.

    ``vars()`` on one class is the read this whole file exists to replace: a mixin's method
    is still called through the provider, and a base read that stops at one body cannot see
    it.
    """
    names: set[str] = set()
    for sub in ast.walk(node) if not isinstance(node, ast.ClassDef) else node.bases:
        if isinstance(sub, ast.Name):
            names.add(sub.id)
    return names


def _provider_mro_closure(tree: ast.Module) -> set[str]:
    """Every module-local class the provider inherits from, transitively."""
    classes = _module_classes(tree)
    assert "QdrantMemoryProvider" in classes, "QdrantMemoryProvider is not defined in __init__.py"
    seen: set[str] = set()
    pending = ["QdrantMemoryProvider"]
    while pending:
        name = pending.pop()
        if name in seen or name not in classes:
            continue
        seen.add(name)
        pending.extend(sorted(_class_names_in(classes[name])))
    return seen


def _tool_methods_by_class(tree: ast.Module) -> dict[str, set[str]]:
    """Class name -> every ``_tool_*`` method that class body defines."""
    out: dict[str, set[str]] = {}
    for name, cls in _module_classes(tree).items():
        methods = {
            sub.name[len("_tool_"):]
            for sub in cls.body
            if isinstance(sub, ast.FunctionDef) and sub.name.startswith("_tool_")
        }
        if methods:
            out[name] = methods
    return out


def _tool_method_names(tree: ast.Module | None = None) -> set[str]:
    """Every ``self._tool_*`` implementation the provider can call, over its whole MRO.

    Reading only the provider's own ``ClassDef`` body is the same ``vars()``-style read the
    hook derivation was fixed for in round 6 and review 8's N3 named here: a ``_tool_*``
    defined on a mixin is called through the provider, so it is a route, but it was not in
    the defined set. A ``_tool_*`` on a class *outside* that closure is loud, because either
    it is inherited (and this read is wrong) or it is a method named like a route that
    nothing calls.
    """
    tree = tree if tree is not None else _module_tree()
    closure = _provider_mro_closure(tree)
    by_class = _tool_methods_by_class(tree)
    outside = {name: sorted(m) for name, m in by_class.items() if name not in closure}
    assert not outside, (
        "a _tool_* method is defined on a class outside the provider's module-local MRO: "
        f"{outside} — either it is reached through inheritance this read cannot follow, or "
        "it is dead code shaped like a route"
    )
    return {method for name in closure for method in by_class.get(name, set())}


def _function_node(name: str) -> ast.FunctionDef:
    """The one module-level or class-level function with this name."""
    for node in ast.walk(_module_tree()):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name} is not defined in __init__.py")


def _self_attribute_calls(name: str) -> set[str]:
    """``self.<attr>`` calls made by one function's code, as ``self.<attr>`` strings.

    Read from ``ast.Call`` nodes rather than by searching unparsed source. That distinction is
    this file's whole subject: unparsed source renders a docstring sentence and the statement
    it describes to the same text, so a substring read cannot tell the mechanism from the
    sentence about it. Review 9 found exactly that here — the claim "read-only, because the
    code passes ``update_access=False``" was satisfiable by the docstring saying so, and a
    mutation of the sentence survived untouched code. Presence and absence claims about code
    are read from the tree, not from its rendering.
    """
    found: set[str] = set()
    for sub in ast.walk(_function_node(name)):
        if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute):
            base = sub.func.value
            if isinstance(base, ast.Name) and base.id == "self":
                found.add(f"self.{sub.func.attr}")
    return found


def _read_only_violations_in(node: ast.AST, keyword: str, callee: str, where: str) -> list[str]:
    """``_read_only_call_violations`` over any node, so a synthetic function can be fed to it."""
    violations: list[str] = []
    calls = 0
    for sub in ast.walk(node):
        if not isinstance(sub, ast.Call) or not isinstance(sub.func, ast.Attribute):
            continue
        if sub.func.attr != callee:
            continue
        calls += 1
        passed = [kw for kw in sub.keywords if kw.arg == keyword]
        if not passed:
            violations.append(
                f"line {sub.lineno} calls .{callee}(...) without {keyword}= at all"
            )
            continue
        for kw in passed:
            value = kw.value
            if isinstance(value, ast.Constant) and value.value is False:
                continue
            violations.append(f"line {sub.lineno} passes {keyword}={ast.unparse(value)}")
    if not calls:
        violations.append(
            f"{where} makes no call to .{callee}(...) at all, so the claim names a call that "
            "does not exist"
        )
    return violations


def _read_only_call_violations(name: str, keyword: str, callee: str) -> list[str]:
    """Every call to ``callee`` inside one function whose ``keyword=`` is not a literal ``False``.

    The claim is about a call, so the read is bound to the call. Reading the keyword off every
    ``ast.Call`` in the body — the first version of this check — let two shapes through, both
    found by falsifying it rather than by reading it (review 9, N2):

    - the argument dropped from the real call while ``dict(update_access=False)`` sits anywhere
      else in the body: the read still saw a literal ``False`` and the route now writes access
      metadata while the row calls it read-only;
    - the real call kept and a *second* call to the same method added without the keyword. That
      second call defaults to a write, and the read was satisfied by the first.

    So every call to the named callee has to carry the literal, a call carrying nothing is
    reported as such, and a non-literal argument is reported by its own source text instead of
    by a marker, so the failure names the expression that was actually passed.
    """
    return _read_only_violations_in(_function_node(name), keyword, callee, f"_tool_{name}")


def _dispatched_tool_methods() -> set[str]:
    """Every ``self._tool_*`` implementation ``handle_tool_call`` reaches by attribute.

    A branch that reaches its handler through a name the dispatch derivation cannot read
    (``getattr(self, "_tool_" + name)``) still has to define that method, so comparing
    implementations against reached implementations catches the shapes the string read
    misses.
    """
    tree = ast.parse((REPO / "__init__.py").read_text(encoding="utf-8"))
    reached: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "handle_tool_call":
            reached |= {
                sub.attr[len("_tool_"):]
                for sub in ast.walk(node)
                if isinstance(sub, ast.Attribute) and sub.attr.startswith("_tool_")
            }
    return reached


def _registered_names_in(node: ast.AST) -> set[str]:
    """Class names handed to ``register_memory_provider`` inside one body.

    Every call site has to be readable. The previous version knew one shape,
    ``register_memory_provider(Name())``, and skipped the others silently: a registration
    written as ``register_memory_provider(registry.pop())`` produced no entry and no
    failure, which is a check that answers "nothing to see" when it cannot read the page.
    """
    names: set[str] = set()
    for sub in ast.walk(node):
        if not (isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute)):
            continue
        if sub.func.attr != "register_memory_provider":
            continue
        assert sub.args, (
            "register_memory_provider is called with no argument, so the class it registers "
            f"cannot be read: {ast.unparse(sub)}"
        )
        arg = sub.args[0]
        if isinstance(arg, ast.Call) and isinstance(arg.func, ast.Name):
            names.add(arg.func.id)
        elif isinstance(arg, ast.Name):
            names.add(arg.id)
        else:
            raise AssertionError(
                "register_memory_provider is called with a shape this pin cannot read, so "
                "the registered class is unknown: "
                f"{ast.unparse(sub)}"
            )
    return names


def _registered_from(
    tree: ast.Module, class_names: set[str], function_names: set[str] | None = None
) -> set[str]:
    """Class names ``register()`` hands to the host, walking its direct callees once.

    Two shapes the lexical read missed (review 8, N4): a second class registered from a
    module-level function ``register()`` calls (nothing in ``register``'s body mentions it),
    and a local rebinding of a module-level class name (which keeps the name the pin reads
    while changing the class it resolves to). The first is walked one level; the second is
    refused, because a pin that reads a shadowed name cannot say which class it saw.

    Residual, stated: a registration two helpers deep, or reached through a method call
    rather than a module-level function, or performed by the host on our behalf, is still
    outside this read.
    """
    register = next(
        (node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "register"),
        None,
    )
    assert register is not None, "register() is not defined in __init__.py"
    shadowed = sorted(
        target.id
        for node in ast.walk(register)
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign))
        for target in ([node.target] if isinstance(node, ast.AnnAssign) else node.targets)
        if isinstance(target, ast.Name) and target.id in class_names
    )
    assert not shadowed, (
        "register() rebinds a module-level class name, so the pin cannot tell which class is "
        f"registered: {shadowed}"
    )
    registered = _registered_names_in(register)
    if function_names is None:
        function_names = set(_module_functions(tree))
    for callee in sorted(
        node.func.id
        for node in ast.walk(register)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    ):
        if callee in function_names:
            registered |= _registered_names_in(_module_functions(tree)[callee])
    return registered


def _registered_provider_classes() -> set[str]:
    """Every class ``register()`` hands to the host.

    ``register()`` is the seam the hook derivation cannot see: hooks on a *second*
    registered provider class are invisible to ``_provider_hooks``, which reads the one
    class this file names. Pinning the registered set makes a second class loud here
    instead of silent there.
    """
    tree = _module_tree()
    return _registered_from(tree, set(_module_classes(tree)), set(_module_functions(tree)))


def _tool_names() -> set[str]:
    """Every tool name the provider dispatches, from four independent derivations.

    1. the string constants ``handle_tool_call`` compares against;
    2. ``TOOL_SCHEMAS``, the declaration the runtime registers and routes by;
    3. every tool-name-shaped literal anywhere in the module;
    4. every ``_tool_*`` implementation ``handle_tool_call`` reaches, less the pinned
       undispatched set.

    A route present in one and not the others is a failure here instead of an
    unclassified route that both the size pin and the detector accept.

    Residual, stated rather than implied, and reproduced as mutants that survive: a branch
    that answers to a name nothing ever spells out. Two shapes do that — the incoming name
    rewritten before the comparison (``tool_name.replace("wipe_all", "forget")``), and the
    incoming name matched by pattern to an existing handler
    (``tool_name.endswith("wipe_all")``) — and both are invisible to (1), (3) and (4). A
    third shape joins them, named by review 8's N2: a name the module *does* spell out but
    which is reached through a lookup the reads do not model (``{"x": …}.get(name)``, an
    alias) rather than through a comparison, a schema key or a shaped literal. Every such
    branch is unreachable through the host, which routes a tool call only when
    ``memory_manager.has_tool(name)`` is true and builds that mapping from
    ``get_tool_schemas()``: a name absent from ``TOOL_SCHEMAS`` is a direct-Python-caller
    route. That is why (2) is the derivation that has to hold for anything that can ship,
    and why the other three are read as corroboration rather than as the guarantee. It is
    also why closing this residue would not be a test change: the only way to remove it is
    to stop dispatching on a name chain at all.

    Two claims in this function are about *reachability* rather than about names, and both
    are checked against the source instead of described: every entry in
    ``_UNREACHED_TOOL_METHODS`` is asserted to have no call site anywhere in the module, and
    every entry in ``_HELPER_TOOL_METHODS`` is asserted to be reached from the handler of a
    declared route, at the call site it names.
    """
    dispatched = _dispatched_tool_names()
    declared = {str(schema["name"]) for schema in _plugin_module().TOOL_SCHEMAS}
    _assert_declared_names_are_shaped(declared)
    assert dispatched == declared, (
        "the tool dispatch and TOOL_SCHEMAS disagree — one of them names a route the "
        f"other does not: dispatched_only={sorted(dispatched - declared)} "
        f"declared_only={sorted(declared - dispatched)}"
    )
    literals = _literal_tool_names()
    assert literals == declared, (
        "a tool-name-shaped string literal is not a declared tool, or a declared tool is "
        "never spelled out in the module: "
        f"literals_only={sorted(literals - declared)} "
        f"declared_only={sorted(declared - literals)}"
    )
    unwired = _tool_method_names() - _dispatched_tool_methods()
    pinned = set(_UNREACHED_TOOL_METHODS) | set(_HELPER_TOOL_METHODS)
    assert unwired == pinned, (
        "a _tool_* implementation is not reached by handle_tool_call and is not in the pinned "
        f"sets: unexpected={sorted(unwired - pinned)} "
        f"pinned_but_now_wired={sorted(pinned - unwired)}"
    )
    _assert_no_unreached_claim_has_a_call_site()
    _assert_every_helper_claim_holds(declared)
    return dispatched


def _handlers_reaching(route: str) -> set[str]:
    """``self._tool_*`` handlers called by the ``handle_tool_call`` branch that tests ``route``.

    The handler used to be derived from the route's name — ``"_tool_" + route[len(prefix):]`` —
    which is a naming convention asserted as a fact about dispatch. Swapping the two handlers
    inside ``handle_tool_call`` left every assertion in this file green while the row's claim
    "reached from ``qdrant_memory_retrieve``" became false, because the name still matched the
    old function: the row was pinning the convention, not the branch (review 9, N1).

    The branch is the authority, so this reads it: for each ``ast.If`` whose test puts the route
    constant on one side, the handlers called in the branch that runs when the test holds.
    ``orelse`` is read as well, because a dispatch may test the complement (``if name != route``)
    with the handler on the other side. A route whose branch yields nothing, or more than one
    handler, is not silently resolved from its name — the caller has to declare it and say why.
    """
    tree = _module_tree()
    reached: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef) or node.name != "handle_tool_call":
            continue
        for sub in ast.walk(node):
            if not isinstance(sub, ast.If) or route not in _comparison_strings(sub.test):
                continue
            negative = any(
                isinstance(op, (ast.NotEq, ast.NotIn))
                for cmp in ast.walk(sub.test)
                if isinstance(cmp, ast.Compare)
                for op in cmp.ops
            )
            branch = sub.orelse if negative else sub.body
            for inner in ast.walk(ast.Module(body=list(branch), type_ignores=[])):
                if isinstance(inner, ast.Call) and isinstance(inner.func, ast.Attribute):
                    base = inner.func.value
                    if isinstance(base, ast.Name) and base.id == "self":
                        if inner.func.attr.startswith("_tool_"):
                            reached.add(inner.func.attr)
    return reached


def _tool_method_references(tree: ast.Module, name: str) -> list[str]:
    """Every reference to ``_tool_<name>`` in a tree, call or not, any receiver, any form.

    The absence claim used to be a substring read of ``self._tool_{name}(`` in the source. Three
    call sites slipped through it — a call through a local alias (``fn = self._tool_x; fn(args)``),
    a ``getattr(self, "_tool_x")`` call and ``cls._tool_x(`` — while a comment that spelled the
    call out would have failed the check falsely. Written as a function over a tree so the
    shapes can be fed to it directly (review 9, N3).
    """
    hits: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr == f"_tool_{name}":
            hits.append(f"line {node.lineno}: {ast.unparse(node)}")
        elif isinstance(node, ast.Constant) and node.value == f"_tool_{name}":
            hits.append(f"line {node.lineno}: the string {node.value!r}")
    return hits


def _assert_declared_names_are_shaped(declared: set[str]) -> None:
    """Every declared tool name has to match the shape three of the four derivations filter by.

    A declared name outside the shape is invisible to the comparison read, the literal read and
    the method read at once, so it is the one hole the size pin and the completeness detector
    would both accept. The assertion used to live inline in ``_tool_names``, where no test could
    reach it: deleting it left every test in this file green. Factored out so it is called with
    a synthetic name (review 9, N4).
    """
    unshaped = sorted(name for name in declared if not _TOOL_NAME_SHAPE.match(name))
    assert not unshaped, (
        "a declared tool name is outside the shape the literal read filters by, so three of "
        f"the four derivations cannot see it: {unshaped}"
    )


def _assert_no_unreached_claim_has_a_call_site() -> None:
    """A method pinned as unreached must have no call site anywhere in the module.

    This is the check the false claim could not survive. ``retrieve_learning`` was pinned as
    a direct-Python-caller route while ``_tool_retrieve`` called it; the entry is now a
    helper with its reaching route named, and this assertion is why that entry cannot come
    back without its call site going away first.

    Read from the tree, like the presence claims. A substring read of ``self._tool_x(`` missed
    a call through a local alias (``fn = self._tool_x; fn(args)``), a ``getattr`` call and a
    ``cls._tool_x(`` — all of which are call sites — and it would have failed on a comment that
    spelled the call out. Any reference to the name anywhere in the module counts: call or not,
    any receiver, attribute or string form.
    """
    for name, reason in _UNREACHED_TOOL_METHODS.items():
        hits = _tool_method_references(_module_tree(), name)
        assert not hits, (
            f"{name} is pinned as unreached but the module refers to it "
            f"({sorted(hits)}), so the pinned reason is false ({reason!r}): move it to "
            "_HELPER_TOOL_METHODS with the route that reaches it"
        )


def _assert_every_helper_claim_holds(declared: set[str]) -> None:
    """A helper's reaching route, its call site, and its read-only evidence are all checked.

    The route has to be a declared tool, the handler of that route has to actually call the
    helper, and the handler is read from the dispatch branch rather than derived from the
    route's name — the name is a convention, the branch is the dispatch. If the branch cannot
    be read the entry has to declare the handler and say why, so the convention never comes
    back as a fallback. The helper's own code has to pass the declared keyword as a literal
    ``False`` **to the callee the entry names**: a literal anywhere in the body, or a second
    call to that callee without it, is reported as a violation rather than read as a pass.
    All of it is read from the AST: no docstring sentence can stand in for any of the three.
    """
    for name, spec in _HELPER_TOOL_METHODS.items():
        route = str(spec["reached_from"])
        assert route in declared, (
            f"_tool_{name} is pinned as reached from {route}, which is not a declared tool"
        )
        handlers_reaching = _handlers_reaching(route)
        declared_handler = spec.get("handler")
        if len(handlers_reaching) == 1:
            handler = next(iter(handlers_reaching))
            assert declared_handler in (None, handler), (
                f"{route} is pinned with handler {declared_handler!r}, but the dispatch branch "
                f"that tests it calls {handler!r}"
            )
        else:
            assert declared_handler, (
                f"the handler of {route} cannot be read from the dispatch branch "
                f"({len(handlers_reaching)} candidates: {sorted(handlers_reaching)}); declare it "
                "in the entry's 'handler' key and record why the branch cannot be read, "
                "instead of deriving it from the route's name"
            )
            handler = str(declared_handler)
        assert str(spec["call_site"]) in _self_attribute_calls(handler), (
            f"_tool_{name} is pinned as reached from {route}, but {handler} makes no call to "
            f"{spec['call_site']!r} (calls it makes: {sorted(_self_attribute_calls(handler))})"
        )
        evidence = spec["read_only_evidence"]
        kind = evidence["kind"] if isinstance(evidence, dict) else "unknown"
        if kind != "keyword_false":
            raise AssertionError(
                f"_tool_{name} pins read-only evidence of kind {kind!r}, which no reading in "
                "this file interprets: add the reading before adding the claim, or the claim "
                "is prose with a dict around it"
            )
        keyword = str(evidence["keyword"])
        callee = str(evidence["callee"]) if "callee" in evidence else ""
        assert callee, (
            f"_tool_{name} pins read-only evidence without naming the callee it is about: the "
            "claim is about a call, so the entry has to say which call"
        )
        violations = _read_only_call_violations("_tool_" + name, keyword, callee)
        assert not violations, (
            f"_tool_{name} is pinned as read-only because every call to .{callee}(...) in its "
            f"code passes {keyword}=False, but " + "; ".join(violations)
        )


def _cli_commands() -> set[str]:
    """Every registered CLI command path, e.g. ``backup restore``.

    Loaded by path: ``import cli`` resolves to the host application's module, not this
    plugin's.
    """
    spec = importlib.util.spec_from_file_location("_route_inventory_cli", REPO / "cli.py")
    assert spec and spec.loader
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)

    parser = argparse.ArgumentParser()
    cli.register_cli(parser)

    def walk(current: argparse.ArgumentParser, prefix: str = "") -> set[str]:
        found: set[str] = set()
        for action in current._actions:
            if isinstance(action, argparse._SubParsersAction):
                for name, sub in action.choices.items():
                    path = f"{prefix} {name}".strip()
                    found.add(path)
                    found |= walk(sub, path)
        return found

    commands = walk(parser)
    assert commands, "register_cli produced no subcommands"
    return commands


def _provider_hooks() -> set[str]:
    """Provider hook overrides — the write routes the runtime calls directly.

    ``sync_turn`` reaches the same deterministic id as the store and is enabled by a
    config flag, so it is a write route; it is invisible to both the dispatch parser and
    the CLI parser.

    Derived by identity against the host base class across the whole MRO, not by reading
    ``vars(provider_cls)``: reading the class body saw only methods written in the
    provider itself, so a hook implemented in a plugin-side mixin the provider inherits
    from was invisible to the derivation, to the size pin and to the completeness
    detector all at once.

    Environment, stated rather than implied: the set is derived by identity against the
    *host* base, and ``__init__.py`` falls back to its own stub when
    ``agent.memory_provider`` is not importable — precisely so this repo's tests can run
    without Hermes installed, which is what its CI does. A stub is not a base with no
    overrides; it is the absence of the question, so this pin skips there instead of
    returning an empty set that would silently drop every hook row from the inventory.
    ``test_the_hook_derivation_declares_a_missing_host_base`` pins that branch.

    Residual, stated rather than implied: this reads the class, so a hook installed on
    the *instance* (an attribute assigned in ``__init__``), a hook served by
    ``__getattr__``, and a hook on a second registered provider class are all invisible
    here. The second case is why ``test_the_module_registers_exactly_the_modeled_provider``
    pins the registered set; the first two are direct-installation shapes no class-level
    derivation can see, and ``test_no_inventory_row_is_stale`` fails loudly if a row
    names a hook the derivation no longer produces.
    """
    module = _plugin_module()
    provider_cls = module.QdrantMemoryProvider
    base = module.MemoryProvider
    if not _host_provider_base_is_importable():
        pytest.skip(
            "the host's agent.memory_provider.MemoryProvider is not importable here, so this "
            "pin cannot answer its question: the hook set is derived by identity against the "
            "host base, and __init__.py's standalone fallback stub has no overridable public "
            "API, so read as a surface it would drop every hook row out of the inventory. The "
            "repo's CI runs the tests without Hermes installed; this pin runs where it is."
        )
    hooks: set[str] = set()
    for name in dir(base):
        if name.startswith("_"):
            continue
        base_attr = getattr(base, name)
        if not callable(base_attr):
            continue
        if getattr(provider_cls, name, None) is not base_attr:
            hooks.add(name)
    assert hooks, (
        "the host base is importable but no public callable of it is overridden by the "
        "provider, so the hook derivation matched nothing: either the provider stopped "
        "overriding the host API or the identity comparison broke"
    )
    return hooks


def _host_provider_base_is_importable() -> bool:
    """Is ``__init__.MemoryProvider`` the host's real base, or the standalone fallback stub?"""
    try:
        from agent.memory_provider import MemoryProvider as host_base
    except Exception:  # pragma: no cover - the branch CI takes
        return False
    return _plugin_module().MemoryProvider is host_base


def _surface() -> set[str]:
    return _tool_names() | _cli_commands() | _provider_hooks()


# --- the inventory ---------------------------------------------------------
#
# kind:
#   protection  a route that can delete or replace an existing payload; it must refuse
#               the protected shapes and its refusal must be pinned by a test that
#               exists in this tree
#   transition  the reviewed lineage path, which is allowed to write
#   read_only   never deletes or replaces a payload; may read, may set access metadata
#               on the points it returns, and may write local artifacts
#   dispatch    a router; every route it can reach is classified separately
#   uncovered   mutates, and its refusal is reasoned from code but not executed

FORGET_PINS = [
    "tests/test_consolidation_apply.py::test_structural_lineage_refusal_wording_is_caller_agnostic",
    "tests/test_consolidation_apply.py::test_a_demoted_dependent_stays_retirable",
    "tests/test_consolidation_apply.py::test_forget_refuses_any_dependent_and_deletes_all_clear_targets",
    "tests/test_consolidation_apply.py::test_forget_lock_contention_refuses_without_any_write_and_respects_timeout",
]
STORE_PINS = [
    "tests/test_consolidation_apply.py::test_store_refuses_to_overwrite_a_protected_point_in_every_mode",
    "tests/test_consolidation_apply.py::test_store_refuses_a_history_only_dependent_in_every_mode",
    "tests/test_consolidation_apply.py::test_store_fails_closed_when_the_target_cannot_be_read",
    "tests/test_consolidation_apply.py::test_store_answers_the_retirement_text_for_a_point_that_carries_both",
]
EXTRACTION_PINS = [
    "tests/test_consolidation_apply.py::test_extraction_approval_refuses_to_overwrite_a_protected_point",
    "tests/test_consolidation_apply.py::test_extraction_approval_refuses_a_history_only_dependent",
    "tests/test_consolidation_apply.py::test_extraction_approval_fails_closed_when_the_target_cannot_be_read",
]
RESTORE_PINS = [
    "tests/test_backup_cli.py::test_restore_refuses_to_overwrite_every_protected_payload",
    "tests/test_backup_cli.py::test_restore_refuses_a_protected_target_in_the_learnings_scope",
    "tests/test_backup_cli.py::test_restore_refuses_the_review_only_shape",
    "tests/test_backup_cli.py::test_restore_still_overwrites_an_ordinary_payload",
    "tests/test_backup_cli.py::test_restore_refuses_missing_chunk_retired_by_committed_event",
]
INDEX_OFF_PINS = [
    "tests/test_lineage.py::test_off_mode_refuses_to_rewrite_a_chunk_carrying_only_review_state",
    "tests/test_lineage.py::test_off_mode_refuses_to_rewrite_a_chunk_carrying_one_identity_field",
    "tests/test_lineage.py::test_off_mode_refuses_to_destroy_or_duplicate_captured_file",
    "tests/test_lineage.py::test_directory_off_mode_does_not_delete_removed_lineage_managed_chunks",
    "tests/test_lineage.py::test_a_removed_file_whose_siblings_are_ordinary_is_blocked_as_a_file",
    "tests/test_lineage.py::test_a_removed_file_with_one_protected_sibling_survives_force_too",
    "tests/test_lineage.py::test_a_removed_file_whose_only_protected_chunk_is_foreign_scope_is_blocked",
    "tests/test_lineage.py::test_a_present_file_whose_only_protected_chunk_is_foreign_scope_is_blocked_too",
    "tests/test_lineage.py::test_off_mode_force_skips_lineage_managed_filter_delete_in_dry_and_live_runs",
]
CONSOLIDATION_PINS = [
    "tests/test_consolidation_apply.py::test_destructive_consolidation_refuses_lineage_dependents_outside_reconcile",
]
# The learning store has no payload predicate: its safety is the collection-separation
# invariant, because an upsert there would otherwise be able to replace a memory point.
LEARNING_STORE_PINS = [
    "tests/test_config_schema_scoring.py::test_config_rejects_one_collection_for_memories_and_learnings",
]
# `watcher run` only reaches a mutation through `--autonomy-mode guarded-auto`, which
# calls `apply_guarded_auto` → the apply path, so its refusal is pinned through that
# caller rather than through the tool it eventually invokes.
GUARDED_AUTO_PINS = [
    "tests/test_consolidation_apply.py::test_watcher_guarded_auto_refuses_a_lineage_managed_target",
]
SYNC_TURN_PINS = [
    "tests/test_consolidation_apply.py::test_the_sync_turn_hook_cannot_overwrite_a_protected_target",
]

ROUTE_INVENTORY: dict[str, dict[str, object]] = {}


def _declare(names, kind, *, pins=None, note="", reason=""):
    for name in names:
        ROUTE_INVENTORY[name] = {"kind": kind, "pins": list(pins or []), "note": note, "reason": reason}


_declare(
    [
        "qdrant_memory_store", "store",
    ], "protection", pins=STORE_PINS,
    note="replaces the payload at a deterministic content-derived id",
)
_declare(["qdrant_memory_forget", "forget"], "protection", pins=FORGET_PINS)
_declare(
    ["qdrant_memory_extraction_approve", "learning approve"], "protection", pins=EXTRACTION_PINS,
    note="upserts a regenerated candidate id",
)
_declare(["restore"], "protection", pins=RESTORE_PINS, note="the third in-place overwrite route")
_declare(["qdrant_memory_index [off]", "index [off]"], "protection", pins=INDEX_OFF_PINS,
         note="rewrites chunk payloads at content-derived ids and deletes stale ids, per file")
_declare(["qdrant_memory_consolidation_apply", "apply"], "protection", pins=CONSOLIDATION_PINS,
         note="merge/delete/quarantine; the managed check runs before the lock (N5)")
_declare(
    ["qdrant_learning_store", "learning store", "qdrant_learning_approve"], "protection",
    pins=LEARNING_STORE_PINS,
    note="no payload predicate: protected by the collection-separation invariant only",
)
_declare(["watcher run"], "protection", pins=GUARDED_AUTO_PINS,
         note="report-only by default; --guarded-auto reaches the apply path through apply_guarded_auto")
_declare(["qdrant_memory_index [capture]", "index [capture]"], "transition")
_declare(["qdrant_memory_index [reconcile]", "index [reconcile]"], "transition")

# --- provider hooks --------------------------------------------------------
# The runtime calls these directly, so none of them is reachable from the parsers above.

_declare(["sync_turn"], "protection", pins=SYNC_TURN_PINS,
         note="writes the turn through ConversationWriter at the store's deterministic id")
_declare(["handle_tool_call", "get_tool_schemas"], "dispatch",
         note="the router itself; every route it can reach is classified in this file")
_declare(
    [
        "initialize", "shutdown", "is_available", "system_prompt_block",
        "prefetch", "queue_prefetch", "on_session_switch",
    ], "read_only",
    note="config, client construction, in-process cache and session bookkeeping; no payload write",
)
_declare(["on_pre_compress", "on_session_end"], "read_only",
         note="collects extraction candidates into local artifacts; the store modes that would write are inert")

_declare(["qdrant_memory_improve_apply", "improve apply"], "uncovered",
         reason="predicate read on 567eeac; stricter than the shared one on the traced shapes, never executed")
_declare(["qdrant_memory_raptor_apply"], "uncovered",
         reason="same as improve apply: read, not executed")

_declare([
    # provider reads
    "qdrant_memory_status", "qdrant_memory_search", "qdrant_memory_graph_search",
    "qdrant_memory_context", "qdrant_memory_retrieve", "qdrant_memory_inspect",
    "qdrant_memory_trace", "qdrant_memory_expand", "qdrant_memory_source_status",
    "qdrant_memory_consolidate", "qdrant_memory_memory_pr", "qdrant_memory_raptor_status",
    "qdrant_memory_improve_preview", "qdrant_memory_extraction_preview",
    "qdrant_learning_preview", "qdrant_learning_search",
    # cli reads
    "status", "doctor", "config", "config show", "search", "graph-search", "retrieve",
    "context", "show", "inspect", "trace", "expand", "source-status",
    "export", "export memory", "export learning", "reports", "reports list",
    "reports show", "proposals", "proposals show", "eval", "eval-capture", "eval-gate",
    "learning", "learning search", "learning preview", "consolidate",
    "improve", "improve preview", "backup", "backup list", "backup inspect",
    "watcher", "watcher status", "watcher logs", "watcher inspect-state",
    "watcher install", "watcher uninstall", "watcher reset-signature",
], "read_only",
    note="searches set access metadata on the points they return (set/merge, never delete or replace); "
         "watcher status/install and the report commands write local artifacts only")

# Writes outside Qdrant (files, reports, proposal state) with no payload to protect.
_declare(["backup create", "eval-capture"], "read_only",
         note="writes files only; the restore route that consumes them is the protection row")


# --- frozen classification -------------------------------------------------
# A hand-written classification that nothing defends is a claim surface. These maps are
# the contract: a row reclassified away from `protection`, or a pin list trimmed, fails
# here even though every individual route still looks fine.
#
# The pin lists in EXPECTED_PROTECTION_ROUTES are written out as literals on purpose.
# Referencing the constants above would compare each list with itself, so trimming a
# constant (or a row's `pins=`) would move both sides together and the frozen map would
# survive its own mutation. That is the defect this file exists to catch, one level up:
# a check that reads the same source twice is not a check. Because these are duplicates,
# adding a pin means updating both the constant and this literal.

# The derived surface — 25 tools, 50 CLI commands, 12 provider hooks. A route added or
# removed changes this number, which forces someone to touch this file and classify it,
# even if the completeness check in the test body is later weakened.
FROZEN_SURFACE_SIZE = 87

# Every `[mode]` row, frozen by name and by kind. Completeness used to ask only whether
# *some* mode row existed for the route, so deleting both `[capture]` rows, deleting both
# `[reconcile]` rows, or reclassifying `[capture]` as `read_only` left all ten tests green
# while the route table said nothing about the mode. A mode row is a claim about a mode,
# so the claim is what gets frozen.
EXPECTED_MODE_ROWS = {
    "qdrant_memory_index [off]": "protection",
    "index [off]": "protection",
    "qdrant_memory_index [capture]": "transition",
    "index [capture]": "transition",
    "qdrant_memory_index [reconcile]": "transition",
    "index [reconcile]": "transition",
}

EXPECTED_PROTECTION_ROUTES = {
    "qdrant_memory_store": (
        "tests/test_consolidation_apply.py::test_store_refuses_to_overwrite_a_protected_point_in_every_mode",
        "tests/test_consolidation_apply.py::test_store_refuses_a_history_only_dependent_in_every_mode",
        "tests/test_consolidation_apply.py::test_store_fails_closed_when_the_target_cannot_be_read",
        "tests/test_consolidation_apply.py::test_store_answers_the_retirement_text_for_a_point_that_carries_both",
    ),
    "store": (
        "tests/test_consolidation_apply.py::test_store_refuses_to_overwrite_a_protected_point_in_every_mode",
        "tests/test_consolidation_apply.py::test_store_refuses_a_history_only_dependent_in_every_mode",
        "tests/test_consolidation_apply.py::test_store_fails_closed_when_the_target_cannot_be_read",
        "tests/test_consolidation_apply.py::test_store_answers_the_retirement_text_for_a_point_that_carries_both",
    ),
    "qdrant_memory_forget": (
        "tests/test_consolidation_apply.py::test_structural_lineage_refusal_wording_is_caller_agnostic",
        "tests/test_consolidation_apply.py::test_a_demoted_dependent_stays_retirable",
        "tests/test_consolidation_apply.py::test_forget_refuses_any_dependent_and_deletes_all_clear_targets",
        "tests/test_consolidation_apply.py::test_forget_lock_contention_refuses_without_any_write_and_respects_timeout",
    ),
    "forget": (
        "tests/test_consolidation_apply.py::test_structural_lineage_refusal_wording_is_caller_agnostic",
        "tests/test_consolidation_apply.py::test_a_demoted_dependent_stays_retirable",
        "tests/test_consolidation_apply.py::test_forget_refuses_any_dependent_and_deletes_all_clear_targets",
        "tests/test_consolidation_apply.py::test_forget_lock_contention_refuses_without_any_write_and_respects_timeout",
    ),
    "qdrant_memory_extraction_approve": (
        "tests/test_consolidation_apply.py::test_extraction_approval_refuses_to_overwrite_a_protected_point",
        "tests/test_consolidation_apply.py::test_extraction_approval_refuses_a_history_only_dependent",
        "tests/test_consolidation_apply.py::test_extraction_approval_fails_closed_when_the_target_cannot_be_read",
    ),
    "learning approve": (
        "tests/test_consolidation_apply.py::test_extraction_approval_refuses_to_overwrite_a_protected_point",
        "tests/test_consolidation_apply.py::test_extraction_approval_refuses_a_history_only_dependent",
        "tests/test_consolidation_apply.py::test_extraction_approval_fails_closed_when_the_target_cannot_be_read",
    ),
    "restore": (
        "tests/test_backup_cli.py::test_restore_refuses_to_overwrite_every_protected_payload",
        "tests/test_backup_cli.py::test_restore_refuses_a_protected_target_in_the_learnings_scope",
        "tests/test_backup_cli.py::test_restore_refuses_the_review_only_shape",
        "tests/test_backup_cli.py::test_restore_still_overwrites_an_ordinary_payload",
        "tests/test_backup_cli.py::test_restore_refuses_missing_chunk_retired_by_committed_event",
    ),
    "qdrant_memory_index [off]": (
        "tests/test_lineage.py::test_off_mode_refuses_to_rewrite_a_chunk_carrying_only_review_state",
        "tests/test_lineage.py::test_off_mode_refuses_to_rewrite_a_chunk_carrying_one_identity_field",
        "tests/test_lineage.py::test_off_mode_refuses_to_destroy_or_duplicate_captured_file",
        "tests/test_lineage.py::test_directory_off_mode_does_not_delete_removed_lineage_managed_chunks",
        "tests/test_lineage.py::test_a_removed_file_whose_siblings_are_ordinary_is_blocked_as_a_file",
        "tests/test_lineage.py::test_a_removed_file_with_one_protected_sibling_survives_force_too",
        "tests/test_lineage.py::test_a_removed_file_whose_only_protected_chunk_is_foreign_scope_is_blocked",
        "tests/test_lineage.py::test_a_present_file_whose_only_protected_chunk_is_foreign_scope_is_blocked_too",
        "tests/test_lineage.py::test_off_mode_force_skips_lineage_managed_filter_delete_in_dry_and_live_runs",
    ),
    "index [off]": (
        "tests/test_lineage.py::test_off_mode_refuses_to_rewrite_a_chunk_carrying_only_review_state",
        "tests/test_lineage.py::test_off_mode_refuses_to_rewrite_a_chunk_carrying_one_identity_field",
        "tests/test_lineage.py::test_off_mode_refuses_to_destroy_or_duplicate_captured_file",
        "tests/test_lineage.py::test_directory_off_mode_does_not_delete_removed_lineage_managed_chunks",
        "tests/test_lineage.py::test_a_removed_file_whose_siblings_are_ordinary_is_blocked_as_a_file",
        "tests/test_lineage.py::test_a_removed_file_with_one_protected_sibling_survives_force_too",
        "tests/test_lineage.py::test_a_removed_file_whose_only_protected_chunk_is_foreign_scope_is_blocked",
        "tests/test_lineage.py::test_a_present_file_whose_only_protected_chunk_is_foreign_scope_is_blocked_too",
        "tests/test_lineage.py::test_off_mode_force_skips_lineage_managed_filter_delete_in_dry_and_live_runs",
    ),
    "qdrant_memory_consolidation_apply": (
        "tests/test_consolidation_apply.py::test_destructive_consolidation_refuses_lineage_dependents_outside_reconcile",
    ),
    "apply": (
        "tests/test_consolidation_apply.py::test_destructive_consolidation_refuses_lineage_dependents_outside_reconcile",
    ),
    "qdrant_learning_store": (
        "tests/test_config_schema_scoring.py::test_config_rejects_one_collection_for_memories_and_learnings",
    ),
    "learning store": (
        "tests/test_config_schema_scoring.py::test_config_rejects_one_collection_for_memories_and_learnings",
    ),
    "qdrant_learning_approve": (
        "tests/test_config_schema_scoring.py::test_config_rejects_one_collection_for_memories_and_learnings",
    ),
    "watcher run": (
        "tests/test_consolidation_apply.py::test_watcher_guarded_auto_refuses_a_lineage_managed_target",
    ),
    "sync_turn": (
        "tests/test_consolidation_apply.py::test_the_sync_turn_hook_cannot_overwrite_a_protected_target",
    ),
}


# --- assertions ------------------------------------------------------------


# The `[mode]` suffix exists for the routes whose behaviour depends on the lineage mode,
# and for those only. A `forget [off]` row would otherwise classify the base `forget`
# route while describing a mode it does not have, which is the same claim-surface hole
# one level up: the row exists, the label is wrong, and completeness says fine.
_MODE_SCOPED = ("qdrant_memory_index", "index")
_MODES = ("off", "capture", "reconcile")


def _base(entry: str) -> str:
    return entry.split(" [", 1)[0]


def _mode_rows(name: str, inventory) -> set[str]:
    return {mode for mode in _MODES if f"{name} [{mode}]" in inventory}


def _classified_by(name: str, inventory) -> bool:
    """A mode-scoped route is classified only when **every** mode has a row.

    ``any`` was the same hole in the detector that ``EXPECTED_MODE_ROWS`` closes in the
    classification: one surviving mode row kept the route looking classified while the
    other modes had been dropped.
    """
    if name in inventory:
        return True
    if name not in _MODE_SCOPED:
        return False
    return _mode_rows(name, inventory) == set(_MODES)


def _unclassified(derived: set[str], inventory=None) -> list[str]:
    inventory = ROUTE_INVENTORY if inventory is None else inventory
    return sorted(name for name in derived if not _classified_by(name, inventory))


def _unresolved_pins(inventory=None, repo: Path | None = None) -> list[str]:
    inventory = ROUTE_INVENTORY if inventory is None else inventory
    repo = REPO if repo is None else repo
    unresolved: list[str] = []
    for entry, row in sorted(inventory.items()):
        if row["kind"] != "protection":
            continue
        pins = list(row["pins"])  # type: ignore[arg-type]
        if not pins:
            # A protection row with no pin is the claim-surface hole this file exists to
            # close, and it resolves silently: there is no node id to fail to find.
            unresolved.append(f"{entry} -> a protection row must name at least one pin")
            continue
        for node in pins:
            path, _, name = node.partition("::")
            name = name.split("[", 1)[0]
            target = repo / path
            if not target.is_file() or not re.search(
                rf"^def {re.escape(name)}\(", target.read_text(encoding="utf-8"), re.M
            ):
                unresolved.append(f"{entry} -> {node}")
    return unresolved


def test_every_surface_entry_is_classified():
    """A new tool, command or hook must be classified before it can ship.

    The size pin below is deliberate: it makes a new route change a number that has to
    be updated by hand, so the gate keeps working even if someone later guts the
    ``missing`` computation in this body. The detector functions themselves are tested
    separately, against deliberately unclassified input.
    """
    derived = _surface()
    assert len(derived) == FROZEN_SURFACE_SIZE, (
        f"the derived surface changed from {FROZEN_SURFACE_SIZE} to {len(derived)} entries. "
        "If a route was added or removed on purpose, classify it in ROUTE_INVENTORY and "
        "update this pin; if not, find out why the derivation sees something new."
    )
    missing = _unclassified(derived)
    assert missing == [], (
        "these entry points are not classified in ROUTE_INVENTORY: "
        f"{missing}. Decide what each one does to a protected payload: protection "
        "(with pins), transition, read_only, dispatch, or uncovered with a reason."
    )


def test_the_completeness_detector_rejects_an_unclassified_entry():
    """The detector must fail when it should: this is what makes the row above a gate."""
    assert _unclassified({"qdrant_memory_newroute"}, ROUTE_INVENTORY) == ["qdrant_memory_newroute"]
    assert _unclassified({"qdrant_memory_store"}, ROUTE_INVENTORY) == []
    assert _unclassified({"index"}, ROUTE_INVENTORY) == []


def test_a_mode_suffix_only_classifies_a_mode_scoped_route():
    """`forget [off]` must not stand in for `forget`."""
    suffix_only = {"forget [off]": {"kind": "read_only", "pins": [], "note": "", "reason": ""}}
    assert _unclassified({"forget"}, suffix_only) == ["forget"]
    index_suffix = {
        key: {"kind": "protection" if "[" in key else "read_only", "pins": [], "note": "", "reason": ""}
        for key in ("index [off]", "index [capture]", "index [reconcile]", "forget")
    }
    assert _unclassified({"index", "forget"}, index_suffix) == []


def test_no_inventory_row_is_stale():
    derived = _surface()
    stale = sorted(
        key for key in ROUTE_INVENTORY
        if _base(key) not in derived
        or (
            " [" in key
            and (
                _base(key) not in _MODE_SCOPED
                # A mode label that is not a mode: `index [Off]` classified the base
                # route before this check, because only the base half of the row was
                # ever validated.
                or key.rsplit(" [", 1)[1].rstrip("]") not in _MODES
            )
        )
    )
    assert stale == [], f"ROUTE_INVENTORY rows that no longer name a real entry point: {stale}"


def test_every_protection_route_names_a_pin_that_exists():
    """The route table may not claim a pin that is not there.

    ``docs/SAFETY.md`` said "pinned" three times for rows no test covered. A claim is
    only as good as its resolvable node id, so this resolves them.
    """
    unresolved = _unresolved_pins()
    assert unresolved == [], f"protection routes naming a pin that does not exist: {unresolved}"


def test_the_pin_resolver_rejects_a_pin_that_does_not_exist():
    """The resolver must fail when it should, including on a renamed node."""
    broken = {"route": {"kind": "protection", "pins": ["tests/test_lineage.py::test_not_a_real_node"], "note": "", "reason": ""}}
    assert _unresolved_pins(broken) == ["route -> tests/test_lineage.py::test_not_a_real_node"]
    renamed = {"route": {"kind": "protection", "pins": ["tests/test_lineage.py::test_off_mode_refuses_to_rewrite_a_chunk_carrying_only_review_state"], "note": "", "reason": ""}}
    assert _unresolved_pins(renamed) == []


def test_every_uncovered_route_declares_a_reason():
    missing = sorted(
        entry for entry, row in ROUTE_INVENTORY.items()
        if row["kind"] == "uncovered" and not str(row["reason"]).strip()
    )
    assert missing == [], f"uncovered routes without a reason: {missing}"


def test_a_route_is_classified_once_with_one_kind():
    """One route, one kind — and a mode-scoped route is not classified twice.

    The docstring said "once" while the body only checked that the kind was known. A base
    row `index` declared `read_only` beside the `index [off]` protection row passed: the
    same route answering twice, with the answer a reader finds first being the reassuring
    one, and the mode rows outside the reach of the frozen protection set.
    """
    kinds = {"protection", "transition", "read_only", "dispatch", "uncovered"}
    bad = {entry: row["kind"] for entry, row in ROUTE_INVENTORY.items() if row["kind"] not in kinds}
    assert bad == {}, f"unknown kinds: {bad}"

    doubled = sorted(
        name for name in _MODE_SCOPED
        if name in ROUTE_INVENTORY and any(key.startswith(f"{name} [") for key in ROUTE_INVENTORY)
    )
    assert doubled == [], (
        f"these mode-scoped routes are classified twice, as a base row and per mode: {doubled}. "
        "A base row here is a second answer to the question the mode rows answer."
    )


def test_every_mode_scoped_route_classifies_all_three_modes():
    """Frozen: dropping both `[capture]` rows, or reclassifying one, fails here.

    The two modes other than `off` carry the transition the closure depends on: `capture`
    records and `reconcile` retires. A route table that names the mode in prose while the
    inventory has no row for it is the claim surface this file exists to remove.
    """
    rows = {
        entry: row["kind"]
        for entry, row in ROUTE_INVENTORY.items()
        if " [" in entry and _base(entry) in _MODE_SCOPED
    }
    assert rows == EXPECTED_MODE_ROWS


def test_an_unshaped_declared_name_fails_loudly():
    """The shape assertion is called with a synthetic name instead of trusted to be in place.

    It used to live inline in ``_tool_names``: deleting it left every test in this file green,
    so the "unshaped names fail loudly" disposition proved the regex admits digits and nothing
    about the derivation's behaviour on a name outside the shape (review 9, N4).
    """
    _assert_declared_names_are_shaped({"qdrant_memory_retrieve_v2", "qdrant_learning_search2"})
    for unshaped in ("qdrant_memory_Wipe", "qdrant-memory-wipe", "forget", "qdrant_memory_"):
        try:
            _assert_declared_names_are_shaped({unshaped})
        except AssertionError as exc:
            assert unshaped in str(exc), str(exc)
        else:
            raise AssertionError(f"a declared name outside the shape was accepted: {unshaped!r}")
    here = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    called = {
        str(node.func.id)
        for fn in ast.walk(here)
        if isinstance(fn, ast.FunctionDef) and fn.name == "_tool_names"
        for node in ast.walk(fn)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "_assert_declared_names_are_shaped" in called, (
        "the shape guard is defined but no longer called by the derivation that needs it, so "
        "an unshaped declared name would be invisible to three of the four reads"
    )


def test_the_read_only_claim_cannot_be_satisfied_by_the_docstring():
    """The first version of this check was satisfied by a sentence. That has to stay a failure.

    Both directions are fed to the read directly, because the shape that has to be excluded —
    a docstring asserting the contract while the code omits it — is exactly the shape a
    mutation of the *plugin* cannot express without also changing behaviour.
    """
    prose_only = ast.parse(
        'def f(self):\n'
        '    """Read-only: the code passes update_access=False to search."""\n'
        '    return store.search(query, top_k=1)\n'
    )
    assert _read_only_violations_in(prose_only, "update_access", "search", "f"), (
        "a docstring sentence satisfied a claim about the code"
    )
    code_only = ast.parse(
        'def f(self):\n'
        '    """The code passes update_access=True and writes access metadata."""\n'
        '    return store.search(query, update_access=False)\n'
    )
    assert not _read_only_violations_in(code_only, "update_access", "search", "f"), (
        "the literal argument was not read from the call"
    )
    absent = ast.parse("def f(self):\n    return store.other(query)\n")
    assert _read_only_violations_in(absent, "update_access", "search", "f"), (
        "a claim about a call that does not exist was read as satisfied"
    )


def test_the_unreached_claim_is_read_from_the_tree():
    """The absence claim has to see an alias, a ``getattr`` and a ``cls.`` call site.

    A substring read of ``self._tool_x(`` sees none of them, and it fails on the comment
    spelling the call out — the two errors are the same error: it reads the rendering instead
    of the tree (review 9, N3).
    """
    tree = ast.parse(
        "def f(self):\n"
        "    # self._tool_orphan(args) would be a call site, but this line is a comment\n"
        "    fn = self._tool_orphan\n"
        "    fn(args)\n"
        "    getattr(self, \"_tool_orphan\")(args)\n"
        "    cls._tool_orphan(args)\n"
    )
    hits = _tool_method_references(tree, "orphan")
    assert len(hits) >= 3, hits
    assert all("orphan" in hit for hit in hits), hits
    assert _tool_method_references(ast.parse("def f(self):\n    return self._tool_keep()\n"), "orphan") == []


def test_the_tool_name_shape_covers_names_the_host_can_register():
    """The shape is a coverage claim about three of the four derivations.

    A name with a digit in it was invisible to the comparison read, the literal read and the
    method read at once; only ``TOOL_SCHEMAS`` would have caught it. Freezing the shape
    against a synthetic name makes narrowing it back a failure here instead of a silence
    there.
    """
    assert _TOOL_NAME_SHAPE.match("qdrant_memory_retrieve_v2")
    assert _TOOL_NAME_SHAPE.match("qdrant_learning_search2")
    assert not _TOOL_NAME_SHAPE.match("forget")
    assert not _TOOL_NAME_SHAPE.match("qdrant_memory_")


def test_a_helper_method_is_reached_through_the_route_it_names():
    """The reachability claim, checked instead of asserted in a reason string.

    ``_tool_retrieve_learning`` is invisible to a read that asks what ``handle_tool_call``
    calls — it is called one level down, inside the handler the route maps to. The pinned
    route, the pinned call site, the read-only evidence and that blindness are all checked
    here, so the entry cannot survive its own call site being removed, and cannot be parked
    back in the unreached set while something calls it.
    """
    assert _HELPER_TOOL_METHODS, "the helper set is empty: this test would pass vacuously"
    _assert_every_helper_claim_holds({str(s["name"]) for s in _plugin_module().TOOL_SCHEMAS})
    _assert_no_unreached_claim_has_a_call_site()
    for name in _HELPER_TOOL_METHODS:
        assert f"self._tool_{name}" not in _self_attribute_calls("handle_tool_call"), (
            f"_tool_{name} is pinned as reached one level down, but handle_tool_call calls "
            "it directly, so the entry describes a shape that no longer exists"
        )


def test_the_tool_method_set_is_read_over_the_whole_mro():
    """Every ``_tool_*`` in the module lives on the provider or on a class it inherits.

    The set used to be read from one ``ClassDef`` body. A method on a mixin is called
    through the provider, so the closure is the set that has to be complete — and a
    ``_tool_*`` outside it is loud rather than absent.
    """
    tree = _module_tree()
    closure = _provider_mro_closure(tree)
    assert "QdrantMemoryProvider" in closure
    assert set(_tool_methods_by_class(tree)) <= closure
    assert _tool_method_names() == {
        method for name in closure for method in _tool_methods_by_class(tree).get(name, set())
    }


def test_the_tool_method_set_refuses_a_mixin_outside_the_mro():
    """A ``_tool_*`` on a class the provider does not inherit from is a failure, not a gap."""
    tree = ast.parse(
        "class Lone:\n"
        "    def _tool_orphan(self):\n"
        "        pass\n"
        "\n"
        "class QdrantMemoryProvider:\n"
        "    def _tool_keep(self):\n"
        "        pass\n"
    )
    try:
        _tool_method_names(tree)
    except AssertionError as exc:
        assert "Lone" in str(exc) and "orphan" in str(exc), str(exc)
    else:
        raise AssertionError("a _tool_* outside the provider MRO was accepted")


def test_the_hook_derivation_declares_a_missing_host_base(monkeypatch):
    """A stub base is the absence of the question, not a base with no overrides.

    Stands in for the environment the repo's CI runs in: no Hermes, so
    ``agent.memory_provider`` is not importable and ``__init__.py`` supplies its own base.
    The pin has to say the question cannot be asked there — not fail the suite, and not
    return an empty hook set, which would quietly drop every hook row from the inventory.
    """
    module = _plugin_module()

    class StandaloneStub:
        """Stands in for __init__.py's fallback base class."""

    monkeypatch.setattr(module, "MemoryProvider", StandaloneStub)
    # ``_plugin_module`` re-executes __init__.py from disk on every call, so patching the
    # module object it already returned would be invisible to the read under test.
    monkeypatch.setattr(sys.modules[__name__], "_plugin_module", lambda: module)
    with pytest.raises(pytest.skip.Exception):
        _provider_hooks()


def test_the_ast_tool_method_set_matches_the_class_the_host_loads():
    """The AST closure is corroborated against the live class, not only against itself.

    ``_provider_mro_closure`` follows ``ast.Name`` bases that resolve to a module-level
    ``ClassDef``. A ``_tool_*`` on a base imported from another module is therefore not in the
    module tree, not in the closure and not reported: silent rather than loud (review 9, N5).
    ``MemoryProvider``, the host base, is exactly such a class. Comparing the AST set with the
    live MRO turns that silence into a failure the moment it stops being hypothetical.
    """
    module = _plugin_module()
    live = {
        method
        for klass in module.QdrantMemoryProvider.__mro__
        for method in vars(klass)
        if method.startswith("_tool_")
    }
    ast_set = {f"_tool_{name}" for name in _tool_method_names()}
    assert live == ast_set, (
        "the live class and the AST read disagree about which _tool_* methods exist: "
        f"live_only={sorted(live - ast_set)} ast_only={sorted(ast_set - live)}"
    )


def test_the_register_pin_walks_a_helper_called_by_register():
    """A second class registered from a helper is read, not skipped.

    ``register()`` calling ``_register_extra()`` which registers a second provider class left
    the pinned set at one name while two classes were registered, and the hook derivation
    reads one class. The walk is one level deep, which is the level this host uses.
    """
    tree = ast.parse(
        "class QdrantMemoryProvider:\n    pass\n"
        "class SecondProvider:\n    pass\n"
        "def _register_extra():\n"
        "    host.register_memory_provider(SecondProvider())\n"
        "def register():\n"
        "    host.register_memory_provider(QdrantMemoryProvider())\n"
        "    _register_extra()\n"
    )
    assert _registered_from(tree, {"QdrantMemoryProvider", "SecondProvider"}) == {
        "QdrantMemoryProvider",
        "SecondProvider",
    }


def test_the_register_pin_refuses_a_shape_it_cannot_read():
    """An unreadable registration call is a failure, not an empty set.

    ``register_memory_provider(registry.pop())`` used to produce no entry and no error: the
    check answered "nothing to see" about a page it could not read.
    """
    tree = ast.parse(
        "def register():\n"
        "    host.register_memory_provider(registry.pop())\n"
    )
    try:
        _registered_from(tree, set())
    except AssertionError as exc:
        assert "cannot read" in str(exc), str(exc)
    else:
        raise AssertionError("an unreadable registration shape was accepted")


def test_the_register_pin_refuses_a_local_rebinding():
    """A rebinding of a module-level class name inside ``register()`` cannot be read."""
    tree = ast.parse(
        "class QdrantMemoryProvider:\n    pass\n"
        "def register():\n"
        "    QdrantMemoryProvider = Other\n"
        "    host.register_memory_provider(QdrantMemoryProvider())\n"
    )
    try:
        _registered_from(tree, {"QdrantMemoryProvider"})
    except AssertionError as exc:
        assert "rebinds" in str(exc), str(exc)
    else:
        raise AssertionError("a shadowed class name was accepted as a registration")


def test_the_module_registers_exactly_the_modeled_provider():
    """One provider class, so the hook derivation reads the class the runtime registers.

    ``register()`` is the seam ``_provider_hooks`` cannot see: a hook on a second
    registered class is invisible to a derivation that names one class. Pinning the
    registered set makes that loud here.
    """
    assert _registered_provider_classes() == {"QdrantMemoryProvider"}


def test_the_protection_set_is_exactly_this():
    """Frozen: reclassifying any of these away from `protection` fails here.

    Covers the four routes four reviews opened (store, extraction approval, restore, the
    off-mode reindex), the two the plan opened (forget, destructive consolidation), the
    learning store's collection invariant, the guarded-auto caller, and the hook writer.
    """
    protected = {entry: tuple(row["pins"]) for entry, row in ROUTE_INVENTORY.items() if row["kind"] == "protection"}
    assert protected == EXPECTED_PROTECTION_ROUTES


def test_every_protection_row_keeps_its_exact_pin_list():
    """Frozen: trimming a pin list to one entry fails here.

    A list that is only checked for "some pins exist and they resolve" survives
    mutation; the point of the list is that every claim in it is enumerated.
    """
    trimmed = {
        entry: row["pins"] for entry, row in ROUTE_INVENTORY.items()
        if row["kind"] == "protection" and list(row["pins"]) != list(EXPECTED_PROTECTION_ROUTES.get(entry, []))  # type: ignore[arg-type]
    }
    assert trimmed == {}, f"protection rows whose pin list drifted: {sorted(trimmed)}"

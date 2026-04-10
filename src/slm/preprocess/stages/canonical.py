from __future__ import annotations

import ast
import builtins
import io
import tokenize
from dataclasses import dataclass
from functools import partial
from typing import Any, Literal

import ray

_BUILTINS = set(dir(builtins))


@dataclass
class CanonResult:
    ok: bool
    rep: Any  # str (dump) or list[str] (node_types)
    err: str | None = None


@dataclass
class CanonicalizerConfig:
    representation: str = "node_types"
    traversal: str = "preorder"

    remove_docstrings: bool = True
    rename_locals: bool = True
    rename_args: bool = True
    rename_function_names: bool = False
    rename_class_names: bool = False

    normalize_literals: str = "basic"
    keep_builtins: bool = True

    max_code_chars: int = 200_000
    on_parse_error: str = "skip"


class Canonicalizer(ast.NodeTransformer):
    def __init__(self, cfg: CanonicalizerConfig):
        super().__init__()
        self.cfg = cfg
        self.name_map: dict[str, str] = {}
        self.counter = 0

    def _map_name(self, s: str, prefix: str = "VAR") -> str:
        if self.cfg.keep_builtins and s in _BUILTINS:
            return s
        if s not in self.name_map:
            self.name_map[s] = f"{prefix}_{self.counter}"
            self.counter += 1
        return self.name_map[s]

    def _strip_docstring_from_body(self, body):
        if (not body) or (not self.cfg.remove_docstrings):
            return body
        first = body[0]
        if isinstance(first, ast.Expr) and isinstance(
            getattr(first, "value", None), ast.Constant
        ):
            if isinstance(first.value.value, str):
                return body[1:]
        return body

    def visit_Module(self, node: ast.Module):
        node = self.generic_visit(node)
        node.body = self._strip_docstring_from_body(node.body)
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef):
        if self.cfg.rename_function_names:
            node.name = self._map_name(node.name, prefix="FUNC")
        node = self.generic_visit(node)
        node.body = self._strip_docstring_from_body(node.body)
        return node

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        if self.cfg.rename_function_names:
            node.name = self._map_name(node.name, prefix="AFUNC")
        node = self.generic_visit(node)
        node.body = self._strip_docstring_from_body(node.body)
        return node

    def visit_ClassDef(self, node: ast.ClassDef):
        if self.cfg.rename_class_names:
            node.name = self._map_name(node.name, prefix="CLS")
        node = self.generic_visit(node)
        node.body = self._strip_docstring_from_body(node.body)
        return node

    def visit_arg(self, node: ast.arg):
        if self.cfg.rename_args:
            node.arg = self._map_name(node.arg, prefix="ARG")
        return node

    def visit_Name(self, node: ast.Name):
        if self.cfg.rename_locals:
            node.id = self._map_name(node.id, prefix="VAR")
        return node

    def visit_Constant(self, node: ast.Constant):
        mode = self.cfg.normalize_literals
        if mode == "none":
            return node

        v = node.value
        if isinstance(v, bool):
            node.value = "BOOL" if mode in ("basic", "aggressive") else v
        elif v is None:
            node.value = "NONE" if mode in ("basic", "aggressive") else v
        elif isinstance(v, (int, float, complex)):
            node.value = "NUM" if mode in ("basic", "aggressive") else v
        elif isinstance(v, str):
            node.value = "STR" if mode in ("basic", "aggressive") else v
        elif isinstance(v, bytes):
            node.value = "BYTES" if mode in ("basic", "aggressive") else v
        else:
            if mode == "aggressive":
                node.value = "CONST"
        return node


def _preorder_node_types(root: ast.AST) -> list[str]:
    out: list[str] = []
    stack = [root]
    while stack:
        node = stack.pop()
        out.append(type(node).__name__)
        children = list(ast.iter_child_nodes(node))
        stack.extend(reversed(children))
    return out


def _postorder_node_types(root: ast.AST) -> list[str]:
    out: list[str] = []
    stack = [(root, False)]
    while stack:
        node, visited = stack.pop()
        if visited:
            out.append(type(node).__name__)
        else:
            stack.append((node, True))
            children = list(ast.iter_child_nodes(node))
            for ch in reversed(children):
                stack.append((ch, False))
    return out


def _bfs_node_types(root: ast.AST) -> list[str]:
    return [type(n).__name__ for n in ast.walk(root)]


def fallback_representation(
    code: str, cfg: CanonicalizerConfig, max_tokens: int = 5000
):
    try:
        toks = []
        for tok in tokenize.generate_tokens(io.StringIO(code).readline):
            ttype, tval = tok.type, tok.string

            if ttype in (
                tokenize.NL,
                tokenize.NEWLINE,
                tokenize.INDENT,
                tokenize.DEDENT,
                tokenize.ENCODING,
            ):
                continue
            if ttype == tokenize.COMMENT:
                continue

            if ttype == tokenize.NAME:
                toks.append("NAME")
            elif ttype == tokenize.NUMBER:
                toks.append("NUM")
            elif ttype == tokenize.STRING:
                toks.append("STR")
            else:
                toks.append(tval)

            if len(toks) >= max_tokens:
                break

        if cfg.representation == "dump":
            return " ".join(toks)
        return toks

    except Exception:
        norm = " ".join(code.replace("\r\n", "\n").replace("\r", "\n").split())
        if cfg.representation == "dump":
            return norm
        return norm.split()[:max_tokens]


def canonicalize(code: str, cfg: CanonicalizerConfig) -> CanonResult:
    try:
        if cfg.max_code_chars and len(code) > cfg.max_code_chars:
            return CanonResult(
                ok=False,
                rep=[] if cfg.representation == "node_types" else "",
                err="too_long",
            )

        tree = ast.parse(code)
        canon_tree = Canonicalizer(cfg).visit(tree)
        ast.fix_missing_locations(canon_tree)

        if cfg.representation == "dump":
            canon_str = ast.dump(
                canon_tree, annotate_fields=False, include_attributes=False
            )
            return CanonResult(ok=True, rep=canon_str)

        if cfg.traversal == "preorder":
            seq = _preorder_node_types(canon_tree)
        elif cfg.traversal == "postorder":
            seq = _postorder_node_types(canon_tree)
        else:
            seq = _bfs_node_types(canon_tree)

        return CanonResult(ok=True, rep=seq)

    except SyntaxError as e:
        if cfg.on_parse_error == "fallback":
            rep = fallback_representation(code, cfg)
            return CanonResult(ok=True, rep=rep, err=f"SyntaxError_fallback:{e.msg}")
        return CanonResult(
            ok=False,
            rep=[] if cfg.representation == "node_types" else "",
            err=f"SyntaxError:{e.msg}",
        )
    except Exception as e:
        return CanonResult(
            ok=False,
            rep=[] if cfg.representation == "node_types" else "",
            err=f"{type(e).__name__}:{e}",
        )


def transform_canonicalize_row(row: dict[str, Any], canon_cfg):
    if row.get("language") != "python" or not row.get("has_code"):
        row["parse_ok"] = False
        row["parse_err"] = "not_python_or_no_code"

        if canon_cfg.representation == "dump":
            row["canon_dump"] = ""
        else:
            row["node_types"] = []
        return row

    res = canonicalize(row.get("code_ref", "") or "", canon_cfg)
    row["parse_ok"] = res.ok
    row["parse_err"] = res.err

    if canon_cfg.representation == "dump":
        row["canon_dump"] = res.rep if res.ok else ""
    else:
        row["node_types"] = res.rep if res.ok else []

    return row


def apply_canonicalize(ds: ray.data.Dataset, canon_cfg):
    row_fn = partial(transform_canonicalize_row, canon_cfg=canon_cfg)
    return ds.map(row_fn)
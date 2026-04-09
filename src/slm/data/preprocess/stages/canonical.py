import ast
import builtins
from dataclasses import dataclass
from typing import Any, Literal
import io
import tokenize
import ray

_BUILTINS = set(dir(builtins))


@dataclass
class CanonResult:
    ok: bool
    rep: Any  # str (dump) or list[str] (node_types)
    err: str | None = None


@dataclass
class CanonicalizerConfig:
    # Output choice
    representation: Literal["node_types", "dump"] = "node_types"
    traversal: Literal["preorder", "postorder", "bfs"] = (
        "preorder"  # only for node_types
    )

    # Canonicalization knobs
    remove_docstrings: bool = True
    rename_locals: bool = True
    rename_args: bool = True
    rename_function_names: bool = False
    rename_class_names: bool = False

    normalize_literals: Literal["none", "basic", "aggressive"] = "basic"
    keep_builtins: bool = True

    max_code_chars: int = 200_000
    on_parse_error: Literal["skip", "fallback"] = (
        "skip"  # fallback = use normalized text etc.
    )


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


################# TRAVERSAL METHOD #####################
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


################# TRAVERSAL METHOD #####################


def fallback_representation(
    code: str, cfg: CanonicalizerConfig, max_tokens: int = 5000
):
    """
    Fallback when ast.parse fails.
    - Uses Python tokenizer (not AST) to produce a stable-ish token stream.
    - Returns:
        - string if cfg.representation=="dump"
        - list[str] if cfg.representation=="node_types"
    """
    try:
        toks = []
        for tok in tokenize.generate_tokens(io.StringIO(code).readline):
            ttype, tval = tok.type, tok.string

            # Drop pure formatting
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

            # Normalize token values a bit
            if ttype == tokenize.NAME:
                toks.append("NAME")  # normalize identifiers
            elif ttype == tokenize.NUMBER:
                toks.append("NUM")
            elif ttype == tokenize.STRING:
                toks.append("STR")
            else:
                toks.append(tval)  # operators / punctuation etc.

            if len(toks) >= max_tokens:
                break

        if cfg.representation == "dump":
            return " ".join(toks)  # string fallback
        return toks  # token-list fallback

    except Exception:
        # last resort: whitespace normalize
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

        # node_types
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


def run_stage_canonicalize(cfg, stage_paths: dict[str, str]):
    ds = ray.data.read_parquet(cfg.run.input_dir)

    if getattr(cfg.run, "debug", False):
        ds = ds.limit(getattr(cfg.run, "debug_max_rows", 2000))

    def add_canon(row):
        # skip non-python/no-code
        if row.get("language") != "python" or not row.get("has_code"):
            row["parse_ok"] = False
            row["parse_err"] = "not_python_or_no_code"
            # keep representation column consistent
            if cfg.canonicalize.representation == "dump":
                row["canon_dump"] = ""
            else:
                row["node_types"] = []
            return row

        res = canonicalize(row.get("code_ref", "") or "", cfg.canonicalize)
        row["parse_ok"] = res.ok
        row["parse_err"] = res.err

        if cfg.canonicalize.representation == "dump":
            row["canon_dump"] = res.rep if res.ok else ""
        else:
            row["node_types"] = res.rep if res.ok else []

        return row

    ds2 = ds.map(add_canon)
    # If you want downstream stages to only see parse_ok rows:
    ds_canon = ds2.filter(lambda r: r.get("parse_ok", False))

    out_dir = stage_paths["canonicalize"]
    ds_canon.write_parquet(out_dir)
    print("wrote canonicalize:", out_dir, "| rows:", ds_canon.count())

    return ds_canon


if __name__ == "__main__":
    import ray

    ray.init()
    ds = ray.data.read_parquet(
        "../../dataset/processed_datasets/unified_python/29_000000_000000.parquet"
    )

    cfg = CanonicalizerConfig(
        representation="node_types", traversal="preorder", on_parse_error="skip"
    )

    def add_node_types(row):
        if row.get("language") != "python" or not row.get("has_code"):
            row["parse_ok"] = False
            row["node_types"] = []
            row["parse_err"] = "not_python_or_no_code"
            return row

        res = canonicalize(row["code_ref"], cfg)
        row["parse_ok"] = res.ok
        row["node_types"] = res.rep if res.ok else []
        row["parse_err"] = res.err
        return row

    ds2 = ds.map(add_node_types)
    rows = ds2.take(3)
    for r in rows:
        print(
            r["id"],
            r["parse_ok"],
            "len(node_types)=",
            len(r["node_types"]),
            r["parse_err"] or "",
        )
        print(r["node_types"][:25], "...\n")

    out_dir = "intermediate/canon_node_types_v0001_single"

    ds2_single = ds2.repartition(1)  # one output shard
    ds2_single.write_parquet(out_dir)
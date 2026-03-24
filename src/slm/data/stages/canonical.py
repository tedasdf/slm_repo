import ast
import builtins
from dataclasses import dataclass
from typing import Any, Literal
import io
import tokenize


_BUILTINS = set(dir(builtins))

@dataclass 
class CanonResult:
    ok: bool
    rep: Any
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
        self.name_map: dict[str, str] = {} # assign the varaible name here
        self.counter = 0

    def _map_name(self, s:str, prefix: str = "VAR") -> str:
        if self.cfg.keep_builtins and s in _BUILTINS :
            return ss
        if s not in self.name_map:
            self.name_map[s] = f"{prefix}_{self.counter}"
            self.counter += 1
        return self.name_map[s]
    


def canonicalize(code: str, cfg: CanonicalizerConfig) -> CanonResult:
    try:
        if cfg.max_code_chars and len(code) > cfg.max_code_chars:
            return CanonResult(
                ok=False,
                rep=[] if cfg.representation == "node_types" else "",
                err="too_long",
            )
        # turn into Module(body= [...]) 
        tree = ast.parse(code)
        # turn cfg into canon_tree
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

if __name__ == "__main__":
    


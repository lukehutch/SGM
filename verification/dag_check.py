#!/usr/bin/env python3
"""Check that a LaTeX manuscript and optional check file obey the backward DAG.

Usage:
    python3 dag_check.py PAPER.tex [CHECKFILE.py]

Enforces the positional backward-only DAG and Supplement Firewall from
STANDARDS.md (Sec. 4 and Sec. 8):
  - Regions are identified by standard LaTeX markers:
      abstract    \\begin{abstract} ... \\end{abstract}
      intro       \\section[*][...]{Introduction} up to the next \\section
      main        after intro (or start) up to \\appendix / bibliography
      appendix    \\appendix (or \\begin{appendices}) up to the bibliography
      supplement  from the bibliography marker (\\bibliography{...},
                  \\printbibliography, or \\begin{thebibliography}) to EOF
  - Positional forward references are permitted only from front/abstract/intro
    (within pre-bibliography regions), from main text into a bound appendix,
    or to a float (table, figure, longtable) on the same side of the
    bibliography.
  - Supplement Firewall: nothing before the bibliography (including front
    matter, abstract, intro, main text, bound appendices, and pre-bibliography
    floats) may reference any label or check block in the supplement.
  - Duplicate \\label{...} definitions and common forward prose pointers
    ("shown below", "next section", "will be proved", etc.) outside front
    matter/abstract/intro are flagged as violations.
  - With an optional check file (verification/<paper>_check.py), uses Python's
    AST to extract every eq(label, line, ...) call site and verifies:
      (a) literal (label, line) arguments and in-range source line numbers,
      (b) non-decreasing source line order across blocks, and
      (c) every paper label or check-only block ID cited on a "# Derived from:"
          comment line is defined, respects the Supplement Firewall, and does
          not point forward (with the same main->appendix and same-side float
          exemptions as the paper check).

Exit status is 0 if no violation is found and 1 otherwise. Static linting is
necessary, not sufficient: manual review of global logical acyclicity, block
coverage, mutants, and prose supplement pointers remains mandatory.
"""
import argparse
import ast
import re
import sys
from pathlib import Path

REF = re.compile(r"\\(?:eqref|ref|cref|Cref|autoref|pageref)\*?\{([^}]*)\}")
LABEL = re.compile(r"\\label\{([^}]*)\}")
POINTER = re.compile(
    r"\b((?:see|shown|given|derived|discussed|treated|proved|defined|stated|"
    r"computed|described|explained|listed|returned to)\s+(?:in detail\s+)?"
    r"(?:below|later)|(?:next|following|subsequent|later)\s+"
    r"(?:section|subsection|appendix)|"
    r"will be (?:shown|derived|proved|discussed|treated|given)|we will)\b",
    re.I,
)
COMMENT_RE = re.compile(r"(?<!\\)((?:\\\\)*)%.*")
INTRO_RE = re.compile(r"\\section\*?(?:\[[^\]]*\])?\{Introduction\}")
SECTION_RE = re.compile(r"\\section\*?(?:\[[^\]]*\])?\{")
APPENDIX_RE = re.compile(r"\\appendix(?![A-Za-z@])|\\begin\{appendices\}")
BIB_RE = re.compile(
    r"\\bibliography\{|\\printbibliography(?![A-Za-z@])|\\begin\{thebibliography\}"
)
FLOAT_RE = re.compile(
    r"\\begin\{(table|figure|longtable)\*?\}.*?\\end\{\1\*?\}", re.S
)


def strip_comments(text):
    """Blank out unescaped % comments (preserving \\% and \\\\% breaks)."""
    return "\n".join(COMMENT_RE.sub(r"\1", ln) for ln in text.split("\n"))


def line_of(text, pos):
    return text.count("\n", 0, pos) + 1


def regions(text):
    """Return a function mapping a character offset to its region name."""
    def find(rx, start=0):
        m = rx.search(text, start)
        return m.start() if m else None

    end = len(text)
    abs_a = find(re.compile(r"\\begin\{abstract\}"))
    abs_b = find(re.compile(r"\\end\{abstract\}"), abs_a + 1) if abs_a is not None else None
    intro = find(INTRO_RE)
    bib = find(BIB_RE)
    appendix = find(APPENDIX_RE)
    if bib is not None and appendix is not None and appendix > bib:
        # An \appendix inside the supplement does not create a bound appendix.
        appendix = None

    pre_bib_end = min(x for x in (appendix, bib, end) if x is not None)
    after_intro = None
    if intro is not None:
        nxt = find(SECTION_RE, intro + 1)
        if nxt is not None and nxt < pre_bib_end:
            after_intro = nxt
        else:
            after_intro = pre_bib_end

    bounds = []
    if abs_a is not None:
        bounds.append((abs_a, abs_b if abs_b is not None else end, "abstract"))
    if intro is not None:
        bounds.append((intro, after_intro, "intro"))
    main_start = after_intro if after_intro is not None else 0
    if main_start < pre_bib_end:
        bounds.append((main_start, pre_bib_end, "main"))
    if appendix is not None:
        bounds.append((appendix, bib if bib is not None else end, "appendix"))
    if bib is not None:
        bounds.append((bib, end, "supplement"))

    def region(pos):
        for a, b, name in bounds:
            if a <= pos < b:
                return name
        return "front"

    return region


def float_spans(text):
    return [(m.start(), m.end()) for m in FLOAT_RE.finditer(text)]


def check_paper(path):
    text = strip_comments(Path(path).read_text(encoding="utf-8", errors="replace"))
    region = regions(text)
    floats = float_spans(text)
    labels = {}
    bad = []
    for m in LABEL.finditer(text):
        key = m.group(1).strip()
        if key in labels:
            first_ln = line_of(text, labels[key])
            dup_ln = line_of(text, m.start())
            bad.append((dup_ln, key, f"duplicate label (first defined at L{first_ln})"))
        else:
            labels[key] = m.start()

    n_refs = 0
    for m in REF.finditer(text):
        src = m.start()
        src_reg = region(src)
        for key in (k.strip() for k in m.group(1).split(",") if k.strip()):
            n_refs += 1
            if key not in labels:
                bad.append((line_of(text, src), key, "label not defined"))
                continue
            tgt = labels[key]
            tgt_reg = region(tgt)
            # Supplement firewall: nothing before bibliography may point into supplement.
            if src_reg != "supplement" and tgt_reg == "supplement":
                bad.append((
                    line_of(text, src),
                    key,
                    f"{src_reg} text points into the supplement (L{line_of(text, tgt)})",
                ))
                continue
            is_float = any(a <= tgt < b for a, b in floats)
            if tgt <= src or is_float or src_reg in ("abstract", "intro", "front"):
                continue
            if src_reg == "main" and tgt_reg == "appendix":
                continue
            bad.append((
                line_of(text, src),
                key,
                f"forward reference from {src_reg} to {tgt_reg} L{line_of(text, tgt)}",
            ))

    pointers = []
    for m in POINTER.finditer(text):
        if region(m.start()) in ("main", "appendix", "supplement"):
            ln = line_of(text, m.start())
            ctx = text[max(0, m.start() - 70):m.end() + 50].replace("\n", " ")
            pointers.append((ln, ctx))

    print(f"{path}: {len(labels)} labels, {n_refs} references checked")
    for ln, key, why in bad:
        print(f"  VIOLATION L{ln}: \\ref{{{key}}}: {why}")
    for ln, s in pointers:
        print(f"  VIOLATION L{ln}: prose pointer to later text: ...{s}...")
        bad.append((ln, "", "prose pointer"))
    return labels, text, bad


def extract_eq_calls(source_text, path):
    """Extract all eq(label, line, ...) call sites using Python's AST."""
    tree = ast.parse(source_text, filename=str(path))
    blocks = []
    errors = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not (isinstance(node.func, ast.Name) and node.func.id == "eq"):
            continue
        if len(node.args) < 2:
            errors.append(f"{path}:{node.lineno}: eq() call has fewer than 2 arguments")
            continue
        arg0, arg1 = node.args[0], node.args[1]
        if (
            isinstance(arg0, ast.Constant)
            and isinstance(arg0.value, str)
            and isinstance(arg1, ast.Constant)
            and isinstance(arg1.value, int)
            and not isinstance(arg1.value, bool)
        ):
            blocks.append((node.lineno, arg1.value, arg0.value))
        else:
            errors.append(
                f"{path}:{node.lineno}: eq() call requires literal (str, int) first two arguments"
            )
    blocks.sort(key=lambda item: item[0])
    return blocks, errors


def check_checkfile(path, labels, text):
    raw = Path(path).read_text(encoding="utf-8", errors="replace")
    src = raw.split("\n")
    bad = []
    try:
        blocks, parse_errors = extract_eq_calls(raw, path)
    except SyntaxError as exc:
        bad.append(f"{path}:{exc.lineno or 0}: Python syntax error: {exc.msg}")
        print(f"{path}: 0 equation blocks checked (syntax error)")
        for b in bad:
            print("  VIOLATION " + b)
        return bad

    bad.extend(parse_errors)

    for (i0, t0, l0), (i1, t1, l1) in zip(blocks, blocks[1:]):
        if t1 < t0:
            bad.append(f"{path}:{i1}: block {l1} (L{t1}) comes after {l0} (L{t0})")

    region = regions(text)
    floats = float_spans(text)
    line_start = [0] + [m.end() for m in re.finditer("\n", text)]
    n_tex_lines = len(line_start)
    in_float = lambda pos: any(a <= pos < b for a, b in floats)

    # Index both paper labels and check-only block IDs (e.g., chk:step_1).
    block_pos = {}
    for i, tex_line, label in blocks:
        if 1 <= tex_line <= n_tex_lines:
            block_pos.setdefault(label, (tex_line, line_start[tex_line - 1]))
        else:
            bad.append(
                f"{path}:{i}: block {label} line {tex_line} is out of range (1..{n_tex_lines})"
            )

    paper_prefixes = {k.split(":")[0] for k in labels if ":" in k}
    check_prefixes = {
        lbl.split(":")[0]
        for _, _, lbl in blocks
        if ":" in lbl and lbl not in labels
    }
    all_prefixes = sorted(paper_prefixes | check_prefixes)
    label_pat = (
        re.compile(
            r"(?<![A-Za-z0-9_:])((?:"
            + "|".join(re.escape(p) for p in all_prefixes)
            + r"):[\w:.\-]+)"
        )
        if all_prefixes
        else None
    )

    for k, (i, tex_line, label) in enumerate(blocks):
        if not (1 <= tex_line <= n_tex_lines):
            continue
        start = blocks[k - 1][0] if k else 0
        chunk = src[start:i]
        derived = []
        on = False
        for ln in chunk:
            s = ln.strip()
            if s.startswith("# Derived from:"):
                on = True
            elif on and (not s.startswith("#") or re.match(r"#\s*(Test|Mutant|States):", s)):
                on = False
            if on and label_pat is not None:
                derived += label_pat.findall(s)

        pos = line_start[tex_line - 1]
        src_reg = region(pos)
        for d in derived:
            d = d.rstrip(".:")
            if d == label:
                continue
            if d in labels:
                tgt_pos = labels[d]
                d_line = line_of(text, tgt_pos)
            elif d in block_pos:
                d_line, tgt_pos = block_pos[d]
            else:
                bad.append(f"{path}:{i}: {label} cites unknown label {d}")
                continue

            tgt_reg = region(tgt_pos)
            if src_reg != "supplement" and tgt_reg == "supplement":
                bad.append(
                    f"{path}:{i}: {label} (L{tex_line}, {src_reg}) depends on supplement {d} (L{d_line})"
                )
                continue
            if (
                in_float(pos)
                or in_float(tgt_pos)
                or (src_reg == "main" and tgt_reg == "appendix")
            ):
                continue
            if d_line > tex_line:
                bad.append(
                    f"{path}:{i}: {label} (L{tex_line}) is derived from later {d} (L{d_line})"
                )

    print(f"{path}: {len(blocks)} equation blocks checked for order and derivation sources")
    for b in bad:
        print("  VIOLATION " + b)
    return bad


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Verify backward-only DAG and Supplement Firewall for a LaTeX paper and optional check file."
    )
    parser.add_argument("paper", help="Path to the root .tex manuscript")
    parser.add_argument(
        "checkfile",
        nargs="?",
        help="Optional path to verification/<paper>_check.py",
    )
    args = parser.parse_args(argv)
    labels, text, bad = check_paper(args.paper)
    if args.checkfile:
        bad = bad + check_checkfile(args.checkfile, labels, text)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

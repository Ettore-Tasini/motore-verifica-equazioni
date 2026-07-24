"""
Core verification logic: parsing, relation classification, and the final
check against the true solution set. Pure Python + SymPy, no web framework
dependency, so it can be tested and reasoned about in isolation.

Scope of this first version (per the agreed incremental roadmap): single
polynomial equations, degree 1 or 2, one variable, no parameters. Radicals,
fractions, logs/exponentials, trig, and systems come in later phases and
will currently fall back to an honest "not supported yet" result rather
than a wrong answer.
"""
import re
from sympy import (
    Symbol, Eq, expand, simplify, Poly, S, FiniteSet, EmptySet, sqrt
)
from sympy.parsing.sympy_parser import (
    parse_expr, standard_transformations, implicit_multiplication_application, convert_xor
)
from sympy import Integer as _Integer, Float as _Float
from sympy import sqrt as _sqrt, log as _log, sin as _sin, cos as _cos, tan as _tan, exp as _exp, pi as _pi, E as _E, Rational as _Rational

# Only these names are exposed to the parser — deliberately restrictive,
# since this endpoint will be reachable on the internet (even behind a
# shared key). No arbitrary code execution should ever be possible here.
# Integer/Float/Symbol are required by SymPy's own auto_number/auto_symbol
# transformations (they build numbers and variable names out of these), not
# an extra surface — everything else here is plain math.
SAFE_GLOBALS = {
    'sqrt': _sqrt, 'log': _log, 'sin': _sin, 'cos': _cos, 'tan': _tan,
    'exp': _exp, 'pi': _pi, 'E': _E, 'Rational': _Rational,
    'Integer': _Integer, 'Float': _Float, 'Symbol': Symbol,
}
TRANSFORMATIONS = standard_transformations + (implicit_multiplication_application, convert_xor)


def safe_parse(text):
    """Parses a math expression from student/OCR text into a SymPy expression.
    Restricted namespace: only the names in SAFE_GLOBALS are available, plus
    auto-created single-letter symbols for the variable(s)."""
    text = text.strip()
    return parse_expr(text, transformations=TRANSFORMATIONS, global_dict=dict(SAFE_GLOBALS), evaluate=True)


def split_equation(text):
    """'lhs = rhs' (textual) -> (lhs_expr, rhs_expr). Raises ValueError with a
    human-readable reason on anything that isn't exactly one '='."""
    if text.count('=') != 1:
        raise ValueError(f"attesa esattamente una '=', trovate {text.count('=')}")
    left_txt, right_txt = text.split('=')
    return safe_parse(left_txt), safe_parse(right_txt)


def detect_variable(*exprs, hint=None):
    free = set()
    for e in exprs:
        free |= e.free_symbols
    if hint:
        for s in free:
            if s.name == hint:
                return s
    if free:
        return sorted(free, key=lambda s: s.name)[0]
    return Symbol('x')


def poly_coeffs(expr, var):
    """Expands expr and returns {0: c0, 1: c1, 2: c2} as EXACT SymPy values,
    or None if expr isn't a degree<=2 polynomial in `var` alone (e.g. it has
    another free symbol/parameter, or a degree we don't support yet)."""
    expr = expand(expr)
    if expr.free_symbols - {var}:
        return None  # extra symbols/parameters: out of scope for now
    try:
        p = Poly(expr, var)
    except Exception:
        return None
    if p.degree() > 2:
        return None
    coeffs = {0: S(0), 1: S(0), 2: S(0)}
    for monom, coeff in zip(p.monoms(), p.coeffs()):
        coeffs[monom[0]] = coeff
    return coeffs


def check_relation(prev_lhs, prev_rhs, cur_lhs, cur_rhs, var):
    """Exact analogue of the JS engine's proportionality check, but symbolic
    (no numeric sampling, no floating-point tolerance needed)."""
    p = poly_coeffs(prev_lhs - prev_rhs, var)
    c = poly_coeffs(cur_lhs - cur_rhs, var)
    if p is None or c is None:
        return {'supported': False}
    k = None
    for deg in (2, 1, 0):
        if simplify(p[deg]) != 0:
            k = simplify(c[deg] / p[deg])
            break
    if k is None:
        # prev side was identically 0=0 (degenerate) — equivalent only if cur is too
        equivalent = all(simplify(c[d]) == 0 for d in (0, 1, 2))
        return {'supported': True, 'equivalent': equivalent, 'k': None, 'p': p, 'c': c}
    equivalent = (k != 0) and all(simplify(c[deg] - k * p[deg]) == 0 for deg in (0, 1, 2))
    return {'supported': True, 'equivalent': equivalent, 'k': k, 'p': p, 'c': c}


def fmt_num(v):
    v = simplify(v)
    return str(v)


def fmt_term(coeff, degree, var):
    coeff = simplify(coeff)
    if coeff == 0:
        return '0'
    sign = '-' if coeff.could_extract_minus_sign() else '+'
    absval = simplify(-coeff if coeff.could_extract_minus_sign() else coeff)
    numstr = fmt_num(absval)
    if degree == 0:
        return f"{sign}{numstr}"
    varpart = str(var) if degree == 1 else f"{var}^{degree}"
    return f"{sign}{varpart}" if numstr == '1' else f"{sign}{numstr}{varpart}"


def term_label(degree, var):
    if degree == 0:
        return 'termine noto'
    if degree == 1:
        return f"termine con {var}"
    return f"termine con {var}^{degree}"


def diagnose_step(prev_lhs, prev_rhs, cur_lhs, cur_rhs, var):
    """Per-side (left vs right) coefficient comparison — same style as the
    JS diagnosis, but exact/symbolic instead of numeric-fitted."""
    pL = poly_coeffs(prev_lhs, var); pR = poly_coeffs(prev_rhs, var)
    cL = poly_coeffs(cur_lhs, var); cR = poly_coeffs(cur_rhs, var)
    if None in (pL, pR, cL, cR):
        return "Il passaggio è troppo complesso per essere analizzato con precisione automaticamente: ricontrolla con calma ogni numero e segno."

    relevant = [d for d in (0, 1, 2) if any(simplify(x[d]) != 0 for x in (pL, pR, cL, cR))]
    mism = [d for d in relevant if simplify((cL[d] - pL[d]) - (cR[d] - pR[d])) != 0]

    if len(mism) == 0:
        return None  # shouldn't happen if caller already knows it's an error
    if len(mism) >= 3 or (len(relevant) > 1 and len(mism) >= len(relevant)):
        return "Qui non torna quasi nulla: ricontrolla questo passaggio con calma, sembra che più cose insieme non quadrino."

    parts = []
    for d in sorted(mism, reverse=True)[:2]:
        label = term_label(d, var)
        left_unchanged = simplify(pL[d] - cL[d]) == 0
        right_unchanged = simplify(pR[d] - cR[d]) == 0
        if left_unchanged and not right_unchanged:
            parts.append(f"nel {label} a sinistra sei rimasto a {fmt_term(pL[d], d, var)} (invariato), ma a destra sei passato da {fmt_term(pR[d], d, var)} a {fmt_term(cR[d], d, var)}: l'operazione va applicata su entrambi i lati")
        elif right_unchanged and not left_unchanged:
            parts.append(f"nel {label} a destra sei rimasto a {fmt_term(pR[d], d, var)} (invariato), ma a sinistra sei passato da {fmt_term(pL[d], d, var)} a {fmt_term(cL[d], d, var)}: l'operazione va applicata su entrambi i lati")
        else:
            parts.append(f"nel {label} a sinistra sei passato da {fmt_term(pL[d], d, var)} a {fmt_term(cL[d], d, var)}, a destra invece da {fmt_term(pR[d], d, var)} a {fmt_term(cR[d], d, var)}: i due lati devono cambiare allo stesso modo")
    text = ' ; '.join(parts) + '.'
    return text[0].upper() + text[1:]


def solve_original(lhs, rhs, var):
    """Solves the ORIGINAL equation (first row) for the true solution set,
    used by the final check. Real solutions only, matching the scope of
    these grades (no complex roots expected/taught here)."""
    from sympy import solveset
    return solveset(Eq(lhs, rhs), var, domain=S.Reals)


_SPLIT_RE = re.compile(r'\bor\b|\bo\b|∨|;|,')


def parse_declared_solutions(text, var):
    """Parses a student's final-answer line into a list of SymPy values.
    Accepts 'x=3', 'x=2 or x=3', 'x=2, x=3', etc."""
    parts = _SPLIT_RE.split(text)
    values = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        rhs_txt = part.split('=', 1)[1] if '=' in part else part
        try:
            values.append(simplify(safe_parse(rhs_txt)))
        except Exception:
            continue
    return values


def compare_solutions(true_set, declared_values):
    if not isinstance(true_set, FiniteSet):
        return {'status': 'unknown'}
    true_vals = list(true_set)
    matched = set()
    invalid = []
    for d in declared_values:
        found = False
        for i, t in enumerate(true_vals):
            if i in matched:
                continue
            if simplify(d - t) == 0:
                matched.add(i)
                found = True
                break
        if not found:
            invalid.append(d)
    missing = [t for i, t in enumerate(true_vals) if i not in matched]
    if invalid:
        return {'status': 'invalid_solution_present', 'invalid': invalid, 'missing': missing, 'correct': true_vals}
    if missing:
        return {'status': 'incomplete', 'missing': missing, 'correct': true_vals}
    return {'status': 'ok', 'correct': true_vals}


# Ruoli che questa prima versione sa gestire per il confronto passo-passo.
# Gli altri (domain, substitution, side_calc, case_split) sono riconosciuti
# dal contratto ma non ancora implementati: lo diciamo onestamente invece
# di ignorarli o fingere di capirli.
SUPPORTED_STEP_ROLES = {"equation", None, "equazione", ""}


def process_sheet(rows, variable_hint=None):
    """Framework-independent request handler: rows is a list of plain dicts
    with keys index/plain/role_hint (latex is accepted but unused server-side).
    Returns a plain dict matching the steps/final_check contract exactly —
    no FastAPI/Pydantic dependency, so this can be unit-tested directly."""
    steps = []
    prev_parsed = None
    first_equation = None
    solution_row = None

    for row in rows:
        idx = row.get('index')
        plain = row.get('plain', '') or ''
        role = (row.get('role_hint') or 'equation').lower()

        if role in ('solution', 'soluzione'):
            solution_row = row
            steps.append({'index': idx, 'status': 'skip', 'relation': 'solution'})
            continue

        if role not in SUPPORTED_STEP_ROLES:
            steps.append({
                'index': idx, 'status': 'unreadable', 'relation': role,
                'note': f"Le righe di tipo '{role}' non sono ancora supportate in questa versione: "
                        f"per ora vengono ignorate nel controllo, non contano né come errore né come ok."
            })
            continue

        if not plain.strip():
            steps.append({'index': idx, 'status': 'skip'})
            continue

        try:
            lhs, rhs = split_equation(plain)
            var = detect_variable(lhs, rhs, hint=variable_hint)
        except Exception as e:
            steps.append({'index': idx, 'status': 'unreadable',
                          'note': f"Non sono riuscito a interpretare questa riga come equazione: {e}"})
            continue

        if first_equation is None:
            first_equation = (lhs, rhs, var)

        if prev_parsed is None:
            steps.append({'index': idx, 'status': 'first'})
            prev_parsed = (lhs, rhs, var)
            continue

        p_lhs, p_rhs, p_var = prev_parsed
        rel = check_relation(p_lhs, p_rhs, lhs, rhs, var)

        if not rel['supported']:
            steps.append({
                'index': idx, 'status': 'unreadable', 'relation': None,
                'note': "Questo passaggio è troppo complesso per questa versione del motore "
                        "(per ora gestisce solo equazioni di 1° e 2° grado in una sola variabile, senza parametri)."
            })
        elif rel['equivalent']:
            steps.append({'index': idx, 'status': 'ok', 'relation': 'equivalence'})
        else:
            diag = diagnose_step(p_lhs, p_rhs, lhs, rhs, var)
            steps.append({'index': idx, 'status': 'error', 'relation': 'equivalence', 'diagnosis': diag})

        prev_parsed = (lhs, rhs, var)

    # ---------- Controllo finale ----------
    if solution_row is None and rows:
        solution_row = rows[-1]

    if first_equation is None or solution_row is None or not (solution_row.get('plain') or '').strip():
        final = {'status': 'no_final_solution',
                 'message': 'Non ho trovato una riga di soluzione finale da controllare.'}
    else:
        f_lhs, f_rhs, f_var = first_equation
        try:
            true_set = solve_original(f_lhs, f_rhs, f_var)
            declared = parse_declared_solutions(solution_row['plain'], f_var)
            cmp = compare_solutions(true_set, declared)
        except Exception:
            cmp = {'status': 'unknown'}

        if cmp['status'] == 'ok':
            final = {'status': 'ok', 'message': 'Hai trovato tutte le soluzioni corrette.'}
        elif cmp['status'] == 'incomplete':
            missing = [str(v) for v in cmp['missing']]
            final = {
                'status': 'incomplete',
                'message': "Le soluzioni che hai scritto sono corrette, ma non sono tutte: "
                           f"ti manca {'la soluzione' if len(missing) == 1 else 'le soluzioni'} {', '.join(missing)}.",
                'correct_solutions': [str(v) for v in cmp['correct']],
            }
        elif cmp['status'] == 'invalid_solution_present':
            invalid = [str(v) for v in cmp['invalid']]
            final = {
                'status': 'invalid_solution_present',
                'message': f"{'La soluzione' if len(invalid) == 1 else 'Le soluzioni'} {', '.join(invalid)} "
                           f"non {'è valida' if len(invalid) == 1 else 'sono valide'} per l'equazione di partenza. "
                           f"Le soluzioni corrette sono: {', '.join(str(v) for v in cmp['correct'])}.",
                'correct_solutions': [str(v) for v in cmp['correct']],
            }
        else:
            final = {
                'status': 'unknown',
                'message': "Non sono riuscito a determinare con certezza la soluzione di questo esercizio "
                           "(potrebbe essere troppo complesso per questa versione del motore)."
            }

    return {'steps': steps, 'final_check': final}

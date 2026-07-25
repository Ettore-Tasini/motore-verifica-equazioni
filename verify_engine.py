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

from sympy import E as _E
from sympy import (
    Eq,
    FiniteSet,
    Mul,
    Poly,
    S,
    Symbol,
    cancel,
    expand,
    lcm,
    simplify,
    together,
)
from sympy import Float as _Float
from sympy import Integer as _Integer
from sympy import Rational as _Rational
from sympy import cos as _cos
from sympy import exp as _exp
from sympy import log as _log
from sympy import pi as _pi
from sympy import sin as _sin
from sympy import sqrt as _sqrt
from sympy import tan as _tan
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

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
        return min(free, key=lambda s: s.name)
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


def side_denominator(expr, var):
    """Denominatore di expr dopo averlo ricondotto a una frazione unica
    (together); Integer(1) se expr non ha alcuna frazione. Usa together (non
    cancel) apposta: cancel semplificherebbe via i fattori comuni tra
    numeratore e denominatore, nascondendo un valore che andrebbe escluso dal
    dominio (es. (x+3)/(x**2-9) ha comunque x=3 da escludere)."""
    _, denom = together(expr).as_numer_denom()
    return denom


def row_has_fraction(lhs, rhs, var):
    """True se lhs o rhs contiene, dopo together, un denominatore che
    dipende da var — cioè questa riga è (o contiene) una frazione algebrica
    in var."""
    return var in side_denominator(lhs, var).free_symbols or var in side_denominator(rhs, var).free_symbols


def common_denominator(lhs, rhs, var):
    """mcm dei denominatori var-dipendenti di lhs e rhs, calcolato SOLO sui
    lati passati (mai mischiando prev e cur: moltiplicare due volte una riga
    già cancellata falserebbe il confronto). Integer(1) se nessuno dei due
    lati ha un denominatore che dipende da var."""
    dl, dr = side_denominator(lhs, var), side_denominator(rhs, var)
    dl_has_var, dr_has_var = var in dl.free_symbols, var in dr.free_symbols
    if dl_has_var and dr_has_var:
        return lcm(dl, dr)
    if dl_has_var:
        return dl
    if dr_has_var:
        return dr
    return S(1)


def clear_denominator(lhs, rhs, D):
    """(lhs*D, rhs*D) espansi dopo cancel. Il risultato può non essere
    polinomiale se D non è il denominatore giusto per questi lati: il
    chiamante verifica il risultato via poly_coeffs/check_relation, questa
    funzione non lo garantisce da sola."""
    return expand(cancel(lhs * D)), expand(cancel(rhs * D))


def rational_ratio(prev_lhs, prev_rhs, cur_lhs, cur_rhs, var):
    """Classificatore primario per transizioni tra righe con frazioni:
    calcola ratio = cancel((cur_lhs-cur_rhs)/(prev_lhs-prev_rhs)). Se il
    rapporto risulta 'pulito' (var non resta al denominatore del rapporto
    stesso) il passaggio è valido — equivalenza se il rapporto è una costante
    pura, implicazione se dipende da var (moltiplicazione per un'espressione
    che può annullarsi). Se il rapporto resta 'sporco' non è classificabile
    con sicurezza: {'supported': False} — non è MAI usato per dedurre un
    errore, solo equivalence/implication."""
    diff_p = cancel(prev_lhs - prev_rhs)
    diff_c = cancel(cur_lhs - cur_rhs)
    if diff_p == 0:
        return {'supported': True, 'equivalent': diff_c == 0, 'relation_kind': 'equivalence'}

    ratio = cancel(diff_c / diff_p)
    _, ratio_denom = ratio.as_numer_denom()
    if var in ratio_denom.free_symbols:
        return {'supported': False}
    if var in ratio.free_symbols:
        return {'supported': True, 'equivalent': True, 'relation_kind': 'implication'}
    return {'supported': True, 'equivalent': ratio != 0, 'relation_kind': 'equivalence'}


def check_fraction_relation(prev_lhs, prev_rhs, cur_lhs, cur_rhs, var):
    """Layer sopra check_relation per transizioni con frazioni: prova prima
    rational_ratio (decide ok/implication in un colpo solo, anche per casi
    insoliti come moltiplicare per il denominatore più un fattore extra). Se
    il rapporto è sporco e la riga precedente ha un denominatore var-
    dipendente e la riga corrente è già polinomiale pulita, calcola il
    denominatore comune di prev e riusa check_relation (e quindi
    diagnose_step, tramite diagnose_fraction_step) per vedere se è un errore
    di cancellazione riconoscibile. Ritorna un dict compatibile con
    check_relation ({'supported', 'equivalent'}) più 'relation_kind', e
    quando equivalent=False anche 'expected_lhs'/'expected_rhs' per la
    diagnosi. Se nessuno dei due percorsi si applica: {'supported': False}
    (fallback onesto, mai un errore inventato)."""
    ratio_result = rational_ratio(prev_lhs, prev_rhs, cur_lhs, cur_rhs, var)
    if ratio_result['supported']:
        return ratio_result

    if not row_has_fraction(prev_lhs, prev_rhs, var):
        return {'supported': False}

    if row_has_fraction(cur_lhs, cur_rhs, var):
        # La riga corrente ha ancora una frazione: non è un tentativo di
        # eliminare il denominatore (quello produce un polinomio pulito),
        # ma di raccogliere più frazioni dello stesso lato in una sola.
        # Il confronto giusto è contro prev ricombinato (together), non
        # contro prev moltiplicato per D.
        expected_lhs, expected_rhs = together(prev_lhs), together(prev_rhs)
        diag = diagnose_combine_step(prev_lhs, prev_rhs, expected_lhs, expected_rhs, cur_lhs, cur_rhs, var)
        if diag is None:
            return {'supported': False}
        return {'supported': True, 'equivalent': False, 'relation_kind': 'equivalence', 'combine_diagnosis': diag}

    D = common_denominator(prev_lhs, prev_rhs, var)
    expected_lhs, expected_rhs = clear_denominator(prev_lhs, prev_rhs, D)

    cleared = check_relation(expected_lhs, expected_rhs, cur_lhs, cur_rhs, var)
    if not cleared['supported']:
        return {'supported': False}

    result = {'supported': True, 'equivalent': cleared['equivalent'], 'relation_kind': 'implication'}
    if not cleared['equivalent']:
        result['expected_lhs'] = expected_lhs
        result['expected_rhs'] = expected_rhs
    return result


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


def side_has_content(pSide, cSide):
    """True if this side is ever anything other than plain 0, before or after."""
    return any(simplify(pSide[d]) != 0 for d in (0, 1, 2)) or any(simplify(cSide[d]) != 0 for d in (0, 1, 2))


def _house_style(s):
    s = s.replace('**', '^')
    s = re.sub(r'(?<=[0-9)])\*(?=[a-zA-Z(])', '', s)
    s = s.replace('*', '')
    return s


def format_expr(expr):
    """Renders a SymPy expression in the same house style used elsewhere
    (2x not 2*x, x^2 not x**2) instead of SymPy's own str() conventions."""
    return _house_style(str(expand(expr)))


def format_expr_raw(expr):
    """Same house style as format_expr, but WITHOUT expand() first — for
    grounded diagnosis on expressions with a fraction, where expand() would
    distribute the numerator over the denominator (e.g. (x+1)/(x-2) becomes
    x/(x-2)+1/(x-2)), no longer matching what the student actually wrote."""
    return _house_style(str(expr))


def _pulled_factor(expr, var):
    """Se expr ha la forma F*(...) — un unico fattore Add (la parentesi
    raccolta) e il resto un monomio puro c*var**k, k>=1 — ritorna (F, k).
    None se questa struttura non è individuabile (nessun raccoglimento
    esplicito, o la parentesi contiene altri simboli). Lavora sull'espressione
    COSÌ COM'È STATA SCRITTA, prima di qualunque expand(): expand
    distribuirebbe subito F dentro la parentesi, cancellando l'unico indizio
    che lo studente ha provato a raccogliere qualcosa."""
    factors = Mul.make_args(expr)
    add_factors = [f for f in factors if f.is_Add]
    if len(add_factors) != 1:
        return None
    rest = [f for f in factors if f is not add_factors[0]]
    F = Mul(*rest) if rest else S(1)
    if F == 1 or F.free_symbols != {var}:
        return None
    try:
        p = Poly(F, var)
    except Exception:
        return None
    monoms = p.monoms()
    if len(monoms) != 1 or monoms[0][0] < 1:
        return None
    return F, monoms[0][0]


def _bad_factor_diagnosis(prev_lhs, prev_rhs, cur_lhs, cur_rhs, var):
    """Riconosce un raccoglimento impossibile: lo studente ha raccolto un
    fattore var**k su un lato (es. x^2) ma quel lato, nel passaggio
    precedente, aveva un termine di grado inferiore a k (es. 3x) — quel
    termine non è un multiplo del fattore raccolto, quindi il raccoglimento
    non è valido a prescindere da cosa scrive dentro la parentesi. None se il
    pattern non si applica (nessun raccoglimento esplicito individuato, o il
    grado raccolto è comunque valido — allora l'errore è altrove, tipicamente
    un pezzo dimenticato dentro la parentesi)."""
    for prev_side, cur_side, label in ((prev_lhs, cur_lhs, "a sinistra"), (prev_rhs, cur_rhs, "a destra")):
        # Il raccoglimento potrebbe non coprire l'intero lato (es. resta un
        # termine noto fuori dalla parentesi: "x^2*(x+3) - 4"): va cercato in
        # ciascun addendo, non solo nel lato preso per intero.
        terms = cur_side.args if cur_side.is_Add else (cur_side,)
        for term in terms:
            pulled = _pulled_factor(term, var)
            if pulled is None:
                continue
            F, k = pulled
            prev_coeffs = poly_coeffs(prev_side, var)
            if prev_coeffs is None:
                continue
            offending = [d for d in range(k) if simplify(prev_coeffs[d]) != 0]
            if not offending:
                continue
            d = max(offending)
            term_str = fmt_term(prev_coeffs[d], d, var)
            if term_str.startswith('+'):
                term_str = term_str[1:]
            factor_str = format_expr_raw(F)
            return (f"Non puoi raccogliere {factor_str} {label}: il termine {term_str} "
                    f"non è un multiplo di {factor_str}.")
    return None


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

    # Is every mismatch confined to ONE side, while the OTHER side never has
    # any content at all (not just "unchanged at this degree", but plainly
    # always 0 everywhere)? That's not "you forgot to do it on both sides" —
    # there's nothing on the other side to do it to. It's a bad rewrite
    # WITHIN one side (wrong distribution/factoring), and needs a completely
    # different message: compare the whole side before/after, not term by term.
    right_is_trivial = not side_has_content(pR, cR)
    left_is_trivial = not side_has_content(pL, cL)
    mism_all_on_left = all(simplify(pR[d] - cR[d]) == 0 for d in mism)
    mism_all_on_right = all(simplify(pL[d] - cL[d]) == 0 for d in mism)

    if mism_all_on_left and right_is_trivial:
        bad_factor = _bad_factor_diagnosis(prev_lhs, prev_rhs, cur_lhs, cur_rhs, var)
        if bad_factor is not None:
            return bad_factor
        prev_s, cur_s = format_expr(prev_lhs), format_expr(cur_lhs)
        return (f"Hai riscritto il lato sinistro in modo scorretto: prima era {prev_s}, ora è {cur_s} — "
                f"non sono la stessa espressione. Se hai provato a raccogliere un fattore comune, "
                f"controlla che moltiplichi davvero TUTTI i termini dentro la parentesi, non solo alcuni.")
    if mism_all_on_right and left_is_trivial:
        bad_factor = _bad_factor_diagnosis(prev_lhs, prev_rhs, cur_lhs, cur_rhs, var)
        if bad_factor is not None:
            return bad_factor
        prev_s, cur_s = format_expr(prev_rhs), format_expr(cur_rhs)
        return (f"Hai riscritto il lato destro in modo scorretto: prima era {prev_s}, ora è {cur_s} — "
                f"non sono la stessa espressione. Se hai provato a raccogliere un fattore comune, "
                f"controlla che moltiplichi davvero TUTTI i termini dentro la parentesi, non solo alcuni.")

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


FRACTION_IMPLICATION_NOTE = (
    "Hai moltiplicato per un'espressione che contiene la incognita: da qui in "
    "poi potrebbero comparire soluzioni che non vanno bene nell'equazione di "
    "partenza — verrà controllato alla fine."
)


def diagnose_fraction_step(prev_lhs, prev_rhs, expected_lhs, expected_rhs, cur_lhs, cur_rhs, var):
    """Diagnosi per un errore nella cancellazione di un denominatore: una
    frase-ponte mostra la riga originale con la frazione (nella notazione
    dello studente) e cosa ci si aspetta dopo averla cancellata
    correttamente, poi riusa il confronto coefficiente-per-coefficiente di
    diagnose_step tra la forma attesa e quella scritta dallo studente."""
    prev_s = f"{format_expr_raw(prev_lhs)} = {format_expr_raw(prev_rhs)}"
    expected_s = f"{format_expr(expected_lhs)} = {format_expr(expected_rhs)}"
    detail = diagnose_step(expected_lhs, expected_rhs, cur_lhs, cur_rhs, var)
    bridge = f"Eliminando il denominatore da {prev_s} ci si aspetta {expected_s}. "
    return bridge + (detail or "")


def diagnose_combine_step(prev_lhs, prev_rhs, expected_lhs, expected_rhs, cur_lhs, cur_rhs, var):
    """Diagnosi per un errore nel raccogliere più frazioni in una sola sullo
    STESSO lato (mcm sbagliato, numeratore calcolato male), senza che
    l'equazione sia stata moltiplicata per nulla — quindi expected_lhs/rhs
    sono semplicemente prev ricombinato (together), non una forma
    polinomiale: il confronto è per cancellazione dell'intera frazione, non
    coefficiente per coefficiente (diagnose_step non si applica, richiede
    lati polinomiali). Usa format_expr_raw ovunque (mai format_expr, che
    farebbe expand e romperebbe la frazione unica in una somma di frazioni,
    non più fedele a quanto scritto)."""
    mism_left = simplify(cancel(cur_lhs - expected_lhs)) != 0
    mism_right = simplify(cancel(cur_rhs - expected_rhs)) != 0
    if not mism_left and not mism_right:
        return None

    def side_msg(label, prev_s, expected_s, cur_s):
        return (f"hai combinato le frazioni {label} in modo scorretto: partendo da {prev_s} "
                f"ci si aspetta {expected_s}, tu hai scritto {cur_s}. Controlla il minimo comune "
                f"denominatore: forse manca un fattore, o il numeratore non è stato calcolato bene")

    parts = []
    if mism_left:
        parts.append(side_msg("a sinistra", format_expr_raw(prev_lhs), format_expr_raw(expected_lhs), format_expr_raw(cur_lhs)))
    if mism_right:
        parts.append(side_msg("a destra", format_expr_raw(prev_rhs), format_expr_raw(expected_rhs), format_expr_raw(cur_rhs)))
    text = '; '.join(parts) + '.'
    return text[0].upper() + text[1:]


def solve_original(lhs, rhs, var):
    """Solves the ORIGINAL equation (first row) for the true solution set,
    used by the final check. Real solutions only, matching the scope of
    these grades (no complex roots expected/taught here)."""
    from sympy import solveset
    result = solveset(Eq(lhs, rhs), var, domain=S.Reals)
    if result == S.EmptySet and simplify(cancel(lhs - rhs)) == 0:
        # equazione razionale che è in realtà un'identità (vera per ogni x
        # tranne i poli): solveset può collassare erroneamente a EmptySet
        # invece di Reals meno i poli — non fidarsi, dichiarare 'unknown'.
        return None
    return result


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


_EMPTY_PHRASES = {
    'impossibile', 'nessunasoluzione', 'nessunesoluzioni', 'nonesistesoluzione',
    'nonesistonosoluzioni', 'insiemevuoto', 's=∅', '∅', 'nessuna', 'nosolution',
    'impossible', 's={}', 's=vuoto', 'equazioneimpossibile',
}


def is_declared_empty(text):
    """Riconosce risposte come 'impossibile', 'nessuna soluzione', 'S = ∅', ecc.
    come dichiarazione esplicita che l'equazione non ha soluzioni — da NON
    confondere con un testo che semplicemente non si riesce a interpretare."""
    t = text.strip().lower().replace(' ', '').rstrip('.')
    return t in _EMPTY_PHRASES


def compare_solutions(true_set, declared_values, declared_is_empty_claim=False):
    if true_set == S.EmptySet:
        true_vals = []
    elif isinstance(true_set, FiniteSet):
        true_vals = list(true_set)
    else:
        return {'status': 'unknown'}

    if declared_is_empty_claim:
        if len(true_vals) == 0:
            return {'status': 'ok', 'correct': []}
        return {'status': 'wrongly_claimed_impossible', 'missing': true_vals, 'correct': true_vals}

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

        p_lhs, p_rhs, _p_var = prev_parsed
        rel = check_relation(p_lhs, p_rhs, lhs, rhs, var)
        relation_label = 'equivalence' if rel['supported'] else None

        if not rel['supported'] and (row_has_fraction(p_lhs, p_rhs, var) or row_has_fraction(lhs, rhs, var)):
            rel = check_fraction_relation(p_lhs, p_rhs, lhs, rhs, var)
            relation_label = rel.get('relation_kind')

        if not rel['supported']:
            bad_factor_diag = _bad_factor_diagnosis(p_lhs, p_rhs, lhs, rhs, var)
            if bad_factor_diag is not None:
                steps.append({'index': idx, 'status': 'error', 'relation': None, 'diagnosis': bad_factor_diag})
            else:
                steps.append({
                    'index': idx, 'status': 'unreadable', 'relation': None,
                    'note': "Questo passaggio è troppo complesso per questa versione del motore "
                            "(per ora gestisce solo equazioni di 1°/2° grado, anche con frazioni algebriche, "
                            "in una sola variabile, senza parametri)."
                })
        elif rel['equivalent']:
            note = FRACTION_IMPLICATION_NOTE if relation_label == 'implication' else None
            steps.append({'index': idx, 'status': 'ok', 'relation': relation_label, 'note': note})
        else:
            if 'combine_diagnosis' in rel:
                diag = rel['combine_diagnosis']
            elif relation_label == 'implication':
                diag = diagnose_fraction_step(p_lhs, p_rhs, rel['expected_lhs'], rel['expected_rhs'], lhs, rhs, var)
            else:
                diag = diagnose_step(p_lhs, p_rhs, lhs, rhs, var)
            steps.append({'index': idx, 'status': 'error', 'relation': relation_label, 'diagnosis': diag})

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
            if true_set is None:
                cmp = {'status': 'unknown'}
            else:
                plain_text = solution_row['plain']
                if is_declared_empty(plain_text):
                    cmp = compare_solutions(true_set, [], declared_is_empty_claim=True)
                else:
                    declared = parse_declared_solutions(plain_text, f_var)
                    cmp = compare_solutions(true_set, declared)
        except Exception:
            cmp = {'status': 'unknown'}

        if cmp['status'] == 'ok':
            final = {'status': 'ok', 'message': 'Hai trovato tutte le soluzioni corrette.'}
        elif cmp['status'] == 'wrongly_claimed_impossible':
            correct = [str(v) for v in cmp['correct']]
            final = {
                'status': 'incomplete',
                'message': "In realtà l'equazione ha soluzione: hai detto che era impossibile, ma non lo è. "
                           f"{'La soluzione corretta è' if len(correct) == 1 else 'Le soluzioni corrette sono'}: {', '.join(correct)}.",
                'correct_solutions': correct,
            }
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
            if not cmp['correct']:
                final = {
                    'status': 'invalid_solution_present',
                    'message': f"{'Il valore' if len(invalid) == 1 else 'I valori'} {', '.join(invalid)} "
                               f"che hai scritto non {'è una soluzione valida' if len(invalid) == 1 else 'sono soluzioni valide'}: "
                               "questa equazione in realtà non ha soluzioni reali (è impossibile).",
                    'correct_solutions': [],
                }
            else:
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

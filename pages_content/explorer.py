"""Question Explorer: inspect one question and the evidence behind each matched section."""
from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd
import streamlit as st

from components import state, ui
from pages_content import _shared as sh
from services import record_export as rx


def _section_text(svc, record: Dict[str, Any], match: Dict[str, Any]) -> Optional[str]:
    """Current text of the matched section, but only if its document is byte-identical to the analysed one."""
    sec = svc.library.get_section(match["section_id"])
    frozen = next((d for d in record["documents"] if d["document_id"] == match["document_id"]), None)
    if not sec or not frozen:
        return None
    try:
        if svc.library.get(sec["document_id"]).file_hash != frozen["file_hash"]:
            return None
    except Exception:
        return None
    return sec["text"]


def _match_card(svc, record, q, m) -> None:
    ui.render(f'<div class="sl-card"><div class="sl-card-title">#{m["rank"]} {ui.esc(m["heading"])}</div>'
              f'<div>{ui.tier_badge(m["tier"])} {ui.badge("similarity %.3f" % m["score"])}</div>'
              f'<div class="sl-card-meta"><span>{ui.esc(m["doc_name"])}</span><span>{ui.esc(m["locator"])}</span></div></div>')
    if m["key_terms"]:
        ui.render("<p class='sl-caption'>Shared terms driving this score: " + " ".join(ui.badge(t, "primary") for t in m["key_terms"]) + "</p>")
    if m["explanation"]:
        st.caption(m["explanation"])
    full = _section_text(svc, record, m)
    with st.expander("Section text" + (" (current document, unchanged since the analysis)" if full else " (excerpt saved with the analysis)")):
        ui.render(f'<div class="sl-snippet">{ui.esc(full if full else m["snippet"])}</div>')
        if full is None:
            st.caption("The source document was changed or removed after this analysis, so only the saved excerpt is shown.")


def render() -> None:
    svc = state.get_services()
    ui.page_header("Question Explorer & Practice", "Browse questions by pedagogical category, practice with grounded answers, and view numerical solutions.")
    subject = state.require_subject(svc)
    state.show_scope(svc, subject)
    loaded = sh.analysis_picker(svc, subject, key="ex_pick")
    if not loaded:
        return
    meta, record = loaded
    sh.staleness_notice(svc, meta)
    questions = record["questions"]

    # Practice mode switcher: Historical PYQs vs Generated Practice Questions
    mode = st.radio("Question Source", ["Previous-Year Exam Questions (PYQs)", "Generated Practice Questions"],
                    horizontal=True, key=f"ex_mode_{meta['id']}")

    if mode == "Generated Practice Questions":
        gen_qs = svc.question_practice.generate_practice_questions(subject.id) if hasattr(svc, "question_practice") else []
        if not gen_qs:
            ui.empty_state("No practice questions generated", "Upload lecture notes to synthesize practice questions.")
            return
        questions = [q.to_dict() for q in gen_qs]

    c1, c2, c3, c4 = st.columns([2, 2, 2, 3])
    tiers = c1.multiselect("Tiers", ["Strong", "Moderate", "Weak", "Unmatched"],
                           default=["Strong", "Moderate", "Weak", "Unmatched"], key=f"ex_tier_{meta['id']}")
    categories = ["All", "Short-Answer & Definition", "Long-Answer & Theory", "Comparison & Distinction",
                  "Derivation & Proof", "Numerical Problem", "Diagram-Based"]
    cat_sel = c2.selectbox("Category", categories, key=f"ex_cat_{meta['id']}")

    multi = len(record["papers"]) > 1 and mode != "Generated Practice Questions"
    paper = c3.selectbox("Paper", [""] + [p["document_id"] for p in record["papers"]], key=f"ex_paper_{meta['id']}", disabled=not multi,
                         format_func=lambda i: "All papers" if not i else next(p["title"] for p in record["papers"] if p["document_id"] == i))
    text = c4.text_input("Search question text", key=f"ex_text_{meta['id']}")

    shown = []
    for q in questions:
        if q.get("best_tier") and q["best_tier"] not in tiers:
            continue
        if multi and paper and q.get("paper_doc_id") != paper:
            continue
        if text and text.lower() not in q.get("text", "").lower():
            continue
        if cat_sel != "All":
            cat_name = q.get("category_name", q.get("category", ""))
            if cat_sel.lower() not in cat_name.lower():
                continue
        shown.append(q)

    if not shown:
        ui.empty_state("No questions match these filters", "Widen the category or tier selection or clear the search.")
        return

    ids = [q["question_id"] for q in shown]
    by_id = {q["question_id"]: q for q in shown}
    chosen = st.selectbox("Question", ids, key=f"ex_q_{meta['id']}",
                          format_func=lambda i: f'{by_id[i].get("display_id", i)}  [{by_id[i].get("category_name", "General")}]  {by_id[i]["text"][:65]}' + (f' [{by_id[i].get("paper_label", "")[:18]}]' if multi else ""))
    q = by_id[chosen]

    # Question Card with Category Badge
    is_generated = q.get("is_pyq") is False
    if is_generated:
        marks = f'<span>{ui.esc(ui.fmt_marks(q["marks"]))}</span>' if q.get("marks") else ""
        category = ui.category_badge(q.get("category", ""), q.get("category_rationale", ""))
        ui.render(
            f'<div class="sl-card"><div class="sl-card-title">{ui.esc(q.get("display_id", "Practice question"))}'
            f' {category} {marks}</div><div>{ui.esc(q.get("text", ""))}</div>'
            f'<div class="sl-card-meta">{ui.esc(q.get("source_label", "Generated practice question"))}</div></div>'
        )
    else:
        ui.render(ui.question_card_html(q, show_paper=True))
        year = q.get("academic_year") or q.get("year")
        session = q.get("exam_session")
        source_details = " · ".join(str(v) for v in (year, session) if v)
        if source_details:
            st.caption(f"Historical source: {q.get('paper_label', 'Question paper')} · {source_details}")
    if q.get("category_rationale"):
        st.caption(f"Category classification: {q['category_rationale']}")

    # Grounded Answer Outline
    ao = q.get("answer_outline")
    if ao:
        with st.expander("Grounded Answer Outline (from study notes)", expanded=True):
            if ao.get("is_grounded"):
                if ao.get("core_answer"):
                    st.markdown(f"**Core Answer / Definition**:\n> {ao['core_answer']}")
                if ao.get("key_points"):
                    st.markdown("**Key Supporting Points**:")
                    for pt in ao["key_points"]:
                        st.markdown(f"- {pt}")
                if ao.get("formula"):
                    st.markdown(f"**Relevant Formula**: `{ao['formula']}`")
                if ao.get("citations"):
                    st.caption("Grounded Source References:")
                    for cit in ao["citations"]:
                        ui.render(f'<div class="sl-card-meta">{ui.badge("Citation", "primary")} <span>{ui.esc(cit.get("doc_name", ""))} ({ui.esc(cit.get("locator", ""))}) &mdash; {ui.esc(cit.get("heading", ""))}</span></div>')
            else:
                ui.notice("warning", "Insufficient Study Material", ao.get("insufficient_reason") or "No section in uploaded notes reached the weak match threshold.")

    # Numerical Problem Solver
    ns = q.get("numerical_solution")
    if not ns and hasattr(svc, "numerical_solver") and (q.get("category") == "numerical" or any(w in q.get("text", "").lower() for w in ("calculate", "compute", "value"))):
        raw_sol = svc.numerical_solver.analyze_and_solve(q.get("text", ""))
        ns = raw_sol.to_dict() if raw_sol else None

    if ns:
        with st.expander("Step-by-Step Numerical Solution", expanded=True):
            st.markdown(f"**Topic**: `{ns.get('topic_name')}` &mdash; *{ns.get('problem_type')}*")
            if ns.get("formula"):
                st.markdown(f"**Governing Formula**: `{ns['formula']}`")

            col_n1, col_n2 = st.columns(2)
            with col_n1:
                st.markdown("**Given Data**:")
                if ns.get("given_data"):
                    for k, v in ns["given_data"].items():
                        st.markdown(f"- **{k}**: `{v}`")
                else:
                    st.caption("No explicit numerical parameters specified in prompt.")

            with col_n2:
                if ns.get("variables_and_units"):
                    st.markdown("**Variables & Units**:")
                    for vk, vu in ns["variables_and_units"].items():
                        st.caption(f"- **{vk}**: {vu}")

            if ns.get("assumptions"):
                st.caption("Assumptions: " + "; ".join(ns["assumptions"]))

            if ns.get("is_solvable"):
                st.markdown("**Step-by-Step Calculation**:")
                for step in ns.get("steps", []):
                    st.markdown(f"- {step}")
                if ns.get("final_answer"):
                    st.success(f"**Final Answer**: {ns['final_answer']}")
                if ns.get("verification"):
                    st.info(ns["verification"])
            else:
                if ns.get("missing_data"):
                    ui.notice("warning", "Missing Required Data", "; ".join(ns["missing_data"]) + " SyllabusLens never invents numbers.")
                for step in ns.get("steps", []):
                    st.caption(step)

    # Concept Diagram
    diag_spec = q.get("diagram_spec")
    if not diag_spec and hasattr(svc, "diagrams"):
        d_obj = svc.diagrams.get_diagram_for_text(q.get("text", ""))
        if d_obj:
            diag_spec = dict(id=d_obj.diagram_id, title=d_obj.title, caption=d_obj.caption, dot=d_obj.dot_source)

    if diag_spec:
        with st.expander(f"Concept Diagram: {diag_spec['title']}"):
            st.caption(f"{diag_spec['caption']} (Deterministic structural representation; not an image extraction)")
            st.graphviz_chart(diag_spec["dot"])

    matches = q.get("matches", [])
    if is_generated:
        st.caption("This generated prompt is based on uploaded notes; matching scores are only available for historical questions.")
    else:
        sh.similarity_caption()
        if not matches or q.get("best_tier") == "Unmatched":
            ui.notice("warning", "No section reached the weak-match threshold",
                      "Either the lecture material does not cover this question, or it uses different vocabulary. Check your notes for it, or add material to the subject."
                      + (" The sections below are the closest by vocabulary and are shown for reference only; they are not matches." if matches else ""))
        for m in matches:
            _match_card(svc, record, q, m)

    # Evidence Table & CSV Download
    c_dl1, c_dl2 = st.columns([3, 1])
    c_dl1.caption(f"Showing {len(shown)} of {len(questions)} question(s).")
    if not is_generated and hasattr(rx, "practice_questions_csv"):
        c_dl2.download_button("Export Questions (CSV)", rx.practice_questions_csv(record, {"id": meta["id"], "created_at": meta["created_at"]}),
                              file_name=f"syllabuslens_{sh.slug(subject.name)}_{meta['id']}_questions.csv", mime="text/csv", key=f"ex_dl_csv_{meta['id']}")

    if not is_generated:
        with st.expander("Evidence table for all questions (searchable)"):
            rows = [{"Paper": x.get("paper_label", ""), "Question": x.get("display_id", ""), "Category": x.get("category_name", ""),
                     "Tier": x.get("best_tier", ""), "Rank": m["rank"], "Section": m["heading"],
                     "Document": m["doc_name"], "Location": m["locator"], "Similarity": m["score"], "Shared terms": ", ".join(m["key_terms"])}
                    for x in shown for m in x.get("matches", [])]
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True) if rows else st.caption("No matches to list.")

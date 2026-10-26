"""Study Planner: rule-based revision priorities with the evidence behind each one."""
from __future__ import annotations

import streamlit as st

from components import nav, record_charts as rc, state, ui
from pages_content import _shared as sh
from services import record_export as rx
from services.study_plan_service import RULES, build_plan

TOPIC_NOTE = {
    "not_detected": "Topic words occur in your notes but no selected question is similar enough.",
    "not_assessable": "Half or fewer of the topic's words occur in your notes, so this tool cannot assess it reliably. That is not evidence the topic is absent from exams.",
}


def _toggle(svc, subject_id: str, section_id: str, key: str) -> None:
    if st.session_state.get(key):
        svc.history.mark_reviewed(subject_id, section_id)
    else:
        svc.history.unmark_reviewed(subject_id, section_id)


def _standalone_notes(svc, subject) -> None:
    """Let students read source-grounded notes before running a paper analysis."""
    st.subheader("Topic-wise Revision Notes")
    if not state.subject_overview(svc, subject)["notes"]:
        ui.empty_state("Upload study notes to get started", "Add lecture notes or slides in the Document Library.")
        if st.button("Open Document Library", key=f"plan_notes_library_{subject.id}"):
            nav.goto("library")
        return

    notes = svc.topic_notes.build_topic_notes(subject.id)
    if not notes:
        ui.empty_state("No topic notes available", "The uploaded study documents do not contain indexed sections.")
        return

    selected = st.selectbox(
        "Topic",
        list(range(len(notes))),
        format_func=lambda i: notes[i].topic,
        key=f"plan_standalone_topic_{subject.id}",
    )
    st.markdown(svc.topic_notes.export_notes_markdown([notes[selected]], subject.name))
    if notes[selected].diagram:
        diagram = notes[selected].diagram
        st.caption(f"{diagram.caption} (Deterministic structural model; not image extraction.)")
        st.graphviz_chart(diagram.dot_source, width="stretch")
    st.download_button(
        "Download all topic notes (Markdown)",
        svc.topic_notes.export_notes_markdown(notes, subject.name),
        file_name=f"syllabuslens_{sh.slug(subject.name)}_revision_notes.md",
        mime="text/markdown",
        key=f"plan_standalone_notes_dl_{subject.id}",
    )
    ui.notice("info", "Next: analyze past papers", "Question Analysis adds historical question links and coverage-based revision priorities.")
    if st.button("Open Question Analysis", key=f"plan_notes_analysis_{subject.id}"):
        nav.goto("analysis")


def render() -> None:
    svc = state.get_services()
    ui.page_header("Study Planner", "Read source-grounded topic notes, then review analysis-backed revision priorities.")
    subject = state.require_subject(svc)
    state.show_scope(svc, subject)
    if not svc.history.list(subject.id, limit=1):
        _standalone_notes(svc, subject)
        return

    loaded = sh.analysis_picker(svc, subject, key="plan_pick")
    if not loaded:
        return
    meta, record = loaded
    sh.staleness_notice(svc, meta)
    ui.notice("info", "How to read this plan", "Each item comes from a fixed rule applied to the papers you selected (shown with its evidence). "
              "It reflects what those past papers contained. It does not predict future exams or guarantee marks.")
    reviewed = svc.history.reviewed_ids(subject.id)
    plan = build_plan(record, reviewed)
    items = [r for r in plan if r["kind"] == "section"]
    done = sum(1 for r in items if r["reviewed"])
    ui.metric_grid([("Revision items", len(items), "Sections suggested by the rules"),
                    ("Reviewed", f"{done}/{len(items)}" if items else "0", "Ticked by you; saved per subject"),
                    ("Questions needing notes", sum(1 for r in plan if r["rule_id"] == "R4"), "Unmatched questions (rule R4)", "danger" if any(r["rule_id"] == "R4" for r in plan) else "")])
    t_notes, t_list, t_topics, t_rules = st.tabs(["Topic-wise Revision Notes", "Revision checklist", "Syllabus gaps", "Rules used"])

    with t_notes:
        topic_notes = record.get("topic_revision_notes")
        if not topic_notes and hasattr(svc, "topic_notes"):
            raw_notes = svc.topic_notes.build_topic_notes(subject.id, [d["document_id"] for d in record["documents"]])
            topic_notes = [n.to_dict() for n in raw_notes]

        if not topic_notes:
            ui.empty_state("No topic revision notes available", "Upload lecture notes or configure topics on the Subjects page to generate topic revision notes.")
        else:
            all_topic_notes = topic_notes
            chosen_topic = st.selectbox(
                "Topic",
                list(range(len(all_topic_notes))),
                format_func=lambda i: all_topic_notes[i].get("topic", "Topic"),
                key=f"plan_note_topic_{subject.id}_{meta['id']}",
            )
            topic_notes = [all_topic_notes[chosen_topic]]
            c_meta1, c_meta2 = st.columns([3, 1])
            c_meta1.caption(f"Topic {chosen_topic + 1} of {len(all_topic_notes)}; synthesized from {len(record['documents'])} study document(s). Source citations preserved.")
            md_content = rx.topic_notes_markdown(record, {"id": meta["id"], "created_at": meta["created_at"]})
            c_meta2.download_button(
                "Download Notes (MD)",
                md_content,
                file_name=f"syllabuslens_{sh.slug(subject.name)}_{meta['id']}_revision_notes.md",
                mime="text/markdown",
                key=f"dl_notes_{meta['id']}",
            )

            for idx, tn in enumerate(topic_notes):
                status_tone = "success" if tn.get("status") == "covered" else ("warning" if tn.get("status") == "partially_covered" else "danger")
                st.markdown(f"#### {tn.get('topic')} {ui.badge(tn.get('status', '').replace('_', ' ').title(), status_tone)}", unsafe_allow_html=True)
                if tn.get("summary"):
                    st.info(tn["summary"])

                col_a, col_b = st.columns(2)
                with col_a:
                    if tn.get("definitions"):
                        with st.expander(f"Key Definitions ({len(tn['definitions'])})", expanded=True):
                            for d in tn["definitions"]:
                                ui.render(f'<div class="sl-card" style="margin-bottom:.4rem"><strong>{ui.esc(d["text"])}</strong>'
                                          f'<div class="sl-card-meta">{ui.badge("Source", "primary")} <span>{ui.esc(d["source"])}</span></div></div>')

                    if tn.get("formulas"):
                        with st.expander(f"Formulas & Equations ({len(tn['formulas'])})", expanded=True):
                            for f in tn["formulas"]:
                                ui.render(f'<div class="sl-card" style="margin-bottom:.4rem"><code>{ui.esc(f["text"])}</code>'
                                          f'<div class="sl-card-meta">{ui.badge("Source", "primary")} <span>{ui.esc(f["source"])}</span></div></div>')

                    if tn.get("derivations_examples"):
                        with st.expander(f"Derivations & Examples ({len(tn['derivations_examples'])})"):
                            for de in tn["derivations_examples"]:
                                ui.render(f'<div class="sl-card" style="margin-bottom:.4rem">{ui.esc(de["text"])}'
                                          f'<div class="sl-card-meta">{ui.badge("Source", "primary")} <span>{ui.esc(de["source"])}</span></div></div>')

                with col_b:
                    if tn.get("explanations"):
                        with st.expander(f"Explanations & Algorithms ({len(tn['explanations'])})", expanded=True):
                            for exp in tn["explanations"]:
                                ui.render(f'<div class="sl-card" style="margin-bottom:.4rem">{ui.esc(exp["text"])}'
                                          f'<div class="sl-card-meta">{ui.badge("Source", "primary")} <span>{ui.esc(exp["source"])}</span></div></div>')

                    if tn.get("pros_cons_limitations"):
                        with st.expander(f"Advantages & Limitations ({len(tn['pros_cons_limitations'])})"):
                            for pcl in tn["pros_cons_limitations"]:
                                ui.render(f'<div class="sl-card" style="margin-bottom:.4rem">{ui.esc(pcl["text"])}'
                                          f'<div class="sl-card-meta">{ui.badge("Source", "primary")} <span>{ui.esc(pcl["source"])}</span></div></div>')

                    if tn.get("missing_material"):
                        with st.expander(f"Missing Material Warnings ({len(tn['missing_material'])})", expanded=True):
                            for mm in tn["missing_material"]:
                                ui.notice("warning", "Missing from Notes", mm)

                # Concept Diagram
                if hasattr(svc, "diagrams"):
                    diag = svc.diagrams.get_diagram_for_text(tn.get("topic", ""))
                    if diag:
                        with st.expander(f"Concept Diagram: {diag.title}"):
                            st.caption(f"{diag.caption} (Deterministic structural model; not OCR/image extraction)")
                            st.graphviz_chart(diag.dot_source)

                if tn.get("sources"):
                    with st.expander("Contributing Source Passages"):
                        for s in tn["sources"]:
                            ui.render(f'<div class="sl-card-meta">{ui.badge("Reference", "neutral")} <span>{ui.esc(s)}</span></div>')

                st.divider()

    with t_list:
        if not items:
            ui.empty_state("No revision items from the rules", "No section is repeatedly or strongly matched in this analysis.")
        for i, r in enumerate(items):
            rkey = f"rev_{subject.id}_{meta['id']}_{i}_{int(r['reviewed'])}"
            st.checkbox(f"{r['title']}", value=r["reviewed"], key=rkey, on_change=_toggle, args=(svc, subject.id, r["section_id"], rkey))
            ui.render(f'<div class="sl-card-meta" style="margin:-.3rem 0 .2rem 1.7rem">{ui.badge(r["rule_id"], "primary", r["rule"])}<span>{ui.esc(r.get("source", ""))}</span></div>')
            with st.expander("Evidence"):
                st.caption(f"Rule {r['rule_id']}: {r['rule']}")
                for e in r["evidence"]:
                    ui.render(f'<div class="sl-card-meta">{ui.esc(e)}</div>')
        qs = [r for r in plan if r["kind"] == "question"]
        if qs:
            st.markdown("**Questions with no match in your notes**")
            for r in qs:
                ui.notice("warning", r["title"], r["evidence"][0] + " The material may be missing from your notes.")
        st.download_button("Download checklist (CSV)", rx.checklist_csv(record, reviewed, {"id": meta["id"], "created_at": meta["created_at"]}),
                           file_name=f"syllabuslens_{sh.slug(subject.name)}_{meta['id']}_checklist.csv", mime="text/csv", key=f"plan_dl_{meta['id']}")
    with t_topics:
        topics = record.get("topic_coverage") or []
        if not topics:
            ui.empty_state("No syllabus topics to assess", "Add topics to this subject on the Subjects page, then run a new analysis.")
        else:
            fig = rc.topic_coverage_bars(record)
            st.plotly_chart(fig, width="stretch", key=f"plan_topic_{meta['id']}") if fig else None
            for status in ("not_detected", "not_assessable"):
                rows = [t for t in topics if t["status"] == status]
                if rows:
                    st.markdown(f"**{rc.TOPIC_LABELS[status]}** ({len(rows)})")
                    st.caption(TOPIC_NOTE[status])
                    for t in rows:
                        extra = f" Closest notes section: {t['best_section_heading']}." if t.get("best_section_heading") and status == "not_detected" else ""
                        ui.render(f'<div class="sl-card-meta"><strong>{ui.esc(t["topic"])}</strong>{ui.esc(extra)}</div>')
    with t_rules:
        for rid, text in RULES.items():
            ui.render(f'<div class="sl-card-meta">{ui.badge(rid, "primary")} {ui.esc(text)}</div>')
        st.caption("Rule thresholds use the analysis's match tiers, which are adjustable heuristics.")

"""Refactored Streamlit UI for NeuroGuard
- Uses environment variables for backend endpoints
- Replaces synthetic QC snippet with explicit placeholder (no fabricated EEG)
- Introduces SeverityLevel Enum and SEVERITY_CONFIG mapping
- Adds type hints and docstrings
- Catches specific exceptions for network calls
- Preserves the original layout & behavior where possible; falls back to legacy string-matching if API doesn't provide structured codes
"""

import os
import hashlib
import requests
import streamlit as st
import plotly.graph_objects as go
import numpy as np
from datetime import datetime
import base64
import uuid
import io
from typing import Any, Dict, Optional
from enum import Enum
from functools import partial

# ReportLab Imports for PDF Generation
from reportlab.lib.pagesizes import LETTER
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

# -----------------------
# Configuration
# -----------------------

# Use environment variables for endpoints so the UI is deployable/configurable
FLASK_PREDICT_URL = os.environ.get("FLASK_PREDICT_URL", "http://127.0.0.1:5000/predict_adaptive")
FLASK_CALIBRATE_URL = os.environ.get("FLASK_CALIBRATE_URL", "http://127.0.0.1:5000/calibrate_multi")

# UI constants
COLORS = {
    "CRITICAL": "#FF2A2A",
    "WARNING": "#FF9900",
    "NOTICE": "#007AFF",
    "GOOD": "#32D74B",
    "BG_MAIN": "#0E1117",
    "BG_CARD": "#1E1F25",
    "TEXT_MAIN": "#FFFFFF",
    "TEXT_MUTED": "#A0A0A0",
    "ACCENT": "#4A90E2"
}

st.set_page_config(
    page_title="NeuroGuard Clinical | v3.1",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

if 'session_id' not in st.session_state:
    st.session_state['session_id'] = str(uuid.uuid4())[:8].upper()
if 'step' not in st.session_state:
    st.session_state['step'] = 1

# -----------------------
# Structured severity mapping
# -----------------------
class SeverityLevel(Enum):
    NORMAL = 0
    WARNING = 1
    CRITICAL = 2

SEVERITY_CONFIG: Dict[SeverityLevel, Dict[str, Any]] = {
    SeverityLevel.CRITICAL: {
        "color": COLORS['CRITICAL'],
        "level": "SEIZURE DETECTED",
        "icon": "🔴",
        "code": "RG-1C",
        "action": "IMMEDIATE REVIEW REQUIRED"
    },
    SeverityLevel.WARNING: {
        "color": COLORS['WARNING'],
        "level": "POSSIBLE / ANOMALY",
        "icon": "🟠",
        "code": "RG-2B",
        "action": "Physician Review Recommended"
    },
    SeverityLevel.NORMAL: {
        "color": COLORS['GOOD'],
        "level": "NORMAL BASELINE",
        "icon": "🟢",
        "code": "RG-4N",
        "action": "Routine Monitoring"
    }
}

# -----------------------
# Styling (unchanged visually)
# -----------------------
st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&family=IBM+Plex+Mono:wght@500&display=swap');
    .stApp {{ background-color: {COLORS['BG_MAIN']}; font-family: 'Inter', sans-serif; }}
    .clinical-card {{ background-color: {COLORS['BG_CARD']}; border-radius:8px; padding:24px; border:1px solid rgba(255,255,255,0.08); margin-bottom:20px; }}
    h1,h2,h3 {{ font-family:'Inter',sans-serif; letter-spacing:-0.5px; }}
    .metric-value {{ font-family: 'IBM Plex Mono', monospace; font-size:32px; font-weight:500; color:#fff; }}
    .metric-label {{ font-size:13px; color:{COLORS['TEXT_MUTED']}; text-transform:uppercase; letter-spacing:1.2px; margin-bottom:8px; }}
    .alert-banner {{ padding:32px 24px; border-radius:8px; margin-bottom:24px; display:flex; align-items:center; justify-content:space-between; box-shadow: 0 8px 24px rgba(0,0,0,0.3); }}
    .alert-title {{ font-family:'Inter',sans-serif; font-weight:800; font-size:32px; margin:0; letter-spacing:-1px; }}
    .breadcrumb {{ display:flex; font-family:'IBM Plex Mono',monospace; font-size:12px; color:{COLORS['TEXT_MUTED']}; margin-bottom:20px; }}
    .sim-badge {{ background-color: rgba(0, 90, 156, 0.2); color: {COLORS['NOTICE']}; padding:4px 8px; border-radius:4px; font-size:10px; font-family:'IBM Plex Mono'; border:1px solid {COLORS['NOTICE']}; display:inline-block; margin-bottom:10px; }}
    .stButton>button {{ width:100%; height:3.5em; font-weight:700; border-radius:6px; }}
    </style>
    """, unsafe_allow_html=True)

# -----------------------
# Utility functions
# -----------------------

def parse_confidence(conf: Any) -> float:
    """Parse a variety of confidence formats into 0..1 float.

    Accepts numeric 0..1, numeric 0..100, strings like '85%' or named levels.
    Returns 0.0 on parse failure.
    """
    if conf is None:
        return 0.0
    try:
        if isinstance(conf, (int, float, np.floating, np.integer)):
            f = float(conf)
            return min(max(f / 100.0, 0.0), 1.0) if f > 1.0 else min(max(f, 0.0), 1.0)
        s = str(conf).strip()
        if s == "":
            return 0.0
        if s.endswith('%'):
            return min(max(float(s.rstrip('%')) / 100.0, 0.0), 1.0)
        f = float(s)
        return min(max(f / 100.0, 0.0), 1.0) if f > 1.0 else min(max(f, 0.0), 1.0)
    except (ValueError, TypeError):
        mapping = {"HIGH": 0.85, "MODERATE": 0.5, "LOW": 0.15, "NONE": 0.0}
        return mapping.get(str(conf).upper(), 0.0)


def determine_severity(prediction_code: Optional[str], prediction_text: Optional[str], confidence: float) -> Dict[str, Any]:
    """Return a UI-friendly severity dict.

    Preferred: backend returns 'pred_code' string like 'ICTAL_EVENT', 'NO_SEIZURE', etc.
    Fallback: use legacy string-matching on prediction_text for compatibility.
    """
    # First try to use structured code from server
    if prediction_code:
        code = prediction_code.upper()
        # Map known backend codes to SeverityLevel
        if code in ("ICTAL_EVENT", "SEIZURE", "EPILEPTIFORM"):
            level = SeverityLevel.CRITICAL if confidence >= 0.75 else SeverityLevel.WARNING
        elif code in ("ANOMALY", "ARTIFACT"):
            level = SeverityLevel.WARNING
        else:
            level = SeverityLevel.NORMAL
    else:
        # Backwards-compatible fallback to text matching
        txt = (prediction_text or "").upper()
        if "NO SEIZURE" in txt or "NORMAL" in txt:
            level = SeverityLevel.NORMAL
        elif any(k in txt for k in ("SEIZURE", "ICTAL", "EPILEPTIFORM", "SPIKE")):
            level = SeverityLevel.CRITICAL if confidence >= 0.75 else SeverityLevel.WARNING
        elif confidence >= 0.80:
            level = SeverityLevel.WARNING
        else:
            level = SeverityLevel.NORMAL

    cfg = SEVERITY_CONFIG[level]
    return {
        'color': cfg['color'],
        'level': cfg['level'],
        'icon': cfg['icon'],
        'code': cfg['code'],
        'action': cfg['action']
    }


def metric_card(col: Any, label: str, value: str, subtext: Optional[str] = None) -> None:
    with col:
        st.markdown(f"""
        <div class="clinical-card" style="padding: 20px; text-align: left;">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            {f'<div style="font-size:12px; color:#888; margin-top:6px;">{subtext}</div>' if subtext else ''}
        </div>
        """, unsafe_allow_html=True)


def create_clinical_pdf(patient_id: str, session_id: str, ui_state: Dict[str, Any], metrics: Dict[str, str], notes: str, fig_bytes: Optional[bytes] = None) -> io.BytesIO:
    """Create PDF report (keeps original layout, minimal change)."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=LETTER, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=72)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='Header', fontSize=18, leading=22, spaceAfter=12, textColor=colors.HexColor(COLORS['BG_MAIN'])))
    story = []
    story.append(Paragraph(f"NeuroGuard Clinical Report", styles['Header']))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')} | Session: {session_id}", styles['Normal']))
    story.append(Spacer(1, 12))
    risk_color = colors.HexColor(ui_state['color'])
    status_text = f"<b>STATUS: {ui_state['level']}</b><br/>RISK CODE: {ui_state['code']}"
    data = [[f"Patient ID: {patient_id}", Paragraph(status_text, styles['Normal'])], [f"File: {st.session_state.get('analyzed_file', 'N/A')}", ""]]
    t = Table(data, colWidths=[3.5*inch, 3*inch])
    t.setStyle(TableStyle([('TEXTCOLOR', (0,0), (-1,-1), colors.black), ('LINEBELOW', (0,1), (-1,1), 1, colors.lightgrey), ('TEXTCOLOR', (1,0), (1,0), risk_color)]))
    story.append(t)
    story.append(Spacer(1, 24))
    metric_data = [["Metric", "Value", "Metric", "Value"], ["Duration", f"{metrics['duration']}", "Peak Prob", f"{metrics['peak_prob']}"], ["Total Epochs", f"{metrics['epochs']}", "Burst Count", f"{metrics['bursts']}"]]
    mt = Table(metric_data, colWidths=[1.5*inch, 1.5*inch, 1.5*inch, 1.5*inch])
    mt.setStyle(TableStyle([('BACKGROUND', (0,0), (-1,0), colors.HexColor('#f0f0f0')), ('GRID', (0,0), (-1,-1), 0.5, colors.lightgrey)]))
    story.append(mt)
    story.append(Spacer(1, 24))
    if fig_bytes:
        try:
            img = RLImage(io.BytesIO(fig_bytes))
            aspect = img.imageHeight / float(img.imageWidth)
            img.drawHeight = 6 * inch * aspect
            img.drawWidth = 6 * inch
            story.append(img)
        except Exception:
            pass
    story.append(Spacer(1, 12))
    story.append(Paragraph(f"Notes: {notes if notes else 'None'}", styles['Normal']))
    doc.build(story)
    buffer.seek(0)
    return buffer

# -----------------------
# Networking helpers with robust error handling
# -----------------------

def _file_params_hash(file_bytes: bytes, params: Dict[str, Any]) -> str:
    """Create a stable cache key for a file + params combination."""
    h = hashlib.sha256()
    h.update(file_bytes)
    # sort params for determinism
    for k in sorted(params.keys()):
        h.update(str(k).encode())
        h.update(str(params[k]).encode())
    return h.hexdigest()

@st.cache_data(show_spinner=False)
def send_predict_request_cached(key: str, url: str, files_payload: Dict[str, Any], data_payload: Dict[str, Any]) -> Dict[str, Any]:
    """Cached wrapper around the network call. Key must be computed by caller.

    This function is intentionally minimal and raises requests exceptions to be handled by caller.
    """
    resp = requests.post(url, files=files_payload, data=data_payload, timeout=120)
    resp.raise_for_status()
    return resp.json()


def send_predict_request(file_bytes: bytes, filename: str, patient_id: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Send file to the prediction endpoint with robust error handling and caching.

    Returns a dict (parsed JSON) or raises an exception (requests.RequestException).
    """
    key = _file_params_hash(file_bytes, {**params, 'patient_id': patient_id, 'filename': filename})
    files_payload = {'file': (filename, file_bytes, 'application/octet-stream')}
    data_payload = {**params, 'patient_id': patient_id}
    try:
        return send_predict_request_cached(key, FLASK_PREDICT_URL, files_payload, data_payload)
    except requests.exceptions.RequestException as e:
        # re-raise to be handled upstream with friendly UI
        raise

# -----------------------
# UI Layout & Logic (kept similar, but safer)
# -----------------------

with st.sidebar:
    st.title("NeuroGuard")
    st.caption(f"Clinical Suite")
    st.markdown("---")
    app_mode = st.radio("MODULE", ["Clinical Analysis", "Calibration"], label_visibility="collapsed")
    st.markdown("---")
    st.markdown("**PATIENT CONTEXT**")
    patient_id = st.text_input("Patient ID", value="chb01")
    with st.expander("🔧 TECHNICIAN CONFIG"):
        st.caption("Warning: Modifying defaults affects sensitivity.")
        ma_window = st.slider("Smoothing (Windows)", 1, 21, 5)
        peak_req = st.slider("Trigger Threshold", 0.5, 0.99, 0.80)
        min_duration = st.slider("Min Duration (s)", 2.0, 30.0, 6.0)
        mad_k = st.slider("Calibration Multiplier (k)", 2.0, 12.0, 6.0, step=0.5)
        show_qc_snippet = st.checkbox("Show QC snippet when no event detected (placeholder only)", value=False)


if app_mode == "Clinical Analysis":

    def render_breadcrumbs(step: int) -> None:
        steps = ["UPLOAD", "ANALYZE", "REVIEW"]

        st.markdown(
            """
            <style>
            .breadcrumb {
                display: flex;
                justify-content: space-between;
                margin: 10px 0 25px 0;
                padding: 0 10px;
            }
            .breadcrumb-item {
                flex: 1;
                text-align: center;
                color: #9CA3AF;
                font-weight: 600;
                padding: 10px 6px;
                border-radius: 8px;
                background: rgba(255,255,255,0.04);
                border: 1px solid rgba(255,255,255,0.08);
                margin: 0 5px;
                letter-spacing: 0.5px;
            }
            .breadcrumb-active {
                background: linear-gradient(90deg,#06b6d4,#0ea5a4);
                color: white !important;
                border: none !important;
                box-shadow: 0 4px 10px rgba(14,165,164,0.25);
                font-weight: 700;
            }
            </style>
            """,
            unsafe_allow_html=True
        )

        html = '<div class="breadcrumb">'
        for i, s in enumerate(steps, start=1):
            active_class = "breadcrumb-active" if i == step else ""
            html += f'<div class="breadcrumb-item {active_class}">{i}. {s}</div>'
        html += "</div>"

        st.markdown(html, unsafe_allow_html=True)

    render_breadcrumbs(st.session_state['step'])
    col_up, col_btn = st.columns([3, 1])
    with col_up:
        uploaded_file = st.file_uploader("Select Daily EEG File (.edf)", type=["edf"])
        if uploaded_file:
            st.session_state['step'] = 2
    with col_btn:
        st.write("")
        st.write("")
        analyze_btn = st.button("RUN ANALYSIS", type="primary", disabled=not uploaded_file)

    if analyze_btn and uploaded_file:
        with st.spinner("Processing Signal..."):
            try:
                file_bytes = uploaded_file.getvalue()
                params = {'ma_window': ma_window, 'peak_req': peak_req, 'min_duration_s': min_duration}
                try:
                    res = send_predict_request(file_bytes, uploaded_file.name, patient_id, params)
                except requests.exceptions.RequestException as e:
                    st.error(f"Network error while contacting prediction service: {e}")
                    res = None

                if res is not None:
                    st.session_state['result'] = res
                    st.session_state['analyzed_file'] = uploaded_file.name
                    st.session_state['analysis_time'] = datetime.now().strftime("%H:%M:%S")
                    st.session_state['step'] = 3

                    # Morphology handling: keep backend morphology if provided, otherwise DO NOT synthesize.
                    morph = res.get('morphology')
                    if morph and isinstance(morph, dict) and 'y' in morph and 'x' in morph:
                        st.session_state['morphology_x'] = np.array(morph['x'])
                        st.session_state['morphology_y'] = np.array(morph['y'])
                        st.session_state.pop('morph_message', None)
                    else:
                        # No morphology available. Do NOT produce synthetic EEG.
                        st.session_state.pop('morphology_x', None)
                        st.session_state.pop('morphology_y', None)
                        st.session_state['morph_message'] = (
                            "No event morphology extracted — backend returned no waveform. "
                            "Do not assume lead connectivity or normal EEG based on a generated plot."
                        )
                else:
                    st.error("Analysis Failed: No response from prediction service")
            except Exception as e:
                # catch unexpected issues but avoid blanket excepts earlier
                st.error(f"System Error: {e}")

    if 'result' in st.session_state:
        res = st.session_state['result']

        # Prefer structured fields where available
        pred_code = res.get('pred_code') or res.get('prediction_code') or None
        overall_text = res.get('overall_prediction')
        conf_val = parse_confidence(res.get('confidence', 0))

        ui_state = determine_severity(pred_code, overall_text, conf_val)

        hex_color = ui_state['color'].lstrip('#')
        r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)

        st.markdown(f"""
        <div class="alert-banner" style="background-color: rgba({r}, {g}, {b}, 0.25); border-left: 12px solid {ui_state['color']};">
            <div>
                <h2 class="alert-title" style="color: {ui_state['color']};">
                    {ui_state['icon']} {ui_state['level']}
                </h2>
                <div style="margin-top:8px; font-family: 'IBM Plex Mono'; font-size: 14px; color: #fff; font-weight: bold;">
                    CODE: {ui_state['code']} | CONFIDENCE: {conf_val*100:.1f}%
                </div>
            </div>
            <div style="text-align:right;">
                <div style="font-size:12px; color:#aaa; letter-spacing: 1px; font-weight: bold;">SUGGESTED ACTION</div>
                <div style="font-size:20px; font-weight:800; color: #fff; text-transform: uppercase;">
                    {ui_state['action']}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div style="display:flex; justify-content:space-between; margin-bottom:24px; padding: 0 5px; font-size:12px; color:#888; font-family:'IBM Plex Mono';">
            <span>FILE: {st.session_state.get('analyzed_file')}</span>
            <span>ANALYZED: {st.session_state.get('analysis_time')}</span>
            <span>SESSION: {st.session_state['session_id']}</span>
        </div>
        """, unsafe_allow_html=True)

        m1, m2, m3, m4 = st.columns(4)
        peak_prob_pct = parse_confidence(res.get('peak_probability', 0)) * 100.0
        metric_card(m1, "Event Duration", f"{float(res.get('detection_span_seconds', 0)):.1f}s", "Continuous Span")
        metric_card(m2, "Peak Probability", f"{peak_prob_pct:.1f}%", "Max Detected")
        metric_card(m3, "Windows Analyzed", f"{int(res.get('windows_analyzed', 0)):,}", "Total 4s Epochs")
        metric_card(m4, "Bursts > Threshold", f"{int(res.get('significant_bursts', 0))}", f"Limit: {peak_req}")

        c_chart, c_morph = st.columns([2, 1])

        with c_chart:
            st.markdown("### Probability Timeline")
            prob_data = res.get('probability_timeline', [])
            fig_time = go.Figure()
            fig_time.add_trace(go.Scatter(
                y=prob_data,
                fill='tozeroy',
                line=dict(color=ui_state['color'], width=3),
                hoverinfo='y'
            ))
            fig_time.add_hline(y=peak_req, line_dash="dot", line_color="white", opacity=0.6)
            fig_time.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                height=300, margin=dict(l=0, r=0, t=10, b=0),
                yaxis=dict(range=[0, 1.1], showgrid=True, gridcolor='rgba(255,255,255,0.05)'),
                xaxis=dict(showgrid=False),
                showlegend=False
            )
            st.plotly_chart(fig_time, use_container_width=True)

        with c_morph:
            st.markdown("### Event Morphology")

            # Decide if a seizure was detected.
            overall = str(res.get('overall_prediction', '')).upper() if res.get('overall_prediction') else ''
            confidence_val = parse_confidence(res.get('confidence', 0))
            morph = res.get('morphology')  # may be None or a dict with 'x','y'

            # Use structured pred_code if available to decide
            pred_code = res.get('pred_code') or res.get('prediction_code') or None
            is_seizure = False
            if pred_code:
                is_seizure = pred_code.upper() in ("ICTAL_EVENT", "SEIZURE", "EPILEPTIFORM") or (confidence_val >= peak_req)
            else:
                # legacy fallback
                is_seizure = ("SEIZURE" in overall) or (confidence_val >= peak_req)

            if is_seizure and morph and isinstance(morph, dict) and 'x' in morph and 'y' in morph:
                st.markdown('<div class="sim-badge">🧪 REAL EVENT WAVEFORM</div>', unsafe_allow_html=True)
                mx = np.array(morph['x'])
                my = np.array(morph['y'])

                fig_eeg = go.Figure()
                fig_eeg.add_trace(go.Scatter(x=mx, y=my, line=dict(color=ui_state['color'], width=2)))
                fig_eeg.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    height=260, margin=dict(l=0, r=0, t=0, b=0),
                    xaxis=dict(title='Time (s)' if len(mx)>1 else '', showgrid=False),
                    yaxis=dict(title='' , showgrid=False),
                    showlegend=False
                )
                st.plotly_chart(fig_eeg, use_container_width=True)

            else:
                # Non-seizure -> explicit message. Do NOT synthesize a plausible EEG.
                st.markdown('<div class="sim-badge">🧪 NO EVENT MORPHOLOGY</div>', unsafe_allow_html=True)
                st.markdown(
                    "<div style='padding:12px; background-color: rgba(255,255,255,0.02); border-radius:8px; color:#BFC7CE;'>"
                    "<b>No event morphology available</b><br>"
                    "No seizure was detected in this recording (or no morphology could be extracted)."
                    "</div>",
                    unsafe_allow_html=True
                )

                # If technician explicitly asked for a QC placeholder, show a clear non-deceptive placeholder box
                if show_qc_snippet:
                    st.markdown("")
                    st.markdown(
                        "<div style='padding:18px; border-radius:8px; background:#111; border:1px dashed rgba(255,255,255,0.04);'>"
                        "<div style='color:#DDD; font-size:13px; font-weight:600;'>QC Placeholder</div>"
                        "<div style='color:#AAA; font-size:12px; margin-top:6px;'>Backend did not return waveform data. This is a placeholder indicating absence of morphology — please verify lead connections and raw data if needed.</div>"
                        "</div>",
                        unsafe_allow_html=True
                    )
                else:
                    st.caption("If you'd like to inspect signal quality, enable 'Show QC snippet' in Technician Config.")

        st.markdown("---")
        notes = st.text_area("Physician Notes", placeholder=f"Enter clinical observations...")

        col_gen, col_dl = st.columns([1, 2])
        metrics_dict = {
            "duration": f"{float(res.get('detection_span_seconds', 0)):.1f}s",
            "peak_prob": f"{peak_prob_pct:.1f}%",
            "epochs": f"{int(res.get('windows_analyzed', 0)):,}",
            "bursts": f"{int(res.get('significant_bursts', 0))}"
        }

        with col_gen:
            if st.button("📄 GENERATE PDF REPORT"):
                with st.spinner("Rendering PDF..."):
                    try:
                        fig_bytes = fig_time.to_image(format="png", width=1200, height=500, scale=2)
                    except Exception:
                        fig_bytes = None
                    pdf_buffer = create_clinical_pdf(patient_id, st.session_state['session_id'], ui_state, metrics_dict, notes, fig_bytes)
                    st.session_state['pdf_buffer'] = pdf_buffer
                    st.success("Report Ready!")

        with col_dl:
            if 'pdf_buffer' in st.session_state:
                b64_pdf = base64.b64encode(st.session_state['pdf_buffer'].getvalue()).decode()
                href = f'<a href="data:application/pdf;base64,{b64_pdf}" download="NeuroGuard_Report.pdf" style="text-decoration:none;"><button style="background-color: {COLORS["ACCENT"]}; color: white; border: none; padding: 10px 20px; border-radius: 5px; font-weight: bold; cursor: pointer; width: 100%;">📥 DOWNLOAD PDF</button></a>'
                st.markdown(href, unsafe_allow_html=True)


elif app_mode == "Calibration":
    st.markdown("## Baseline Calibration")
    st.info("Upload 2+ non-seizure files to establish patient baseline thresholds.")
    calib_files = st.file_uploader("Upload Baseline Files", type=["edf"], accept_multiple_files=True)
    if st.button("RUN CALIBRATION", type="primary"):
        if not calib_files or len(calib_files) < 2:
            st.error("Requirement: At least 2 files needed.")
        else:
            with st.spinner(f"Calibrating NeuroGuard for {patient_id}..."):
                try:
                    files_list = [('files[]', (f.name, f.getvalue(), 'application/octet-stream')) for f in calib_files]
                    data = {'patient_id': patient_id, 'mad_k': mad_k}
                    try:
                        resp = requests.post(FLASK_CALIBRATE_URL, files=files_list, data=data, timeout=120)
                        resp.raise_for_status()
                        st.balloons()
                        st.json(resp.json())
                    except requests.exceptions.RequestException as e:
                        st.error(f"Calibration request failed: {e}")
                except Exception as e:
                    st.error(f"Error: {e}")

# End of file

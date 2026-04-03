"""Refactored Streamlit UI for NeuroGuard - Tech Event Edition (Fixed Scoping)
- Includes "Glassmorphism" UI
- Includes "Theatrical" loading sequences
- Includes Clinical Triage Badges (STAT/URGENT/ROUTINE)
- Updates terminology to "NO SEIZURE DETECTED"
- FIXED: Patient ID scope error in Calibration mode
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
import time
from typing import Any, Dict, Optional
from enum import Enum

# ReportLab Imports for PDF Generation
from reportlab.lib.pagesizes import LETTER
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

# -----------------------
# Configuration
# -----------------------

# Use environment variables for endpoints
FLASK_PREDICT_URL = os.environ.get("FLASK_PREDICT_URL", "http://127.0.0.1:5000/predict_adaptive")
FLASK_CALIBRATE_URL = os.environ.get("FLASK_CALIBRATE_URL", "http://127.0.0.1:5000/calibrate_multi")

# UI constants
COLORS = {
    "CRITICAL": "#FF2A2A",
    "WARNING": "#FF9900",
    "NOTICE": "#007AFF",
    "GOOD": "#32D74B",
    "BG_MAIN": "#050505", 
    "BG_CARD": "#1E1F25",
    "TEXT_MAIN": "#FFFFFF",
    "TEXT_MUTED": "#A0A0A0",
    "ACCENT": "#00C9FF"   
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
# Structured severity mapping (UPDATED WITH CLINICAL CATEGORIES)
# -----------------------
class SeverityLevel(Enum):
    NORMAL = 0
    WARNING = 1
    CRITICAL = 2

SEVERITY_CONFIG: Dict[SeverityLevel, Dict[str, Any]] = {
    SeverityLevel.CRITICAL: {
        "color": COLORS['CRITICAL'],
        "level": "SEIZURE DETECTED",
        "category_label": "CAT 1: STAT",  # <-- Clinical Triage Badge
        "icon": "🔴",
        "code": "RG-1C",
        "action": "IMMEDIATE REVIEW REQUIRED"
    },
    SeverityLevel.WARNING: {
        "color": COLORS['WARNING'],
        "level": "POSSIBLE / ANOMALY",
        "category_label": "CAT 2: URGENT", # <-- Clinical Triage Badge
        "icon": "🟠",
        "code": "RG-2B",
        "action": "Physician Review Recommended"
    },
    SeverityLevel.NORMAL: {
        "color": COLORS['GOOD'],
        "level": "NO SEIZURE DETECTED",   # <-- Updated Language
        "category_label": "CAT 3: ROUTINE", # <-- Clinical Triage Badge
        "icon": "🟢",
        "code": "RG-4N",
        "action": "Continue Monitoring"
    }
}

# -----------------------
# UPDATED STYLING
# -----------------------
st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Inter:wght@400;600;800&display=swap');
    
    /* Global Reset */
    .stApp {{
        background-color: {COLORS['BG_MAIN']};
        font-family: 'Inter', sans-serif;
    }}
    
    /* The "Primary" Button */
    div.stButton > button:first-child {{
        background: linear-gradient(90deg, #00C9FF 0%, #92FE9D 100%);
        color: #000000;
        font-weight: 800;
        border: none;
        border-radius: 8px;
        text-transform: uppercase;
        letter-spacing: 1px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0, 201, 255, 0.3);
    }}
    div.stButton > button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0, 201, 255, 0.5);
    }}

    /* Glassmorphism Cards */
    .glass-card {{
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 16px;
        transition: border 0.3s ease;
        text-align: left;
    }}
    .glass-card:hover {{
        border: 1px solid rgba(255, 255, 255, 0.2);
    }}
    
    /* Typography */
    h1,h2,h3 {{ font-family:'Inter',sans-serif; letter-spacing:-0.5px; }}
    
    .metric-value {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 28px;
        font-weight: 700;
        background: -webkit-linear-gradient(#fff, #aaa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }}
    .metric-label {{
        font-size: 11px;
        color: {COLORS['TEXT_MUTED']};
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 4px;
    }}
    .metric-sub {{
        font-size: 11px;
        color: {COLORS['ACCENT']};
        font-family: 'JetBrains Mono', monospace;
        margin-top: 4px;
    }}
    
    /* Alert Banner Polish */
    .alert-banner {{
        background: radial-gradient(circle at top right, rgba(30,30,30,0.5), rgba(10,10,10,1));
        border: 1px solid rgba(255,255,255,0.1);
        padding: 32px 24px;
        border-radius: 8px;
        margin-bottom: 24px; 
        display: flex; 
        align-items: center; 
        justify-content: space-between;
        box-shadow: 0 10px 30px rgba(0,0,0,0.5);
    }}
    .alert-title {{ font-family:'Inter',sans-serif; font-weight:800; font-size:32px; margin:0; letter-spacing:-1px; }}

    /* Breadcrumbs */
    .breadcrumb {{ display:flex; font-family:'IBM Plex Mono',monospace; font-size:12px; color:{COLORS['TEXT_MUTED']}; margin-bottom:20px; }}
    .breadcrumb-item {{ margin-right: 15px; opacity: 0.5; }}
    .breadcrumb-active {{ opacity: 1; color: {COLORS['ACCENT']}; font-weight: bold; border-bottom: 1px solid {COLORS['ACCENT']};}}
    
    /* Sim Badge */
    .sim-badge {{ background-color: rgba(0, 90, 156, 0.2); color: {COLORS['NOTICE']}; padding:4px 8px; border-radius:4px; font-size:10px; font-family:'IBM Plex Mono'; border:1px solid {COLORS['NOTICE']}; display:inline-block; margin-bottom:10px; }}
    
    /* File Uploader Polish */
    [data-testid='stFileUploader'] {{
        border: 1px dashed rgba(255,255,255,0.2);
        padding: 20px;
        border-radius: 10px;
    }}
    </style>
    """, unsafe_allow_html=True)

# -----------------------
# Utility functions
# -----------------------

def parse_confidence(conf: Any) -> float:
    """Parse a variety of confidence formats into 0..1 float."""
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
    """Return a UI-friendly severity dict including clinical category."""
    if prediction_code:
        code = prediction_code.upper()
        if code in ("ICTAL_EVENT", "SEIZURE", "EPILEPTIFORM"):
            level = SeverityLevel.CRITICAL if confidence >= 0.75 else SeverityLevel.WARNING
        elif code in ("ANOMALY", "ARTIFACT"):
            level = SeverityLevel.WARNING
        else:
            level = SeverityLevel.NORMAL
    else:
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
        'category_label': cfg['category_label'], 
        'icon': cfg['icon'],
        'code': cfg['code'],
        'action': cfg['action']
    }


def metric_card(col: Any, label: str, value: str, subtext: Optional[str] = None) -> None:
    with col:
        st.markdown(f"""
        <div class="glass-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            {f'<div class="metric-sub">↳ {subtext}</div>' if subtext else ''}
        </div>
        """, unsafe_allow_html=True)


def create_clinical_pdf(patient_id: str, session_id: str, ui_state: Dict[str, Any], metrics: Dict[str, str], notes: str, fig_bytes: Optional[bytes] = None) -> io.BytesIO:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=LETTER, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=72)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='Header', fontSize=18, leading=22, spaceAfter=12, textColor=colors.HexColor(COLORS['BG_MAIN'])))
    story = []
    story.append(Paragraph(f"NeuroGuard Clinical Report", styles['Header']))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')} | Session: {session_id}", styles['Normal']))
    story.append(Spacer(1, 12))
    risk_color = colors.HexColor(ui_state['color'])
    status_text = f"<b>STATUS: {ui_state['level']}</b><br/>CATEGORY: {ui_state['category_label']}<br/>RISK CODE: {ui_state['code']}"
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
# Networking helpers
# -----------------------

def _file_params_hash(file_bytes: bytes, params: Dict[str, Any]) -> str:
    h = hashlib.sha256()
    h.update(file_bytes)
    for k in sorted(params.keys()):
        h.update(str(k).encode())
        h.update(str(params[k]).encode())
    return h.hexdigest()

@st.cache_data(show_spinner=False)
def send_predict_request_cached(key: str, url: str, files_payload: Dict[str, Any], data_payload: Dict[str, Any]) -> Dict[str, Any]:
    resp = requests.post(url, files=files_payload, data=data_payload, timeout=120)
    resp.raise_for_status()
    return resp.json()

def send_predict_request(file_bytes: bytes, filename: str, patient_id: str, params: Dict[str, Any]) -> Dict[str, Any]:
    key = _file_params_hash(file_bytes, {**params, 'patient_id': patient_id, 'filename': filename})
    files_payload = {'file': (filename, file_bytes, 'application/octet-stream')}
    data_payload = {**params, 'patient_id': patient_id}
    try:
        return send_predict_request_cached(key, FLASK_PREDICT_URL, files_payload, data_payload)
    except requests.exceptions.RequestException as e:
        raise

# -----------------------
# UI Layout & Logic
# -----------------------

with st.sidebar:
    st.title("NeuroGuard")
    st.caption(f"Clinical Suite v3.1 | Session: {st.session_state['session_id']}")
    st.markdown("---")
    app_mode = st.radio("MODULE", ["Clinical Analysis", "Calibration"], label_visibility="collapsed")
    st.markdown("---")
    # Patient ID removed from here to prevent confusion
    with st.expander("🔧 TECHNICIAN CONFIG"):
        st.caption("Warning: Modifying defaults affects sensitivity.")
        ma_window = st.slider("Smoothing (Windows)", 1, 21, 5)
        peak_req = st.slider("Trigger Threshold", 0.5, 0.99, 0.80)
        min_duration = st.slider("Min Duration (s)", 2.0, 30.0, 6.0)
        mad_k = st.slider("Calibration Multiplier (k)", 2.0, 12.0, 6.0, step=0.5)
        # CHANGED DEFAULT TO TRUE FOR QC SAFETY
        show_qc_snippet = st.checkbox("Show QC snippet when no event detected", value=True)


if app_mode == "Clinical Analysis":
    def render_breadcrumbs(step: int) -> None:
        steps = ["1. UPLOAD", "2. ANALYZE", "3. REVIEW"]
        html = '<div class="breadcrumb">'
        for i, s in enumerate(steps, 1):
            active_class = "breadcrumb-active" if i == step else ""
            html += f'<div class="breadcrumb-item {active_class}">{s}</div>'
        html += '</div>'
        st.markdown(html, unsafe_allow_html=True)

    render_breadcrumbs(st.session_state['step'])
    
    # MOVED PATIENT ID TO MAIN HEADER (Safety Fix)
    col_id, col_filler = st.columns([1,2])
    with col_id:
        patient_id = st.text_input("Patient Record ID", value="chb01", help="Verify Patient ID before analysis")
    
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
        # --- THEATRICAL LOADING SEQUENCE ---
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        steps = [
            (10, "Ingesting EDF signal data..."),
            (30, "Applying Butterworth Bandpass Filter (0.5-40Hz)..."),
            (50, "Segmenting 2s Epochs..."),
            (70, "Running CNN-LSTM Inference Model..."),
            (90, "Analyzing Morphology & Confidence...")
        ]
        
        for p, text in steps:
            time.sleep(0.15)
            progress_bar.progress(p)
            status_text.markdown(f"**{text}**")
            
        status_text.markdown("**Finalizing Diagnostics...**")
        # -----------------------------------

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
                
                progress_bar.empty()
                status_text.empty()

                morph = res.get('morphology')
                if morph and isinstance(morph, dict) and 'y' in morph and 'x' in morph:
                    st.session_state['morphology_x'] = np.array(morph['x'])
                    st.session_state['morphology_y'] = np.array(morph['y'])
                    st.session_state.pop('morph_message', None)
                else:
                    st.session_state.pop('morphology_x', None)
                    st.session_state.pop('morphology_y', None)
                    st.session_state['morph_message'] = (
                        "No event morphology extracted — backend returned no waveform. "
                        "Do not assume lead connectivity or normal EEG based on a generated plot."
                    )
            else:
                st.error("Analysis Failed: No response from prediction service")
        except Exception as e:
            st.error(f"System Error: {e}")

    if 'result' in st.session_state:
        res = st.session_state['result']
        pred_code = res.get('pred_code') or res.get('prediction_code') or None
        overall_text = res.get('overall_prediction')
        conf_val = parse_confidence(res.get('confidence', 0))

        ui_state = determine_severity(pred_code, overall_text, conf_val)
        hex_color = ui_state['color'].lstrip('#')
        r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)

        # Updated HTML with Triage Badge
        st.markdown(f"""
        <style>
            .triage-badge {{
                background-color: rgba(0,0,0,0.4);
                border: 1px solid {ui_state['color']};
                color: {ui_state['color']};
                padding: 4px 12px;
                border-radius: 4px;
                font-family: 'JetBrains Mono', monospace;
                font-size: 14px;
                font-weight: bold;
                margin-left: 15px;
                vertical-align: middle;
                letter-spacing: 1px;
            }}
        </style>
        
        <div class="alert-banner" style="background-color: rgba({r}, {g}, {b}, 0.25); border-left: 12px solid {ui_state['color']};">
            <div>
                <h2 class="alert-title" style="color: {ui_state['color']};">
                    {ui_state['icon']} {ui_state['level']} 
                    <span class="triage-badge">{ui_state['category_label']}</span>
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
            
            # Label the threshold line for context
            fig_time.add_annotation(
                y=peak_req, x=0, text="Seizure Threshold", 
                showarrow=False, yref='y', xref='paper', 
                font=dict(color="white", size=10), yanchor="bottom"
            )
            
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
            overall = str(res.get('overall_prediction', '')).upper() if res.get('overall_prediction') else ''
            confidence_val = parse_confidence(res.get('confidence', 0))
            morph = res.get('morphology')

            pred_code = res.get('pred_code') or res.get('prediction_code') or None
            is_seizure = False
            if pred_code:
                is_seizure = pred_code.upper() in ("ICTAL_EVENT", "SEIZURE", "EPILEPTIFORM") or (confidence_val >= peak_req)
            else:
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
                st.markdown('<div class="sim-badge">🧪 NO EVENT MORPHOLOGY</div>', unsafe_allow_html=True)
                st.markdown(
                    "<div style='padding:12px; background-color: rgba(255,255,255,0.02); border-radius:8px; color:#BFC7CE;'>"
                    "<b>No event morphology available</b><br>"
                    "No seizure was detected in this recording (or no morphology could be extracted)."
                    "</div>",
                    unsafe_allow_html=True
                )
                if show_qc_snippet:
                    st.markdown("")
                    st.markdown(
                        "<div style='padding:18px; border-radius:8px; background:#111; border:1px dashed rgba(255,255,255,0.04);'>"
                        "<div style='color:#DDD; font-size:13px; font-weight:600;'>QC Placeholder</div>"
                        "<div style='color:#AAA; font-size:12px; margin-top:6px;'>Backend did not return waveform data.</div>"
                        "</div>",
                        unsafe_allow_html=True
                    )
                else:
                    st.caption("Enable 'Show QC snippet' in Technician Config to inspect.")

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
                href = f'<a href="data:application/pdf;base64,{b64_pdf}" download="NeuroGuard_Report.pdf" style="text-decoration:none;"><button style="background-color: {COLORS["ACCENT"]}; color: black; border: none; padding: 10px 20px; border-radius: 5px; font-weight: bold; cursor: pointer; width: 100%;">📥 DOWNLOAD PDF</button></a>'
                st.markdown(href, unsafe_allow_html=True)


elif app_mode == "Calibration":
    st.markdown("## Baseline Calibration")
    # --- FIX START ---
    # Added explicit Patient ID input here since it was removed from sidebar
    patient_id = st.text_input("Patient ID for Calibration", value="chb01")
    # --- FIX END ---
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
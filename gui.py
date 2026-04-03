import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox
import requests
import threading
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
from datetime import datetime
import io
import time

# ReportLab for Desktop PDF Generation
from reportlab.lib.pagesizes import LETTER
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch

# --- Configuration ---
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("dark-blue")

# Backend Configuration
FLASK_PREDICT_URL = "http://127.0.0.1:5000/predict_adaptive"
FLASK_CALIBRATE_URL = "http://127.0.0.1:5000/calibrate_multi"

class NeuroGuardApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Window Setup - "Workstation" Dimensions
        self.title("NeuroGuard Workstation | v2.5.0")
        self.geometry("1200x850")
        
        # Grid Layout
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # State Variables
        self.selected_analysis_file = None
        self.selected_calib_files = []
        self.patient_id_var = tk.StringVar(value="chb01")
        self.analysis_result = None 

        # --- Sidebar (Technician Controls) ---
        self.setup_sidebar()

        # --- Main Frames ---
        self.frame_analysis = ctk.CTkFrame(self, corner_radius=0, fg_color="transparent")
        self.frame_calibration = ctk.CTkFrame(self, corner_radius=0, fg_color="transparent")
        
        self.setup_analysis_view()
        self.setup_calibration_view() # This is the upgraded one

        # Start on Analysis
        self.show_frame("Analysis")

    def setup_sidebar(self):
        self.sidebar_frame = ctk.CTkFrame(self, width=250, corner_radius=0, fg_color="#1a1a1a")
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(10, weight=1)

        # Logo Area
        self.logo = ctk.CTkLabel(self.sidebar_frame, text="⚡ NEUROGUARD", font=ctk.CTkFont(size=22, weight="bold"))
        self.logo.grid(row=0, column=0, padx=20, pady=(20, 5), sticky="w")
        ctk.CTkLabel(self.sidebar_frame, text="Technician Console", text_color="#888").grid(row=1, column=0, padx=20, pady=(0, 20), sticky="w")

        # Navigation
        self.btn_nav_analysis = ctk.CTkButton(self.sidebar_frame, text=" DIAGNOSTICS", height=40, anchor="w",
                                              command=lambda: self.show_frame("Analysis"))
        self.btn_nav_analysis.grid(row=2, column=0, padx=10, pady=5, sticky="ew")
        
        self.btn_nav_calib = ctk.CTkButton(self.sidebar_frame, text=" CALIBRATION", height=40, anchor="w",
                                            fg_color="transparent", border_width=1, text_color="gray90",
                                            command=lambda: self.show_frame("Calibration"))
        self.btn_nav_calib.grid(row=3, column=0, padx=10, pady=5, sticky="ew")

        # --- SETTINGS PANEL (Visible Controls) ---
        self.settings_frame = ctk.CTkFrame(self.sidebar_frame, fg_color="#2b2b2b")
        self.settings_frame.grid(row=4, column=0, padx=10, pady=(30, 10), sticky="ew")
        
        ctk.CTkLabel(self.settings_frame, text="SIGNAL PARAMETERS", font=("Arial", 11, "bold"), text_color="#888").pack(anchor="w", padx=10, pady=5)
        
        # Sensitivity
        ctk.CTkLabel(self.settings_frame, text="Trigger Sensitivity").pack(anchor="w", padx=10)
        self.slider_peak = ctk.CTkSlider(self.settings_frame, from_=0.5, to=0.99, number_of_steps=50)
        self.slider_peak.set(0.8)
        self.slider_peak.pack(padx=10, pady=5, fill="x")
        
        # Smoothing
        ctk.CTkLabel(self.settings_frame, text="Smoothing Window").pack(anchor="w", padx=10)
        self.slider_smooth = ctk.CTkSlider(self.settings_frame, from_=1, to=20, number_of_steps=20)
        self.slider_smooth.set(5)
        self.slider_smooth.pack(padx=10, pady=5, fill="x")

    def show_frame(self, frame_name):
        self.frame_analysis.grid_forget()
        self.frame_calibration.grid_forget()
        
        if frame_name == "Analysis":
            self.frame_analysis.grid(row=0, column=1, sticky="nsew", padx=20, pady=20)
            self.btn_nav_analysis.configure(fg_color=["#3B8ED0", "#1F6AA5"])
            self.btn_nav_calib.configure(fg_color="transparent")
        else:
            self.frame_calibration.grid(row=0, column=1, sticky="nsew", padx=20, pady=20)
            self.btn_nav_calib.configure(fg_color=["#3B8ED0", "#1F6AA5"])
            self.btn_nav_analysis.configure(fg_color="transparent")

    # ==========================================
    # VIEW 1: ANALYSIS (The Dashboard)
    # ==========================================
    def setup_analysis_view(self):
        # Top Bar
        top_bar = ctk.CTkFrame(self.frame_analysis, fg_color="transparent")
        top_bar.pack(fill="x", pady=(0, 10))
        
        ctk.CTkLabel(top_bar, text="PATIENT RECORD:", text_color="gray").pack(side="left")
        ctk.CTkEntry(top_bar, textvariable=self.patient_id_var, width=120).pack(side="left", padx=10)
        
        self.btn_select = ctk.CTkButton(top_bar, text="LOAD EDF FILE", width=120, command=self.select_analysis_file)
        self.btn_select.pack(side="right")
        self.lbl_filename = ctk.CTkLabel(top_bar, text="No Data Loaded", text_color="gray")
        self.lbl_filename.pack(side="right", padx=20)

        # Status Banner
        self.banner = ctk.CTkFrame(self.frame_analysis, height=80, fg_color="#212121", corner_radius=8)
        self.banner.pack(fill="x", pady=10)
        self.banner.pack_propagate(False)
        
        self.lbl_status_icon = ctk.CTkLabel(self.banner, text="⚪", font=("Arial", 30))
        self.lbl_status_icon.pack(side="left", padx=(20, 10))
        
        info_frame = ctk.CTkFrame(self.banner, fg_color="transparent")
        info_frame.pack(side="left")
        self.lbl_status_main = ctk.CTkLabel(info_frame, text="SYSTEM IDLE", font=("Arial", 20, "bold"))
        self.lbl_status_main.pack(anchor="w")
        self.lbl_status_sub = ctk.CTkLabel(info_frame, text="Waiting for signal input...", text_color="gray")
        self.lbl_status_sub.pack(anchor="w")

        # Metrics
        metric_grid = ctk.CTkFrame(self.frame_analysis, fg_color="transparent")
        metric_grid.pack(fill="x", pady=15)
        self.card_dur = self.create_metric(metric_grid, "EVENT DURATION", "--")
        self.card_conf = self.create_metric(metric_grid, "AI CONFIDENCE", "--")
        self.card_burst = self.create_metric(metric_grid, "BURST COUNT", "--")
        
        # Graph Area
        self.graph_container = ctk.CTkFrame(self.frame_analysis, fg_color="#111")
        self.graph_container.pack(fill="both", expand=True, pady=10)
        
        self.fig, self.ax = plt.subplots(figsize=(8, 3), dpi=100)
        self.fig.patch.set_facecolor('#111')
        self.ax.set_facecolor('#111')
        self.setup_blank_plot()
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.graph_container)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        # Controls
        ctrl_frame = ctk.CTkFrame(self.frame_analysis, fg_color="transparent")
        ctrl_frame.pack(fill="x", pady=10)
        
        self.progress = ctk.CTkProgressBar(ctrl_frame, height=15)
        self.progress.pack(fill="x", pady=(0, 10))
        self.progress.set(0)
        
        self.btn_run = ctk.CTkButton(ctrl_frame, text="INITIATE ANALYSIS", fg_color="#27AE60", height=40, font=("Arial", 14, "bold"), command=self.run_analysis_thread)
        self.btn_run.pack(side="left", fill="x", expand=True, padx=(0, 10))
        
        self.btn_export = ctk.CTkButton(ctrl_frame, text="📄 EXPORT PDF REPORT", fg_color="#2980B9", height=40, state="disabled", command=self.export_report)
        self.btn_export.pack(side="right", fill="x", expand=True, padx=(10, 0))

    # ==========================================
    # VIEW 2: CALIBRATION (THE PRO UPGRADE)
    # ==========================================
    def setup_calibration_view(self):
        # Main Container with 2 Columns
        self.frame_calibration.grid_columnconfigure(0, weight=1)
        self.frame_calibration.grid_columnconfigure(1, weight=2) # Wider Log view
        self.frame_calibration.grid_rowconfigure(1, weight=1)

        # --- HEADER ---
        header = ctk.CTkFrame(self.frame_calibration, fg_color="transparent")
        header.grid(row=0, column=0, columnspan=2, sticky="ew", padx=20, pady=(20, 10))
        ctk.CTkLabel(header, text="SYSTEM CALIBRATION", font=("Arial", 24, "bold")).pack(anchor="w")
        ctk.CTkLabel(header, text="Establish patient-specific baseline thresholds via non-seizure data ingestion.", text_color="gray").pack(anchor="w")

        # --- LEFT PANEL: INPUTS ---
        input_panel = ctk.CTkFrame(self.frame_calibration, fg_color="#212121")
        input_panel.grid(row=1, column=0, sticky="nsew", padx=(20, 10), pady=10)
        
        ctk.CTkLabel(input_panel, text="1. PATIENT CONTEXT", font=("Arial", 12, "bold"), text_color="#3B8ED0").pack(anchor="w", padx=15, pady=(15, 5))
        ctk.CTkEntry(input_panel, textvariable=self.patient_id_var).pack(fill="x", padx=15, pady=5)
        
        ctk.CTkLabel(input_panel, text="2. DATA INGESTION", font=("Arial", 12, "bold"), text_color="#3B8ED0").pack(anchor="w", padx=15, pady=(20, 5))
        self.btn_calib_select = ctk.CTkButton(input_panel, text="📂 LOAD BASELINE FILES (Min 2)", height=35, fg_color="#2b2b2b", border_width=1, border_color="gray", command=self.select_calib_files)
        self.btn_calib_select.pack(fill="x", padx=15, pady=5)
        
        # File List (Scrollable)
        ctk.CTkLabel(input_panel, text="Selected Files Queue:", font=("Arial", 10, "bold"), text_color="gray").pack(anchor="w", padx=15, pady=(10, 0))
        self.file_list_frame = ctk.CTkScrollableFrame(input_panel, height=150, fg_color="#111")
        self.file_list_frame.pack(fill="both", expand=True, padx=15, pady=5)
        self.lbl_no_files = ctk.CTkLabel(self.file_list_frame, text="[Empty Queue]", text_color="#444")
        self.lbl_no_files.pack(pady=20)

        # Run Button
        self.btn_start_calib = ctk.CTkButton(input_panel, text="INITIATE CALIBRATION", state="disabled", height=50, fg_color="#F39C12", font=("Arial", 13, "bold"), command=self.run_calib_thread)
        self.btn_start_calib.pack(fill="x", padx=15, pady=20)

        # --- RIGHT PANEL: SYSTEM LOG (Terminal) ---
        log_panel = ctk.CTkFrame(self.frame_calibration, fg_color="#111")
        log_panel.grid(row=1, column=1, sticky="nsew", padx=(0, 20), pady=10)
        
        top_log_bar = ctk.CTkFrame(log_panel, height=30, fg_color="#333", corner_radius=0)
        top_log_bar.pack(fill="x")
        ctk.CTkLabel(top_log_bar, text=" TERMINAL OUTPUT", font=("Consolas", 11), text_color="#aaa").pack(side="left", padx=10)
        
        # The Terminal
        self.log_box = ctk.CTkTextbox(log_panel, font=("Consolas", 12), fg_color="#111", text_color="#00FF00", activate_scrollbars=True)
        self.log_box.pack(fill="both", expand=True, padx=5, pady=5)
        self.log_box.insert("0.0", "> System Ready.\n> Waiting for input data streams...\n")
        self.log_box.configure(state="disabled")

    # ==========================================
    # CALIBRATION LOGIC (With Terminal Effects)
    # ==========================================
    def select_calib_files(self):
        files = filedialog.askopenfilenames(filetypes=[("EDF Files", "*.edf")])
        if files:
            self.selected_calib_files = files
            
            # Clear visual queue
            for widget in self.file_list_frame.winfo_children():
                widget.destroy()
            
            # Populate List Visually
            for f in files:
                fname = os.path.basename(f)
                row = ctk.CTkFrame(self.file_list_frame, fg_color="#2b2b2b")
                row.pack(fill="x", pady=2)
                ctk.CTkLabel(row, text="📄", width=30).pack(side="left")
                ctk.CTkLabel(row, text=fname, anchor="w").pack(side="left", fill="x", expand=True)
                ctk.CTkLabel(row, text=f"{os.path.getsize(f)/1024:.0f}KB", text_color="gray", width=60).pack(side="right", padx=5)

            self.log_message(f"Loaded {len(files)} files into buffer.")
            if len(files) >= 2:
                self.btn_start_calib.configure(state="normal")
                self.log_message("Ready for calculation sequence.")
            else:
                self.btn_start_calib.configure(state="disabled")
                self.log_message("Warning: Minimum 2 files required for statistical significance.")

    def log_message(self, msg):
        self.log_box.configure(state="normal")
        ts = datetime.now().strftime("%H:%M:%S")
        self.log_box.insert("end", f"[{ts}] {msg}\n")
        self.log_box.see("end")
        self.log_box.configure(state="disabled")

    def run_calib_thread(self):
        self.btn_start_calib.configure(state="disabled")
        self.btn_calib_select.configure(state="disabled")
        threading.Thread(target=self._calib_task, daemon=True).start()

    def _calib_task(self):
        try:
            self.log_message("--- INITIATING CALIBRATION SEQUENCE ---")
            time.sleep(0.5) 
            
            files_list = []
            open_files = []
            
            for i, f_path in enumerate(self.selected_calib_files):
                fname = os.path.basename(f_path)
                self.log_message(f"Ingesting stream {i+1}/{len(self.selected_calib_files)}: {fname}...")
                f_obj = open(f_path, 'rb')
                open_files.append(f_obj)
                files_list.append(('files[]', (fname, f_obj, 'application/octet-stream')))
                time.sleep(0.2) # Theatrical pause
            
            self.log_message("Computing Median Absolute Deviation (MAD)...")
            data = {'patient_id': self.patient_id_var.get(), 'mad_k': 6.0}
            
            response = requests.post(FLASK_CALIBRATE_URL, files=files_list, data=data)
            
            for f in open_files: f.close()
            
            if response.status_code == 200:
                res_json = response.json()
                self.log_message("Calculation Complete.")
                self.log_message(f"New Threshold: {res_json.get('threshold'):.4f}")
                self.log_message(f"Profile saved for ID: {self.patient_id_var.get()}")
                self.log_message("--- SUCCESS ---")
                self.after(0, lambda: messagebox.showinfo("System Update", "Calibration Successful."))
            else:
                self.log_message(f"CRITICAL ERROR: {response.text}")
                self.after(0, lambda: messagebox.showerror("Error", response.text))
        
        except Exception as e:
            self.log_message(f"EXCEPTION: {str(e)}")
            self.after(0, lambda: messagebox.showerror("Error", str(e)))
        finally:
            self.after(0, lambda: self.btn_start_calib.configure(state="normal"))
            self.after(0, lambda: self.btn_calib_select.configure(state="normal"))

    # ==========================================
    # SHARED HELPERS
    # ==========================================
    def create_metric(self, parent, title, value):
        f = ctk.CTkFrame(parent, fg_color="#2b2b2b")
        f.pack(side="left", fill="x", expand=True, padx=5)
        ctk.CTkLabel(f, text=title, font=("Arial", 10, "bold"), text_color="gray").pack(pady=(10, 0))
        lbl = ctk.CTkLabel(f, text=value, font=("Arial", 24, "bold"))
        lbl.pack(pady=(0, 10))
        return lbl

    def setup_blank_plot(self):
        self.ax.clear()
        self.ax.grid(True, color='#333', linestyle='--')
        self.ax.spines['bottom'].set_color('#555')
        self.ax.spines['left'].set_color('#555')
        self.ax.spines['top'].set_color('none')
        self.ax.spines['right'].set_color('none')
        self.ax.tick_params(axis='x', colors='gray')
        self.ax.tick_params(axis='y', colors='gray')
        self.ax.set_xlabel("Time (Epochs)", color="gray")
        self.ax.set_ylabel("Probability", color="gray")
        self.ax.set_title("Signal Analysis Timeline", color="gray")

    def select_analysis_file(self):
        f = filedialog.askopenfilename(filetypes=[("EDF Files", "*.edf")])
        if f:
            self.selected_analysis_file = f
            self.lbl_filename.configure(text=os.path.basename(f), text_color="white")
            self.btn_export.configure(state="disabled")
            self.setup_blank_plot()
            self.canvas.draw()

    def run_analysis_thread(self):
        if not self.selected_analysis_file:
            messagebox.showwarning("Input Error", "Please load an EEG file first.")
            return
        
        self.btn_run.configure(state="disabled")
        self.progress.configure(mode="indeterminate")
        self.progress.start()
        self.lbl_status_main.configure(text="PROCESSING...", text_color="#F39C12")
        self.lbl_status_sub.configure(text="Running CNN-LSTM Inference Model...")
        
        threading.Thread(target=self._analysis_task, daemon=True).start()

    def _analysis_task(self):
        try:
            files = {'file': open(self.selected_analysis_file, 'rb')}
            ma = int(self.slider_smooth.get())
            peak = float(self.slider_peak.get())
            
            data = {'patient_id': self.patient_id_var.get(), 'ma_window': ma, 'peak_req': peak}
            response = requests.post(FLASK_PREDICT_URL, files=files, data=data)
            
            if response.status_code == 200:
                self.after(0, self.update_ui_success, response.json())
            else:
                self.after(0, lambda: messagebox.showerror("API Error", response.text))
        except Exception as e:
            self.after(0, lambda: messagebox.showerror("Connection Failed", str(e)))
        finally:
            self.after(0, self.reset_ui_state)

    def reset_ui_state(self):
        self.btn_run.configure(state="normal")
        self.progress.stop()
        self.progress.configure(mode="determinate")
        self.progress.set(1)

    def update_ui_success(self, result):
        self.analysis_result = result 
        
        pred = str(result.get('overall_prediction', '')).upper()
        if "SEIZURE" in pred and "NO " not in pred:
            self.banner.configure(fg_color="#591313") 
            self.lbl_status_icon.configure(text="🔴")
            self.lbl_status_main.configure(text="SEIZURE DETECTED", text_color="#FF4B4B")
            self.lbl_status_sub.configure(text=f"Priority: STAT | Confidence: {result.get('peak_probability',0)*100:.1f}%")
        else:
            self.banner.configure(fg_color="#0F3D1F") 
            self.lbl_status_icon.configure(text="🟢")
            self.lbl_status_main.configure(text="NO SEIZURE DETECTED", text_color="#2ECC71")
            self.lbl_status_sub.configure(text="Routine Baseline. Continue Monitoring.")

        self.card_dur.configure(text=f"{result.get('detection_span_seconds', 0):.1f}s")
        self.card_conf.configure(text=f"{result.get('peak_probability', 0)*100:.1f}%")
        self.card_burst.configure(text=f"{result.get('significant_bursts', 0)}")
        
        timeline = result.get('probability_timeline', [])
        self.ax.clear()
        self.setup_blank_plot()
        self.ax.plot(timeline, color='#00C9FF', linewidth=1.5)
        thresh = self.slider_peak.get()
        self.ax.axhline(y=thresh, color='#FF4B4B', linestyle='--', alpha=0.6, label="Trigger")
        self.ax.fill_between(range(len(timeline)), timeline, color='#00C9FF', alpha=0.1)
        self.canvas.draw()
        
        self.btn_export.configure(state="normal")

    def export_report(self):
        if not self.analysis_result: return
        file_path = filedialog.asksaveasfilename(defaultextension=".pdf", filetypes=[("PDF", "*.pdf")])
        if not file_path: return
        
        try:
            buf = io.BytesIO()
            self.fig.savefig(buf, format='png', facecolor='#111')
            buf.seek(0)
            
            doc = SimpleDocTemplate(file_path, pagesize=LETTER)
            styles = getSampleStyleSheet()
            story = []
            story.append(Paragraph(f"NeuroGuard Clinical Report", styles['Title']))
            story.append(Paragraph(f"Patient: {self.patient_id_var.get()} | Date: {datetime.now()}", styles['Normal']))
            story.append(Spacer(1, 12))
            data = [
                ["Metric", "Value"],
                ["Prediction", self.analysis_result.get('overall_prediction')],
                ["Confidence", f"{self.analysis_result.get('peak_probability')*100:.1f}%"],
                ["Duration", f"{self.analysis_result.get('detection_span_seconds')}s"]
            ]
            t = Table(data)
            t.setStyle(TableStyle([('GRID', (0,0), (-1,-1), 1, colors.black)]))
            story.append(t)
            story.append(Spacer(1, 12))
            img = RLImage(buf, width=6*inch, height=3*inch)
            story.append(img)
            doc.build(story)
            messagebox.showinfo("Export", "PDF Saved Successfully!")
        except Exception as e:
            messagebox.showerror("Export Error", str(e))

if __name__ == "__main__":
    app = NeuroGuardApp()
    app.mainloop()
import streamlit as st
import sqlite3
import hashlib
import numpy as np
import time
import json
from PIL import Image
from io import BytesIO

# ---------- DATABASE ----------
conn = sqlite3.connect("users.db", check_same_thread=False)
c = conn.cursor()

c.execute("""
CREATE TABLE IF NOT EXISTS users (
    username TEXT PRIMARY KEY,
    password TEXT NOT NULL
)
""")
conn.commit()

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def signup(username, password):
    try:
        c.execute("INSERT INTO users(username, password) VALUES (?, ?)",
                  (username, hash_password(password)))
        conn.commit()
        return True
    except:
        return False

def login(username, password):
    c.execute("SELECT * FROM users WHERE username=? AND password=?",
              (username, hash_password(password)))
    return c.fetchone() is not None

# ---------- SESSION ----------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "user" not in st.session_state:
    st.session_state.user = ""
    # ---------- LOGIN PAGE ----------
if not st.session_state.logged_in:
    st.set_page_config(page_title="InstruNet AI - Login", layout="centered")

    st.markdown("<h1 style='text-align:center;'>InstruNet AI</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align:center;'>Sign In / Sign Up</h3>", unsafe_allow_html=True)
    st.write("")

    tab1, tab2 = st.tabs(["Sign In", "Sign Up"])

    with tab1:
        st.subheader("Sign In")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("Login"):
            if login(username, password):
                st.session_state.logged_in = True
                st.session_state.user = username
                st.success("Login successful.")
                st.rerun()
            else:
                st.error("Invalid username or password.")

    with tab2:
        st.subheader("Sign Up")
        new_user = st.text_input("Create Username")
        new_pass = st.text_input("Create Password", type="password")
        confirm_pass = st.text_input("Confirm Password", type="password")

        if st.button("Register"):
            if new_pass != confirm_pass:
                st.error("Passwords do not match.")
            elif len(new_pass) < 4:
                st.warning("Password must be at least 4 characters.")
            else:
                if signup(new_user, new_pass):
                    st.success("Account created. Now sign in.")
                else:
                    st.error("Username already exists.")

    st.stop()   # VERY IMPORTANT: stops dashboard showing before login
    # ---------- DASHBOARD PAGE ----------
st.set_page_config(page_title="InstruNet AI", layout="wide")

st.sidebar.write("Logged in as:", st.session_state.user)
if st.sidebar.button("Logout"):
    st.session_state.logged_in = False
    st.session_state.user = ""
    st.rerun()

# Now paste your complete Instrument Recognition UI code below
import streamlit as st
import numpy as np
import time
import json
from PIL import Image
from io import BytesIO

# ---------------- SESSION STATE INIT ----------------
if "analyzed" not in st.session_state:
    st.session_state.analyzed = False

if "show_download" not in st.session_state:
    st.session_state.show_download = False

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="InstruNet AI", layout="wide")

# ---------------- BASIC STYLES ----------------
st.markdown("""
<style>


.block-container {
    padding-top: 0.5 rem;
}

/* Cards */
.card {
    background-color: #111827;
    padding: 10 px;
    border-radius: 10 px;
    margin-bottom: 10 px;
}

/* Title */
.app-title {
    text-align: center;
    font-size: 36px;
    font-weight: 700;
    color: #F54927;
    margin-top: 0;
    margin-bottom: 4px;
}

.app-subtitle {
    text-align: center;
    color: #9ca3af;
    font-size: 14 px;
    margin-bottom: 10 px;
}

/* ALL BUTTONS (Analyze, Download, Browse) */
div.stButton > button,
div.stDownloadButton > button {
    background-color: #F54927 !important;
    color: white ! important;
    border-radius: 6 px;
    padding: 6 px 14 px;
    border: none;
    font-weight: 500;
}

/* File uploader browse button */
section[data-testid="stFileUploader"] button {
    background-color: #F54927 !important;
    color: white ! important;
    border-radius: 6 px;
}

</style>
""", unsafe_allow_html=True)



# ---------------- TITLE ----------------
st.markdown(
    """
    <div class="app-title">InstruNet AI : Music Instrument Recognition</div>
    <div class="app-subtitle"> • Upload • Analyze • Discover</div>
    """,
    unsafe_allow_html=True
)


st.divider()

# ---------------- LAYOUT ----------------
left, center, right = st.columns([1.2, 2.5, 1.2])

# =================================================
# LEFT COLUMN
# =================================================
with left:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Upload Audio")
    audio = st.file_uploader("Choose audio file", type=["wav", "mp3", "flac"])
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Now Playing")

    if audio:
        st.audio(audio)
        st.caption(audio.name)
    else:
        st.caption("No file selected")

    analyze_clicked = st.button(
    "Analyze Track",
    disabled=(audio is None)
)

if audio is None:
    st.caption("Please upload an audio file to enable analysis.")

if analyze_clicked:
    st.session_state.analyzed = True
    st.session_state.show_download = False

    st.markdown("</div>", unsafe_allow_html=True)

# =================================================
# CENTER COLUMN
# =================================================
with center:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Spectrogram")

    if st.session_state.analyzed:
        spectrogram_img = Image.open("spectrogram.png")
        st.image(spectrogram_img, use_container_width=True)
    else:
        st.info("Click **Analyze Track** to view spectrogram")

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Instrument Levels")

    st.progress(80, text="Piano")
    st.progress(70, text="Drums")
    st.progress(65, text="Guitar")
    st.progress(40, text="Bass")

    st.markdown("</div>", unsafe_allow_html=True)

# =================================================
# RIGHT COLUMN
# =================================================
with right:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Detected Instruments")
    st.checkbox("Piano", True)
    st.checkbox("Drums", True)
    st.checkbox("Guitar", True)
    st.checkbox("Bass", True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Instrument Timeline")

    chart = st.empty()

    if st.session_state.analyzed:
        data = np.zeros((40, 1))
        for _ in range(40):
            data = np.roll(data, -1)
            data[-1] = np.random.randint(20, 60)
            chart.line_chart(data)
            time.sleep(0.04)

    # -------- DOWNLOAD SECTION --------
    if st.session_state.analyzed:

        if st.button("Download"):
            st.session_state.show_download = not st.session_state.show_download

        if st.session_state.show_download:

            file_type = st.radio(
                "Select file format",
                ["PDF", "JSON"],
                horizontal=True
            )

            report_data = {
                "instrument_levels": {
                    "Piano": 80,
                    "Drums": 70,
                    "Guitar": 65,
                    "Bass": 40
                },
                "detected_instruments": [
                    "Piano", "Drums", "Guitar", "Bass"
                ]
            }

            # -------- JSON DOWNLOAD --------
            if file_type == "JSON":
                json_data = json.dumps(report_data, indent=4)

                st.download_button(
                    "Download JSON",
                    json_data,
                    "instrunet_report.json",
                    "application/json"
                )

            # -------- PDF DOWNLOAD --------
            elif file_type == "PDF":
                pdf_buffer = BytesIO()
                pdf_buffer.write(b"%PDF-1.4\nInstruNet AI Report\n")
                pdf_buffer.write(bytes(json.dumps(report_data, indent=2), "utf-8"))
                pdf_buffer.seek(0)

                st.download_button(
                    "Download PDF",
                    pdf_buffer,
                    "instrunet_report.pdf",
                    "application/pdf"
                )

    st.markdown("</div>", unsafe_allow_html=True)
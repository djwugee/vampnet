import streamlit as st
import os
import gc
import tempfile
import random
import math
import logging
import numpy as np
import torch
import torchaudio
import audiotools as at
import matplotlib.pyplot as plt
from typing import Optional, Tuple, Dict, Any

# Ensure VampNet is installed in the environment: `pip install git+https://github.com/hugofloresgarcia/vampnet.git`
try:
    from vampnet.interface import Interface
except ImportError:
    st.error("VampNet backend not found. Please install via: pip install git+https://github.com/hugofloresgarcia/vampnet.git")
    st.stop()

# -----------------------------------------------------------------------------
# 1. APPLICATION CONFIGURATION & STATE INIT
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="VampNet Studio | Advanced Masked Token Modeling",
    page_icon="🦇",
    layout="wide",
    initial_sidebar_state="expanded"
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Inject Custom CSS for Advanced Audio Workstation UI
st.markdown("""
    <style>
    .stApp { background-color: #0E1117; color: #FAFAFA; font-family: 'Inter', sans-serif; }
    .stAudio { width: 100%; margin: 1rem 0; outline: none; }
    div[data-testid="stSidebar"] { background-color: #161B22; border-right: 1px solid #30363D; }
    .stButton>button { width: 100%; border-radius: 6px; font-weight: 700; background-color: #8A2BE2; color: white; border: none; padding: 0.75rem; transition: background-color 0.2s ease; }
    .stButton>button:hover { background-color: #9B30FF; }
    .stDownloadButton>button { background-color: #238636; }
    .stDownloadButton>button:hover { background-color: #2EA043; }
    .stSlider label { font-size: 0.9rem; font-weight: 500; }
    .history-card { background: #1C2128; padding: 1rem; border-radius: 8px; border: 1px solid #30363D; margin-bottom: 1rem; }
    </style>
""", unsafe_allow_html=True)

if "history" not in st.session_state:
    st.session_state.history =[]
if "active_model_name" not in st.session_state:
    st.session_state.active_model_name = "default"
if "telephone_iterations" not in st.session_state:
    st.session_state.telephone_iterations = 0
if "current_signal" not in st.session_state:
    st.session_state.current_signal = None

# -----------------------------------------------------------------------------
# 2. CORE BACKEND ARCHITECTURE & CACHING
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_vampnet_interface() -> Interface:
    """Loads the core VampNet neural interface into highly managed GPU memory."""
    logging.info("Initializing VampNet Neural Interface...")
    try:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        
        interface = Interface.default()
        interface.to(device)
        return interface
    except Exception as e:
        st.error(f"Failed to load Base Model: {str(e)}")
        st.stop()

@st.cache_data(show_spinner=False)
def fetch_available_models() -> list:
    return get_vampnet_interface().available_models()

def flush_vram():
    """Aggressive memory de-allocation to prevent out-of-memory errors on parallel generations."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    gc.collect()

def load_audio_signal(file_buffer, start_time: float, duration: float) -> at.AudioSignal:
    """Decodes arbitrary audio binaries into normalized continuous representations."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(file_buffer.getvalue())
        tmp_path = tmp.name
    try:
        signal = at.AudioSignal(tmp_path, offset=start_time, duration=duration)
        signal.to_mono()  # VampNet relies on monophonic discrete acoustic tokens
        if signal.sample_rate != get_vampnet_interface().codec.sample_rate:
            signal.resample(get_vampnet_interface().codec.sample_rate)
        return signal
    finally:
        os.remove(tmp_path)

def plot_spectrogram(signal: at.AudioSignal):
    """Generates a high-fidelity mel-spectrogram visual representation of the tokens."""
    fig, ax = plt.subplots(figsize=(10, 2), facecolor="#0E1117")
    ax.set_facecolor("#0E1117")
    
    # Calculate Mel-Spectrogram utilizing AudioTools core
    stft = signal.stft()
    magnitude = torch.abs(stft).squeeze().cpu().numpy()
    magnitude_db = 20 * np.log10(np.maximum(1e-5, magnitude))
    
    cax = ax.imshow(magnitude_db, aspect='auto', origin='lower', cmap='magma')
    ax.axis('off')
    plt.tight_layout(pad=0)
    
    # Convert plot to streamable object
    tmp_img = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    fig.savefig(tmp_img.name, dpi=150, bbox_inches='tight', transparent=True)
    plt.close(fig)
    return tmp_img.name

# -----------------------------------------------------------------------------
# 3. ADVANCED TENSOR MASKING ENGINE
# -----------------------------------------------------------------------------
def build_advanced_mask(
    interface: Interface,
    codes: torch.Tensor,
    mode: str,
    prefix_s: float,
    suffix_s: float,
    periodic_p: int,
    upper_mask: int,
    dropout: float,
    sample_rate: int = 44100
) -> torch.Tensor:
    """
    Direct tensor-level manipulation of the discrete codec codebooks.
    Generates exact temporal boundary masking for Outpainting, Inpainting, and Vamp continuations.
    Shape of codes:[Batch, N_Codebooks, TimeSteps]
    """
    B, C, T = codes.shape
    mask = torch.ones_like(codes, dtype=torch.bool, device=codes.device)
    
    # Calculate tokens per second (frame rate of the Descript Audio Codec)
    # Usually DAC operates at 86 Hz or similar depending on hop_length
    hop_length = interface.codec.hop_length
    fps = sample_rate / hop_length
    
    prefix_tokens = int(prefix_s * fps)
    suffix_tokens = int(suffix_s * fps)
    
    if mode == "Vamp (Variation)":
        # Base interface masking: retains period structural beats, drops everything else
        mask = interface.build_mask(
            codes, 
            periodic_prompt=periodic_p, 
            upper_codebook_mask=upper_mask
        )
        
        # Apply stochastic dropout on unmasked tokens
        if dropout > 0.0:
            dropout_mask = torch.rand_like(mask, dtype=torch.float32) < dropout
            mask = mask | dropout_mask
            
    elif mode == "Continuation (Outpainting)":
        # Condition on the prefix, mask everything that follows
        mask[:, :, :prefix_tokens] = 0
        mask[:, :, prefix_tokens:] = 1
        
    elif mode == "Inpainting":
        # Condition on prefix AND suffix, mask the middle chunk to bridge them
        mask[:, :, :prefix_tokens] = 0
        if suffix_tokens > 0:
            mask[:, :, -suffix_tokens:] = 0
        mask[:, :, prefix_tokens:-suffix_tokens if suffix_tokens > 0 else T] = 1

    elif mode == "Unconditional Generation":
        # Fully masked tensor; forces model to hallucinate entirely
        mask[:, :, :] = 1
        
    # Upper codebook masking (always applied to preserve base structural fidelity while varying high-frequencies)
    if upper_mask > 0 and C > upper_mask:
        mask[:, upper_mask:, :] = 1
        
    return mask

def crossfade_signals(s1: at.AudioSignal, s2: at.AudioSignal, fade_duration: float = 0.05) -> at.AudioSignal:
    """Seamlessly bridges generated boundaries to prevent zero-crossing clicks."""
    return s1.crossfade(s2, duration=fade_duration)

# -----------------------------------------------------------------------------
# 4. NEURAL INFERENCE PIPELINE
# -----------------------------------------------------------------------------
def execute_generation(
    interface: Interface,
    signal: Optional[at.AudioSignal],
    mode: str,
    params: Dict[str, Any]
) -> at.AudioSignal:
    """Executes the full parallel iterative decoding sequence mapping codes -> masked codes -> acoustic tokens -> waveform."""
    
    with torch.inference_mode():
        # 1. Unconditional Mode Handling
        if mode == "Unconditional Generation" or signal is None:
            # Generate empty sequence of appropriate length
            dummy_duration = params.get("target_duration", 5.0)
            dummy_signal = at.AudioSignal(torch.zeros(1, 1, int(dummy_duration * interface.codec.sample_rate)), sample_rate=interface.codec.sample_rate)
            dummy_signal.to(interface.device)
            codes = interface.encode(dummy_signal)
        else:
            signal.to(interface.device)
            codes = interface.encode(signal)

        # 2. Structural Mask Generation
        mask = build_advanced_mask(
            interface,
            codes,
            mode=mode,
            prefix_s=params.get("prefix_s", 0.0),
            suffix_s=params.get("suffix_s", 0.0),
            periodic_p=params.get("periodic_p", 7),
            upper_mask=params.get("upper_mask", 3),
            dropout=params.get("dropout", 0.0),
            sample_rate=interface.codec.sample_rate
        )

        # 3. Autoregressive / Non-Autoregressive Iterative Decoding
        output_tokens = interface.vamp(
            codes,
            mask,
            return_mask=False,
            temperature=params.get("temperature", 1.0),
            typical_filtering=params.get("typical_filtering", True),
            top_p=params.get("top_p", 1.0)
        )

        # 4. High-Fidelity Neural Decoding
        output_signal = interface.decode(output_tokens)
        output_signal.normalize()
        
        # 5. Boundary Smoothing (Inpainting/Continuation specific)
        if signal is not None and mode in["Continuation (Outpainting)", "Inpainting"]:
            fade_time = 0.05  # 50ms smoothing
            fps = interface.codec.sample_rate / interface.codec.hop_length
            prefix_t = params.get("prefix_s", 0.0)
            
            # Splice exact bounds if we are doing partial replacement to retain exact original audio on unmasked bounds
            if mode == "Continuation (Outpainting)":
                orig_prefix = signal.clone().truncate(0, prefix_t)
                gen_suffix = output_signal.clone().truncate(prefix_t, output_signal.duration)
                output_signal = crossfade_signals(orig_prefix, gen_suffix, fade_time)
                
    return output_signal.cpu()

# -----------------------------------------------------------------------------
# 5. UI CONTROLLERS & WORKSPACE LAYOUT
# -----------------------------------------------------------------------------
def main():
    # Application Header
    st.title("🦇 VampNet Studio | Production Engine")
    st.markdown("Fully featured implementation mapping masked acoustic token modeling for synthesis, compression, inpainting, and variation.")

    interface = get_vampnet_interface()
    available_models = fetch_available_models()

    # --- SIDEBAR: SYSTEM PARAMETERS ---
    with st.sidebar:
        st.header("⚙️ Core Engine Settings")
        
        selected_model = st.selectbox("Active Weight Set", ["default"] + available_models, index=0)
        if selected_model != st.session_state.active_model_name:
            with st.spinner(f"Swapping weights to {selected_model}..."):
                if selected_model == "default":
                    st.cache_resource.clear()
                    interface = get_vampnet_interface()
                else:
                    interface.load_finetuned(selected_model)
                st.session_state.active_model_name = selected_model
                flush_vram()
                st.success(f"Deployed {selected_model}")

        st.divider()
        st.subheader("Physics & Sampling")
        temperature = st.slider("Temperature", 0.1, 2.0, 1.0, 0.05, help="Variance scaling. <1.0 = High fidelity, >1.0 = High chaos")
        top_p = st.slider("Top-P (Nucleus)", 0.0, 1.0, 1.0, 0.05, help="Cumulative probability pruning. 1.0 disables Top-P.")
        typical_filtering = st.checkbox("Enable Typical Filtering", value=True, help="Truncates distribution tails to ensure acoustic coherence.")
        global_seed = st.number_input("Deterministic Seed (-1 for Random)", value=-1, step=1)
        
        st.divider()
        st.markdown("*System Status:*")
        st.code(f"Device: {interface.device}\nCodec Rate: {interface.codec.sample_rate}Hz\nPrecision: FP32")

    # --- MAIN WORKSPACE ---
    mode_tabs = st.tabs([
        "🎛️ Vamping (Variation)", 
        "➡️ Continuation", 
        "🎯 Inpainting", 
        "📞 Token Telephone", 
        "🎲 Unconditional"
    ])

    # Shared Audio Upload Logic
    st.subheader("📥 Source Primer Context")
    uploaded_file = st.file_uploader("Drop target waveform (WAV, MP3, FLAC)", type=["wav", "mp3", "flac"], key="global_uploader")
    
    col_trim1, col_trim2 = st.columns(2)
    start_time = col_trim1.number_input("Input Start Time (s)", 0.0, value=0.0, step=0.1)
    duration = col_trim2.number_input("Input Target Duration (s)", 1.0, 30.0, 10.0, step=0.5)

    current_input_signal = None
    if uploaded_file is not None:
        try:
            current_input_signal = load_audio_signal(uploaded_file, start_time, duration)
            st.audio(uploaded_file, start_time=int(start_time))
            # Spectrogram visualizer
            spec_img = plot_spectrogram(current_input_signal)
            st.image(spec_img, use_column_width=True, caption="Source Mel-Spectrogram")
        except Exception as e:
            st.error(f"Media Load Exception: {e}")

    # Aggregated Execution Parameters
    params = {
        "temperature": temperature,
        "typical_filtering": typical_filtering,
        "top_p": top_p
    }

    # -------------------------------------------------------------------------
    # TAB 1: VAMPING (Variation)
    # -------------------------------------------------------------------------
    with mode_tabs[0]:
        st.markdown("**Generates structural variations based on periodic tempo/beat masking.**")
        v_col1, v_col2 = st.columns(2)
        params["periodic_p"] = v_col1.slider("Periodic Prompt Interval", 0, 16, 7, help="Beat-sync rhythm. 7 works well for 4/4 time.")
        params["upper_mask"] = v_col2.slider("Upper Codebook Mask", 0, 4, 3, help="Filters out high-frequency detail for the model to hallucinate.")
        params["dropout"] = st.slider("Mask Dropout (Chaos %)", 0.0, 1.0, 0.0, 0.05, help="Randomly drop tokens to force extreme variation.")
        
        if st.button("Generate Variation", type="primary", use_container_width=True, disabled=current_input_signal is None):
            trigger_pipeline(interface, current_input_signal, "Vamp (Variation)", params, global_seed)

    # -------------------------------------------------------------------------
    # TAB 2: CONTINUATION / OUTPAINTING
    # -------------------------------------------------------------------------
    with mode_tabs[1]:
        st.markdown("**Extends the audio context natively beyond the provided prefix.**")
        c_col1, c_col2 = st.columns(2)
        params["prefix_s"] = c_col1.number_input("Prefix Condition Length (s)", 0.1, duration, min(3.0, duration/2), step=0.1)
        params["suffix_s"] = 0.0  # Not used in continuation
        
        if st.button("Generate Continuation", type="primary", use_container_width=True, disabled=current_input_signal is None):
            trigger_pipeline(interface, current_input_signal, "Continuation (Outpainting)", params, global_seed)

    # -------------------------------------------------------------------------
    # TAB 3: INPAINTING
    # -------------------------------------------------------------------------
    with mode_tabs[2]:
        st.markdown("**Replaces the middle of an audio sequence, bridging the start and end bounds.**")
        i_col1, i_col2 = st.columns(2)
        params["prefix_s"] = i_col1.number_input("Keep Start Condition (s)", 0.1, duration, min(2.0, duration/3), step=0.1, key="inpaint_pref")
        params["suffix_s"] = i_col2.number_input("Keep End Condition (s)", 0.1, duration, min(2.0, duration/3), step=0.1, key="inpaint_suff")
        
        if st.button("Execute Inpainting", type="primary", use_container_width=True, disabled=current_input_signal is None):
            trigger_pipeline(interface, current_input_signal, "Inpainting", params, global_seed)

    # -------------------------------------------------------------------------
    # TAB 4: TOKEN TELEPHONE (Iterative Processing)
    # -------------------------------------------------------------------------
    with mode_tabs[3]:
        st.markdown("**Sequentially loops generated audio back into the encoder to progressively degrade or evolve the sequence.**")
        t_col1, t_col2 = st.columns(2)
        iterations = t_col1.number_input("Generation Iterations", 2, 10, 3, step=1)
        params["periodic_p"] = t_col2.slider("Telephone Mask Interval", 0, 16, 7, key="tele_p")
        params["upper_mask"] = st.slider("Telephone Upper Codebook", 0, 4, 3, key="tele_u")
        params["dropout"] = 0.0
        
        if st.button("Start Telephone Loop", type="primary", use_container_width=True, disabled=current_input_signal is None):
            trigger_telephone_pipeline(interface, current_input_signal, iterations, params, global_seed)

    # -------------------------------------------------------------------------
    # TAB 5: UNCONDITIONAL
    # -------------------------------------------------------------------------
    with mode_tabs[4]:
        st.markdown("**Ignores input audio. Generates raw hallucinated compositions from scratch.**")
        params["target_duration"] = st.number_input("Target Generation Length (s)", 1.0, 15.0, 5.0, step=0.5)
        
        if st.button("Generate from Void", type="primary", use_container_width=True):
            trigger_pipeline(interface, None, "Unconditional Generation", params, global_seed)

    # -------------------------------------------------------------------------
    # GENERATION HISTORY & PLAYBACK POOL
    # -------------------------------------------------------------------------
    if len(st.session_state.history) > 0:
        st.divider()
        st.header("🕰️ Output History Rack")
        
        for idx, item in enumerate(reversed(st.session_state.history)):
            with st.container():
                st.markdown(f"<div class='history-card'>", unsafe_allow_html=True)
                col_info, col_play = st.columns([1, 2])
                with col_info:
                    st.markdown(f"**Run {len(st.session_state.history) - idx}** | Mode: `{item['mode']}`")
                    st.caption(f"Temp: {item['params']['temperature']} | Top-p: {item['params']['top_p']}")
                with col_play:
                    st.audio(item['path'], format="audio/wav")
                    with open(item['path'], "rb") as f:
                        st.download_button("💾 Export WAV", f, file_name=f"vampnet_out_{idx}.wav", key=f"dl_{idx}")
                st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 6. PIPELINE ORCHESTRATION FUNCTIONS
# -----------------------------------------------------------------------------
def trigger_pipeline(interface: Interface, signal: Optional[at.AudioSignal], mode: str, params: Dict[str, Any], seed: int):
    """Handles deterministic RNG routing, UI spinner rendering, and history appending for a standard run."""
    setup_rng(seed)
    with st.spinner(f"Initiating {mode} Phase..."):
        try:
            output_signal = execute_generation(interface, signal, mode, params)
            save_and_log_generation(output_signal, mode, params)
            st.success("Generation Complete.")
            st.balloons()
        except RuntimeError as e:
            st.error(f"Hardware/Memory Exception: {str(e)}\n\n*Try reducing duration or flushing memory.*")
        except Exception as e:
            st.error(f"Inference Engine Failed: {str(e)}")
        finally:
            flush_vram()

def trigger_telephone_pipeline(interface: Interface, signal: at.AudioSignal, iterations: int, params: Dict[str, Any], seed: int):
    """Handles the recursive iterative sequence degradation mode."""
    setup_rng(seed)
    current_sig = signal.clone()
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        for i in range(iterations):
            status_text.text(f"Telephone Iteration [{i+1}/{iterations}]...")
            current_sig = execute_generation(interface, current_sig, "Vamp (Variation)", params)
            
            # Save intermediate
            save_and_log_generation(current_sig, f"Telephone Loop {i+1}", params)
            progress_bar.progress(int(((i+1)/iterations)*100))
            
        st.success(f"Successfully completed {iterations} iterative degradation loops.")
    except Exception as e:
        st.error(f"Telephone sequence broken at iteration {i}: {str(e)}")
    finally:
        flush_vram()

def setup_rng(seed: int):
    """Enforces absolute determinism across the full torch stack if seed is provided."""
    if seed != -1:
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
    else:
        torch.seed()

def save_and_log_generation(signal: at.AudioSignal, mode: str, params: Dict[str, Any]):
    """Commits output buffer to disk and registers path into the Streamlit session state memory."""
    tmp_out = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    signal.write(tmp_out.name)
    st.session_state.history.append({
        "path": tmp_out.name,
        "mode": mode,
        "params": params.copy()
    })

if __name__ == "__main__":
    main()

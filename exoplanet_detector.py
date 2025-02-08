import streamlit as st
import lightkurve as lk
import numpy as np
import matplotlib.pyplot as plt
from urllib.request import urlopen
from io import BytesIO
from zipfile import ZipFile

st.title("🔭 Exoplanet Detector")
st.write("Enter a **target pixel name** to analyze its light curve and check for possible exoplanet detection.")

# User input for the target pixel name
target_name = st.text_input("Enter Target Pixel Name (e.g., KIC 8462852):", "KIC 8462852")

def fetch_data(target):
    """Download the target pixel file for the given target name."""
    try:
        st.write(f"🔍 Searching for data on {target}...")
        tpf = lk.search_targetpixelfile(target, quarter=16).download(quality_bitmask='hardest')
        return tpf
    except Exception as e:
        st.error(f"❌ Error fetching data: {e}")
        return None

if st.button("Analyze"):
    tpf = fetch_data(target_name)
    
    if tpf is None:
        st.error("⚠ No data found for this target. Try another star.")
    else:
        st.success(f"✅ Data found for {target_name}!")

        # Plot pixel snapshot
        fig, ax = plt.subplots()
        tpf.plot(frame=42, ax=ax)
        st.pyplot(fig)

        # Convert to light curve
        lc = tpf.to_lightcurve(aperture_mask=tpf.pipeline_mask)

        # Plot light curve
        fig, ax = plt.subplots()
        lc.plot(ax=ax)
        st.pyplot(fig)

        # Flatten light curve
        flat_lc = lc.flatten()
        fig, ax = plt.subplots()
        flat_lc.plot(ax=ax)
        st.pyplot(fig)

        # Periodogram analysis
        period = np.linspace(1, 5, 10000)
        bls = lc.to_periodogram(method='bls', period=period, frequency_factor=500)
        fig, ax = plt.subplots()
        bls.plot(ax=ax)
        st.pyplot(fig)

        # Extract periodogram results
        planet_x_period = bls.period_at_max_power
        planet_x_t0 = bls.transit_time_at_max_power
        planet_x_dur = bls.duration_at_max_power

        # Phase-folded light curve
        fig, ax = plt.subplots()
        lc.fold(period=planet_x_period, epoch_time=planet_x_t0).scatter(ax=ax)
        ax.set_xlim(-3,3)
        st.pyplot(fig)

        # Check for potential exoplanet
        if planet_x_period.value > 1:
            st.success(f"🌍 Possible Exoplanet detected with period **{planet_x_period:.2f} days**!")
        else:
            st.info("🔬 No strong indication of an exoplanet found.")

# Function to download and extract TESS data
def download_and_unzip(url, extract_to="."):
    try:
        st.write("📥 Downloading TESS data...")
        http_response = urlopen(url)
        zipfile = ZipFile(BytesIO(http_response.read()))
        zipfile.extractall(path=extract_to)
        st.success("✅ Data downloaded and extracted successfully!")
    except Exception as e:
        st.error(f"❌ Download failed: {e}")

# Button for downloading TESS data
if st.button("Download TESS Data"):
    product_group_id = "96862624"
    url = f"https://mast.stsci.edu/api/v0.1/Download/bundle.zip?previews=false&obsid={product_group_id}"
    destination = "/TESS/"
    download_and_unzip(url, destination)


from pathlib import Path
import json
import tempfile

import joblib
import nibabel as nib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
BASELINE_DIR = ROOT / "01_data_preprocessing" / "clinical_baseline"
MODEL_PATH = BASELINE_DIR / "clinical_logistic_baseline.joblib"
METRICS_PATH = BASELINE_DIR / "clinical_logistic_baseline_metrics.json"
XAI_PATH = BASELINE_DIR / "clinical_feature_coefficients.csv"
SHAP_SUMMARY_PATH = BASELINE_DIR / "clinical_shap_summary.png"
SHAP_WATERFALL_PATH = BASELINE_DIR / "clinical_shap_waterfall.png"
SHAP_IMPORTANCE_PATH = BASELINE_DIR / "clinical_shap_feature_importance.csv"
SHAP_REPORT_PATH = BASELINE_DIR / "clinical_shap_report.json"
PREDICTION_DIR = ROOT / "data" / "processed" / "ct_inference"
CT_METRICS_PATH = PREDICTION_DIR / "test_set_evaluation_summary.json"

st.set_page_config(
    page_title="AI Pancreatic Cancer Research Dashboard",
    page_icon="🩺",
    layout="wide",
)

st.markdown("""
<style>
.stApp {
    background: #f2f6fc;
    color: #14264b;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0b172b, #172c49);
}
[data-testid="stSidebar"] * {
    color: #f4f7ff;
}
[data-testid="stMainBlockContainer"] {
    max-width: 1650px;
    padding-top: 1.3rem;
}
h1, h2, h3 {
    color: #102653;
}
[data-testid="stVerticalBlockBorderWrapper"] {
    background: white;
    border: 1px solid #d9e6f7;
    border-radius: 13px;
    padding: 10px;
}
[data-testid="stMetric"] {
    background: white;
    border: 1px solid #d9e6f7;
    border-radius: 12px;
    padding: 12px;
}
div.stButton > button[kind="primary"] {
    background: #1265e8;
    color: white;
    border-radius: 9px;
    border: none;
    min-height: 40px;
}
[data-testid="stFileUploader"] {
    background: white;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)


def load_json(path):
    if not path.is_file():
        return None
    try:
        with path.open(encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


@st.cache_resource
def load_model():
    if not MODEL_PATH.is_file():
        return None
    return joblib.load(MODEL_PATH)


def make_slice(volume, axis, index):
    if axis == "Axial":
        image = volume[:, :, index]
    elif axis == "Coronal":
        image = volume[:, index, :]
    else:
        image = volume[index, :, :]
    return np.rot90(image)


def show_ct_viewer(volume, spacing):
    st.caption(
        f"Dimensions: {volume.shape} | "
        f"Voxel spacing (mm): {tuple(round(float(x), 3) for x in spacing)}"
    )
    axis = st.selectbox(
        "View orientation",
        ["Axial", "Coronal", "Sagittal"],
        key="ct_axis",
    )
    axis_number = {"Sagittal": 0, "Coronal": 1, "Axial": 2}[axis]
    maximum = int(volume.shape[axis_number] - 1)
    index = st.slider(
        "CT slice",
        min_value=0,
        max_value=max(0, maximum),
        value=max(0, maximum // 2),
        key=f"slice_{axis}",
    )
    image = make_slice(volume, axis, index)
    fig, ax = plt.subplots(figsize=(7, 6))
    finite = image[np.isfinite(image)]
    if finite.size:
        low, high = np.percentile(finite, [1, 99])
        if high <= low:
            high = low + 1
        ax.imshow(image, cmap="gray", vmin=low, vmax=high)
    else:
        ax.imshow(np.zeros_like(image), cmap="gray")
    ax.set_title(f"{axis} view — slice {index}")
    ax.axis("off")
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def get_saved_prediction():
    if not PREDICTION_DIR.is_dir():
        return None
    candidates = sorted(PREDICTION_DIR.glob("*_prediction.nii.gz"))
    return candidates[0] if candidates else None


# Sidebar
with st.sidebar:
    page = st.radio(
        "Navigation",
        [
            "Upload & Predict",
            "CT Image Analysis",
            "Clinical Data",
            "Model Performance",
            "Explainability",
            "About Project",
        ],
    )

metrics = load_json(METRICS_PATH)
ct_metrics = load_json(CT_METRICS_PATH)
model = None
model_error = None
try:
    model = load_model()
except Exception as exc:
    model_error = str(exc)

# Persist uploaded scan and clinical inputs across navigation.
if "ct_volume" not in st.session_state:
    st.session_state.ct_volume = None
if "ct_spacing" not in st.session_state:
    st.session_state.ct_spacing = None
if "ct_name" not in st.session_state:
    st.session_state.ct_name = None
if "clinical_result" not in st.session_state:
    st.session_state.clinical_result = None

elif page == "Upload & Predict":
    left, right = st.columns([1, 1.5], gap="large")

    with left:
        st.subheader("1. Upload CT scan")
        uploaded = st.file_uploader(
            "Choose NIfTI CT scan",
            type=["nii", "gz"],
            help="Use .nii or .nii.gz. Maximum upload size depends on Streamlit configuration.",
        )

        if uploaded is not None:
            valid_name = uploaded.name.lower().endswith((".nii", ".nii.gz"))
            if not valid_name:
                st.error("Upload a .nii or .nii.gz file.")
            else:
                try:
                    suffix = ".nii.gz" if uploaded.name.lower().endswith(".nii.gz") else ".nii"
                    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                        tmp.write(uploaded.getvalue())
                        tmp_path = Path(tmp.name)
                    try:
                        nii = nib.load(str(tmp_path))
                        volume = nii.get_fdata(dtype=np.float32)
                        if volume.ndim != 3:
                            st.error(
                                f"Expected a 3D CT volume; received {volume.ndim} dimensions."
                            )
                        elif not np.isfinite(volume).any():
                            st.error("The uploaded volume contains no finite image values.")
                        else:
                            st.session_state.ct_volume = volume
                            st.session_state.ct_spacing = nii.header.get_zooms()[:3]
                            st.session_state.ct_name = uploaded.name
                            st.success("CT loaded successfully.")
                    finally:
                        tmp_path.unlink(missing_ok=True)
                except Exception as exc:
                    st.error(f"Could not read this NIfTI file: {exc}")

        st.subheader("2. Clinical information")
        with st.form("clinical_input_form"):
            age = st.number_input(
                "Age (years)", min_value=18, max_value=110, value=60
            )
            sex = st.selectbox("Sex", ["F", "M", "Unknown"])
            scanner = st.text_input("Scanner category", value="Unknown")
            submitted = st.form_submit_button(
                "Run clinical baseline",
                type="primary",
                use_container_width=True,
            )

        if submitted:
            if model is None:
                st.error("Clinical model file is unavailable.")
            else:
                sample = pd.DataFrame([{
                    "patient_age_years": age,
                    "patient_sex": sex,
                    "scanner": scanner.strip() or "Unknown",
                }])
                try:
                    score = float(model.predict_proba(sample)[0, 1])
                    st.session_state.clinical_result = {
                        "score": score,
                        "age": age,
                        "sex": sex,
                        "scanner": scanner.strip() or "Unknown",
                    }
                except Exception as exc:
                    st.error(f"Clinical baseline failed: {exc}")

    with right:
        st.subheader("CT Image Analysis")
        if st.session_state.ct_volume is not None:
            st.caption(f"Loaded scan: {st.session_state.ct_name}")
            show_ct_viewer(
                st.session_state.ct_volume,
                st.session_state.ct_spacing,
            )
        else:
            st.info("Upload a NIfTI CT scan to view its slices.")

        st.subheader("Prediction result")
        result = st.session_state.clinical_result
        if result is not None:
            st.metric(
                "Clinical model score for the PDAC dataset label",
                f"{result['score']:.1%}",
            )
            st.caption(
                "Uncalibrated research model score; not an individual clinical "
                "risk estimate or diagnosis. This is not a CT-based prediction."
            )
        else:
            st.warning(
                "No result yet. Run the clinical baseline to display its output."
            )

        st.info(
            "CT cancer classification and CT-clinical fusion are not implemented. "
            "The available U-Net performs experimental segmentation and has "
            "poor tumor Dice performance."
        )

elif page == "CT Image Analysis":
    st.subheader("CT Image Analysis")
    if st.session_state.ct_volume is not None:
        show_ct_viewer(
            st.session_state.ct_volume,
            st.session_state.ct_spacing,
        )
    else:
        st.info("Upload a scan in Upload & Predict first.")

    st.markdown("### Available saved segmentation output")
    pred_path = get_saved_prediction()
    if pred_path is not None:
        try:
            pred_nii = nib.load(str(pred_path))
            pred = pred_nii.get_fdata()
            st.caption(
                f"Saved experimental U-Net output: {pred_path.name} | "
                f"shape: {pred.shape}"
            )
            z = st.slider(
                "Prediction slice",
                0,
                max(0, pred.shape[2] - 1),
                max(0, pred.shape[2] // 2),
                key="prediction_slice",
            )
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.imshow(np.rot90(pred[:, :, z]), cmap="viridis", interpolation="nearest")
            ax.set_title("Predicted segmentation classes")
            ax.axis("off")
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
            st.caption(
                "Class labels are model predictions, not confirmed cancer regions. "
                "This saved output is for a processed test case and is not "
                "automatically matched to the uploaded scan."
            )
        except Exception as exc:
            st.error(f"Could not read saved segmentation: {exc}")
    else:
        st.info(
            "No saved segmentation output found. Live inference is not connected "
            "to uploaded scans in this dashboard."
        )

    st.markdown("### Grad-CAM")
    st.info(
        "Grad-CAM is pending. The current 3D U-Net is a segmentation model, "
        "not a verified cancer-classification model. No Grad-CAM heatmap is shown."
    )

elif page == "Clinical Data":
    st.subheader("Clinical Data Summary")
    result = st.session_state.clinical_result
    if result:
        st.dataframe(
            pd.DataFrame([{
                "Age (years)": result["age"],
                "Sex": result["sex"],
                "Scanner": result["scanner"],
            }]),
            use_container_width=True,
            hide_index=True,
        )
        st.success("Required input fields are present.")
    else:
        st.info("Enter clinical metadata in Upload & Predict to see its summary.")

    st.markdown("### Dataset overview")
    clinical_csv = (
        ROOT / "data" / "processed" / "clinical"
        / "panorama_clinical_preprocessed" / "clinical_model_ready.csv"
    )
    if clinical_csv.is_file():
        try:
            clinical_df = pd.read_csv(clinical_csv)
            st.write(f"Dataset rows: {len(clinical_df):,}")
            if "target_pdac" in clinical_df.columns:
                st.write("Target label counts")
                st.bar_chart(clinical_df["target_pdac"].value_counts().sort_index())
            st.caption(
                "Dataset-level metadata summary. Do not enter identifiable patient data."
            )
        except Exception as exc:
            st.warning(f"Could not load clinical dataset summary: {exc}")
    else:
        st.info("Preprocessed clinical dataset not found.")

elif page == "Model Performance":
    st.subheader("Model Performance")

    st.markdown("### Clinical metadata baseline")
    if metrics:
        names = [
            ("Accuracy", "test_accuracy"),
            ("Precision", "test_precision"),
            ("Recall / sensitivity", "test_recall_sensitivity"),
            ("Specificity", "test_specificity"),
            ("F1-score", "test_f1"),
            ("ROC-AUC", "test_roc_auc"),
            ("Average precision", "test_average_precision_pr_auc"),
        ]
        cols = st.columns(4)
        for i, (label, key) in enumerate(names):
            if key in metrics and metrics[key] is not None:
                with cols[i % 4]:
                    st.metric(label, f"{metrics[key]:.3f}")

        cm = metrics.get("test_confusion_matrix", {})
        if cm:
            st.markdown("#### Confusion matrix — clinical baseline")
            matrix = pd.DataFrame(
                [
                    [cm["true_negative"], cm["false_positive"]],
                    [cm["false_negative"], cm["true_positive"]],
                ],
                index=["Actual negative", "Actual positive"],
                columns=["Predicted negative", "Predicted positive"],
            )
            st.dataframe(matrix, use_container_width=True)
            st.caption("Rows = actual labels; columns = predicted labels.")
        st.caption(
            "These metrics belong to the clinical metadata baseline, not the CT "
            "segmentation model or a fused multimodal model."
        )
    else:
        st.warning("Clinical evaluation metrics file not found.")

    st.divider()
    st.markdown("### CT segmentation — held-out test set")
    if ct_metrics and isinstance(ct_metrics.get("mean_metrics"), dict):
        mean = ct_metrics["mean_metrics"]
        median = ct_metrics.get("median_metrics", {})
        evaluated = ct_metrics.get("successfully_evaluated_cases", "N/A")
        requested = ct_metrics.get("requested_test_cases", "N/A")
        st.write(f"Successfully evaluated cases: **{evaluated} / {requested}**")
        st.caption(
            "Mean voxel-level segmentation metrics across the held-out test set. "
            "Dice and IoU measure overlap with reference masks."
        )

        st.markdown("#### Pancreas segmentation")
        cols = st.columns(4)
        for col, (label, key) in zip(cols, [
            ("Dice", "pancreas_dice"),
            ("IoU", "pancreas_iou"),
            ("Precision", "pancreas_precision"),
            ("Recall", "pancreas_recall"),
        ]):
            if key in mean:
                col.metric(label, f"{mean[key]:.4f}")

        st.markdown("#### Tumor segmentation")
        cols = st.columns(4)
        for col, (label, key) in zip(cols, [
            ("Dice", "tumor_dice"),
            ("IoU", "tumor_iou"),
            ("Precision", "tumor_precision"),
            ("Recall", "tumor_recall"),
        ]):
            if key in mean:
                col.metric(label, f"{mean[key]:.4f}")

        rows = []
        for region in ("pancreas", "tumor"):
            for metric_name in ("dice", "iou", "precision", "recall"):
                key = f"{region}_{metric_name}"
                if key in median:
                    rows.append({
                        "Region": region.capitalize(),
                        "Metric": metric_name.upper() if metric_name in ("dice", "iou") else metric_name.capitalize(),
                        "Median": median[key],
                    })
        if rows:
            st.markdown("#### Median scores across test cases")
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        tumor_dice = mean.get("tumor_dice")
        if tumor_dice is not None:
            st.warning(
                f"Important limitation: mean tumor Dice is {tumor_dice:.4f} "
                f"({tumor_dice:.2%}); median tumor Dice is "
                f"{median.get('tumor_dice', float('nan')):.4f}. The current model "
                "has poor tumor overlap performance and must not be used to "
                "diagnose or rule out pancreatic cancer."
            )
        st.caption(
            "Research evaluation only. These segmentation results do not establish "
            "validated early-cancer detection."
        )
    else:
        st.warning("CT evaluation summary was not found or has an unexpected format.")

    st.divider()
    st.markdown("### Model comparison")
    st.info(
        "The clinical metadata baseline and CT segmentation model perform different "
        "tasks and should not be compared as equivalent classifiers. Patient-level "
        "matched multimodal fusion has not been validated."
    )

elif page == "Explainability":
    st.subheader("Explainability")
    st.markdown("### Clinical model coefficients")
    if XAI_PATH.is_file():
        try:
            coef = pd.read_csv(XAI_PATH)
            if {"encoded_feature", "coefficient"}.issubset(coef.columns):
                coef = coef.dropna(subset=["coefficient"]).copy()
                coef["coefficient"] = pd.to_numeric(
                    coef["coefficient"], errors="coerce"
                )
                coef = coef.dropna(subset=["coefficient"])
                coef = coef.sort_values("coefficient")
                st.caption(
                    "Positive coefficients increase the model's log-odds for "
                    "the dataset label; negative coefficients decrease them, "
                    "holding the other encoded features constant."
                )
                fig, ax = plt.subplots(figsize=(9, max(4, 0.35 * len(coef))))
                colors = [
                    "#168bcb" if value >= 0 else "#8757cf"
                    for value in coef["coefficient"]
                ]
                ax.barh(coef["encoded_feature"], coef["coefficient"], color=colors)
                ax.axvline(0, color="black", linewidth=0.8)
                ax.set_xlabel("Logistic regression coefficient")
                ax.set_title("Clinical baseline feature coefficients")
                fig.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
                st.dataframe(coef, use_container_width=True, hide_index=True)
            else:
                st.warning("Coefficient report does not have expected columns.")
        except Exception as exc:
            st.error(f"Could not read feature explanation report: {exc}")
    else:
        st.info("Clinical coefficient report not found. Run explain_clinical_model.py.")

    st.markdown("### SHAP explainability — clinical metadata model")
    shap_report = load_json(SHAP_REPORT_PATH)
    if SHAP_SUMMARY_PATH.is_file() and SHAP_WATERFALL_PATH.is_file():
        if shap_report:
            st.caption(
                f"Method: {shap_report.get('method', 'SHAP')} | "
                f"Explained examples: {shap_report.get('explained_examples', 'N/A')} | "
                f"Split: {shap_report.get('explained_split', 'N/A')}"
            )
        st.warning(
            "These plots explain the clinical metadata model's dataset-label score. "
            "They do not explain CT pixels, establish causation, or provide a validated "
            "individual cancer risk."
        )
        st.markdown("#### Global SHAP summary")
        st.image(
            str(SHAP_SUMMARY_PATH),
            caption="SHAP summary for age, sex, and scanner metadata",
            use_container_width=True,
        )
        left_shap, right_shap = st.columns(2)
        with left_shap:
            st.markdown("#### Single-example waterfall")
            st.image(
                str(SHAP_WATERFALL_PATH),
                caption="Explanation for one held-out test example",
                use_container_width=True,
            )
        with right_shap:
            st.markdown("#### Mean absolute SHAP importance")
            if SHAP_IMPORTANCE_PATH.is_file():
                try:
                    shap_importance = pd.read_csv(SHAP_IMPORTANCE_PATH)
                    required = {"feature", "mean_absolute_shap_value"}
                    if required.issubset(shap_importance.columns):
                        shap_importance = shap_importance.sort_values(
                            "mean_absolute_shap_value", ascending=True
                        )
                        fig, ax = plt.subplots(
                            figsize=(7, max(3.5, 0.7 * len(shap_importance)))
                        )
                        ax.barh(
                            shap_importance["feature"],
                            shap_importance["mean_absolute_shap_value"],
                            color="#168bcb",
                        )
                        ax.set_xlabel("Mean absolute SHAP value")
                        ax.set_title("Global feature importance")
                        fig.tight_layout()
                        st.pyplot(fig, use_container_width=True)
                        plt.close(fig)
                        st.dataframe(
                            shap_importance,
                            use_container_width=True,
                            hide_index=True,
                        )
                    else:
                        st.warning("SHAP importance CSV has unexpected columns.")
                except Exception as exc:
                    st.error(f"Could not load SHAP importance: {exc}")
            else:
                st.info("SHAP importance CSV not found. Run explain_clinical_shap.py.")
    else:
        st.info(
            "SHAP plots not found. Run explain_clinical_shap.py to generate them."
        )
    st.markdown("### Image explainability / Grad-CAM")
    st.info(
        "Grad-CAM is pending compatibility checks for the 3D segmentation model. "
        "No synthetic heatmap is generated."
    )

elif page == "About Project":
    st.subheader("About the Project")
    st.markdown("""
**Objective:** Explore explainable AI methods for pancreatic CT imaging and clinical metadata.

**Current verified components**
- Clinical metadata Logistic Regression baseline.
- Saved clinical evaluation metrics and test predictions.
- Clinical feature-coefficient explanation report.
- 3D U-Net segmentation model and a saved experimental prediction.
- NIfTI CT slice viewing.

**Pending**
- Verified CT-based cancer classification.
- Valid patient-level matched CT and clinical training records.
- Trained and evaluated multimodal fusion model.
- Compatible image Grad-CAM.
- SHAP explanations for sampled held-out clinical test examples.
- Complete CT segmentation evaluation integrated into this dashboard.

**Important:** The clinical dataset baseline uses age, sex and scanner metadata. It does not use the uploaded CT scan to make its clinical model score.
""")

if model_error:
    st.error(f"Clinical model could not be loaded: {model_error}")


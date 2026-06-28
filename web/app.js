document.addEventListener("DOMContentLoaded", () => {

    const fileInput = document.getElementById("fileInput");
    const fileNameDiv = document.getElementById("fileName");
    const uploadBtn = document.getElementById("uploadBtn");
    const modelSelect = document.getElementById("modelSelect");

    const loadingDiv = document.getElementById("loading");

    const originalPreview = document.getElementById("originalPreview");
    const resultOutput = document.getElementById("resultOutput");

    const gradcamContainer = document.getElementById("gradcamContainer");
    const gradcamImg = document.getElementById("gradcamImg");

    const patientInputContainer = document.getElementById("patientInputContainer");

    // 🔥 correction
    const correctionPanel = document.getElementById("correctionPanel");
    const predictionIdInput = document.getElementById("predictionIdInput");
    const correctedResultInput = document.getElementById("correctedResultInput");
    const submitCorrectionBtn = document.getElementById("submitCorrectionBtn");

    const BASE_URL = "http://127.0.0.1:8000";

    const token = (localStorage.getItem("token") || "").replace(/^["']|["']$/g, '');
    const userRole = (localStorage.getItem("role") || "").replace(/^["']|["']$/g, '').toLowerCase();

    if (!token) {
        alert("Login required");
        window.location.href = "login.html";
        return;
    }

    if (userRole === 'doctor') {
        patientInputContainer.innerHTML = `
            <input type="text" id="patientNameInput" placeholder="Patient Name">
        `;
    }
     // بعد
const roleBadge = document.getElementById('roleBadge');
const roleLabel = document.getElementById('roleLabel');
if (userRole === 'doctor') {
    roleBadge.className = 'role-badge role-doctor';
    roleLabel.innerHTML = ' طبيب';
} else {
    roleBadge.className = 'role-badge role-patient';
    roleLabel.innerHTML = ' مريض';
}


    fileInput.addEventListener("change", (e) => {
        const file = e.target.files[0];
        if (!file) return;

        fileNameDiv.textContent = file.name;

        const reader = new FileReader();
        reader.onload = (event) => {
            originalPreview.innerHTML = `<img src="${event.target.result}">`;
        };
        reader.readAsDataURL(file);
    });

    uploadBtn.addEventListener("click", async () => {

        const file = fileInput.files[0];
        if (!file) return alert("Select file first");

        const formData = new FormData();
        formData.append("file", file);

        if (userRole === "doctor") {
            const name = document.getElementById("patientNameInput").value;
            if (!name) return alert("Enter patient name");
            formData.append("patient_name", name);
        }

        loadingDiv.style.display = "block";
        correctionPanel.style.display = "none";

        try {

            const response = await fetch(`${BASE_URL}${modelSelect.value}`, {
                method: "POST",
                headers: {
                    "Authorization": `Bearer ${token}`
                },
                body: formData
            });

            const data = await response.json();

            if (!response.ok) throw new Error(data.detail || "Error");

            loadingDiv.style.display = "none";

            resultOutput.innerHTML = `
                <p><b>Prediction:</b> ${data.prediction}</p>
                <p><b>Confidence:</b> ${data.confidence}</p>
                <p><b>ID:</b> ${data.prediction_id}</p>
            `;

            // gradcam
            if (data.gradcam_image) {
                gradcamImg.src = data.gradcam_image;
                gradcamContainer.style.display = "block";
            }

            // 🔥 show correction for doctor
            if (userRole === "doctor") {
                correctionPanel.style.display = "block";
                predictionIdInput.value = data.prediction_id || "";
            }

        } catch (err) {
            loadingDiv.style.display = "none";
            resultOutput.innerHTML = `<p style="color:red;">${err.message}</p>`;
        }
    });

    // 🔥 correction submit
    submitCorrectionBtn.addEventListener("click", async () => {

        const id = predictionIdInput.value;
        const corrected = correctedResultInput.value;

        if (!id || !corrected) {
            return alert("Fill all fields");
        }

        try {

            const formData = new FormData();
            formData.append("corrected_result", corrected);

            const res = await fetch(`${BASE_URL}/correction/${id}`, {
                method: "PUT",
                headers: {
                    "Authorization": `Bearer ${token}`
                },
                body: formData
            });

            const data = await res.json();

            if (!res.ok) throw new Error(data.detail || "Failed");

            alert("Correction saved ✅");
            correctedResultInput.value = "";

        } catch (e) {
            alert(e.message);
        }
    });

});
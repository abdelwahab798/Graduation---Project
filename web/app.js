const BACKEND_BASE_URL = "http://127.0.0.1:8000";

// عناصر واجهة المستخدم (DOM Elements)
const fileInput = document.getElementById('fileInput');
const fileName = document.getElementById('fileName');
const uploadBtn = document.getElementById('uploadBtn');
const modelSelect = document.getElementById('modelSelect');
const loading = document.getElementById('loading');
const originalPreview = document.getElementById('originalPreview');
const resultOutput = document.getElementById('resultOutput');
const gradcamContainer = document.getElementById('gradcamContainer');
const gradcamImg = document.getElementById('gradcamImg');

fileInput.addEventListener('change', function() {
    if (this.files && this.files[0]) {
        const file = this.files[0];
        fileName.textContent = file.name;

        const reader = new FileReader();
        reader.onload = function(e) {
            originalPreview.innerHTML = `<img src="${e.target.result}" style="width:100%; max-height:400px; object-fit:contain; border-radius:5px;">`;
        }
        reader.readAsDataURL(file);
    }
});

// 3. دالة إرسال الطلب (HTTP POST Request) للـ API
uploadBtn.addEventListener('click', async () => {
    const file = fileInput.files[0];
    if (!file) {
        alert("من فضلك اختر صورة أشعة أولاً!");
        return;
    }

    // تجهيز الـ FormData لإرسال الملف
    const formData = new FormData();
    formData.append("file", file);

    // إظهار الـ Loading واخفاء النتائج السابقة
    loading.style.display = "block";
    resultOutput.innerHTML = `<p style="color:#666;">جاري التحليل الحسابي...</p>`;
    gradcamContainer.style.display = "none";

    // استخراج الـ Endpoint المختار ديناميكياً
    const targetEndpoint = modelSelect.value;
    const fullUrl = `${BACKEND_BASE_URL}${targetEndpoint}`;

    try {
        const response = await fetch(fullUrl, {
            method: "POST",
            body: formData
        });

        if (response.ok) {
            const data = await response.json();
            
            // إخفاء الـ Loading بعد نجاح الطلب
            loading.style.display = "none";

            // عرض البيانات والتشخيص داخل الـ UI
            let probabilitiesHtml = '';
            if (data.all_probabilities) {
                probabilitiesHtml = `<div style="margin-top:10px; font-size:0.95rem; color:#555;">`;
                for (const [key, value] of Object.entries(data.all_probabilities)) {
                    probabilitiesHtml += `<p>• ${key}: <strong>${(value * 100).toFixed(2)}%</strong></p>`;
                }
                probabilitiesHtml += `</div>`;
            }

            resultOutput.innerHTML = `
                <p>اسم الملف: <strong>${data.filename}</strong></p>
                <p>التشخيص النهائي: <span class="badge-success">${data.prediction}</span></p>
                <p>نسبة التأكيد الكلية: <strong>${(data.confidence * 100).toFixed(2)}%</strong></p>
                ${probabilitiesHtml}
            `;

            // عرض صورة الـ Grad-CAM المشفّرة بـ Base64
            if (data.gradcam_image) {
                gradcamImg.src = data.gradcam_image;
                gradcamContainer.style.display = "block";
            }

        } else {
            const errorData = await response.json();
            throw new Error(errorData.detail || "حدث خطأ غير معروف في السيرفر.");
        }

    } catch (error) {
        loading.style.display = "none";
        resultOutput.innerHTML = `<p style="color: red; font-weight: bold;">❌ خطأ أثناء الفحص: ${error.message}</p>`;
        console.error(error);
    }
});
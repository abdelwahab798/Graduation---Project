/* -------------------------------------------------------------
   المشروع: MediScan-AI - ملف الفحص والربط مع الـ APIs والتحكم في عناصر التشخيص
   ------------------------------------------------------------- */




const BACKEND_BASE_URL = "http://127.0.0.1:8000";

// جلب العناصر الأساسية من الـ HTML بناءً على كود عبد الوهاب
const fileInput = document.getElementById('fileInput');
const fileName = document.getElementById('fileName');
const uploadBtn = document.getElementById('uploadBtn');
const modelSelect = document.getElementById('modelSelect');
const loading = document.getElementById('loading');
const originalPreview = document.getElementById('originalPreview');
const resultOutput = document.getElementById('resultOutput');
const gradcamContainer = document.getElementById('gradcamContainer');
const gradcamImg = document.getElementById('gradcamImg');

// تحديد حاوية النتيجة الخارجية للتحكم في لون توهج الحدود تلقائياً (CSS نيون)
const resultCard = document.querySelector('.card-result') || document.getElementById('result-section') || document.querySelector('.glass-container');

// 1. الاستماع لاختيار ملف الأشعة وعرض المعاينة فوراً
if (fileInput) {
    fileInput.addEventListener('change', function() {
        if (this.files && this.files[0]) {
            const file = this.files[0];
            if (fileName) fileName.textContent = file.name;
            
            const reader = new FileReader();
            reader.onload = function(e) {
                if (originalPreview) {
                    originalPreview.innerHTML = `<img src="${e.target.result}" style="width:100%; max-height:400px; object-fit:contain; border-radius:12px; border: 1px solid var(--glass-border);">`;
                }
            }
            reader.readAsDataURL(file);
        }
    });
}

// 2. بدء عملية الفحص عند الضغط على زر الرفع والربط مع السيرفر (Fetch API)
if (uploadBtn) {
    uploadBtn.addEventListener('click', async () => {
        const file = fileInput ? fileInput.files[0] : null;
        if (!file) {
            alert("من فضلك اختر صورة أشعة أولاً");
            return;
        }

        const formData = new FormData();
        formData.append("file", file);

        // تشغيل وضع التحميل وإخفاء النتائج السابقة مؤقتاً
        if (loading) loading.style.display = "block";
        if (resultOutput) resultOutput.innerHTML = `<p style="color: var(--text-secondary);">جاري التحليل الحسابي وفحص الأنماط الطبية برمجياً... ⚡</p>`;
        if (gradcamContainer) gradcamContainer.style.display = "none";

        // إعادة تعيين الحدود الافتراضية للكرت أثناء فترة التحميل
        if (resultCard) {
            resultCard.style.borderColor = "var(--glass-border)";
            resultCard.style.boxShadow = "0 30px 60px rgba(0, 0, 0, 0.7)";
        }

        const targetEndpoint = modelSelect ? modelSelect.value : "/api/predict";
        const fullUrl = `${BACKEND_BASE_URL}${targetEndpoint}`;

        try {
            // إرسال طلب fetch إلى سيرفر عبد الوهاب
            const response = await fetch(fullUrl, {
                method: "POST",
                body: formData
            });

            if (response.ok) {
                const data = await response.json();

                if (loading) loading.style.display = "none";

                // حساب وعرض تفاصيل الاحتمالات تفصيلياً (Response البيانات)
                let probabilitiesHtml = '';
                if (data.all_probabilities) {
                    probabilitiesHtml = `<div style="margin-top:15px; font-size:0.95rem; color: var(--text-secondary); text-align: right; width: 100%;">`;
                    for (const [key, value] of Object.entries(data.all_probabilities)) {
                        probabilitiesHtml += `<p style="margin-bottom: 5px;">• ${key}: <strong style="color: #ffffff;">${(value * 100).toFixed(2)}%</strong></p>`;
                    }
                    probabilitiesHtml += `</div>`;
                }

                // التحقق من حالة التشخيص لتطبيق التلوين النيوني التفاعلي
                const predictionText = data.prediction;
                const isNormal = predictionText.toLowerCase().includes('normal') || predictionText.includes('سليم') || predictionText.includes('طبيعي');

                // تعديل الاستايل نيون في نفس اللحظة بناءً على النتيجة بالـ CSS فقط عبر الـ JS
                if (resultCard) {
                    if (isNormal) {
                        resultCard.style.borderColor = "var(--neon-medical)";
                        resultCard.style.boxShadow = "0 0 25px rgba(0, 255, 204, 0.3)";
                    } else {
                        resultCard.style.borderColor = "var(--neon-danger)";
                        resultCard.style.boxShadow = "0 0 25px rgba(255, 51, 102, 0.4)";
                    }
                }

                // تحديد لون الشارة المحيطة بنص التشخيص (أخضر لو سليم، أحمر لو في مشكلة)
                const badgeColor = isNormal ? "var(--neon-medical)" : "var(--neon-danger)";

                // طباعة مخرجات الـ Response بالكامل في الـ HTML
                if (resultOutput) {
                    resultOutput.innerHTML = `
                        <div style="width: 100%; text-align: right; line-height: 1.8;">
                            <p style="margin-bottom: 8px;">اسم الملف المرفوع: <strong style="color: #ffffff;">${data.filename}</strong></p>
                            <p style="margin-bottom: 8px;">التشخيص السريري: <span style="background: ${badgeColor}; color: #020617; padding: 4px 12px; border-radius: 6px; font-weight: 700; box-shadow: 0 0 10px ${badgeColor};">${data.prediction}</span></p>
                            <p style="margin-bottom: 8px;">نسبة التأكيد الكلية للنموذج: <strong style="color: var(--accent-blue); text-shadow: 0 0 5px var(--accent-blue);">${(data.confidence * 100).toFixed(2)}%</strong></p>
                            <hr style="border-color: var(--glass-border); margin: 15px 0;">
                            <p style="font-weight: bold; color: #ffffff;">التحليل الإحصائي التفصيلي للاحتمالات:</p>
                            ${probabilitiesHtml}
                        </div>
                    `;
                }

                // عرض صورة الـ GradCAM التفسيرية إذا كانت راجعة من السيرفر
                if (data.gradcam_image && gradcamImg && gradcamContainer) {
                    gradcamImg.src = data.gradcam_image;
                    gradcamImg.style.borderRadius = "12px";
                    gradcamImg.style.border = "1px solid var(--glass-border)";
                    gradcamContainer.style.display = "block";
                }

            } else {
                const errorData = await response.json();
                throw new Error(errorData.detail || "حدث خطأ غير معروف في السيرفر.");
            }

        } catch (error) {
            if (loading) loading.style.display = "none";
            if (resultOutput) {
                resultOutput.innerHTML = `<p style="color: var(--neon-danger); font-weight: bold; text-shadow: 0 0 5px var(--neon-danger);">⚠️ خطأ أثناء الفحص: ${error.message}</p>`;
            }
            console.error(error);
        }
    });
}
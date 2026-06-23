/* -------------------------------------------------------------
   المشروع: MediScan-AI - ملف التأثيرات البصرية الموحد والآمن
   ------------------------------------------------------------- */

document.addEventListener('DOMContentLoaded', () => {
    
    // =============================================================
    // 1. أنميشن خط نبضات القلب الموحد بملء الشاشة (Canvas Background)
    // =============================================================
    const canvas = document.getElementById('heartbeatCanvas');
    
    // شغال فقط لو العنصر موجود في الصفحة الحالية، ولو مش موجود مش هيعطل باقي الألوان
    if (canvas) {
        const ctx = canvas.getContext('2d');
        
        function resizeCanvas() {
            canvas.width = window.innerWidth;
            canvas.height = window.innerHeight;
        }
        resizeCanvas();
        window.addEventListener('resize', resizeCanvas);

        let x = 0;
        const points = [];
        const speed = 2.5; 

        function drawHeartbeat() {
            if (!document.getElementById('heartbeatCanvas')) return; // أمان إضافي لو تم تغيير الصفحة
            requestAnimationFrame(drawHeartbeat);
            
            ctx.fillStyle = 'rgba(2, 6, 23, 0.06)'; 
            ctx.fillRect(0, 0, canvas.width, canvas.height);

            let centerY = canvas.height / 2;
            let y = centerY; 
            
            let cycle = x % 280;
            if (cycle > 40 && cycle < 50) {
                y -= 30; 
            } else if (cycle >= 50 && cycle < 54) {
                y += 12; 
            } else if (cycle >= 54 && cycle < 64) {
                y -= 75; 
            } else if (cycle >= 64 && cycle < 70) {
                y += 60; 
            } else if (cycle >= 85 && cycle < 105) {
                y -= 22; 
            }

            points.push({ x, y });
            if (points.length > canvas.width / speed) {
                points.shift();
            }

            ctx.beginPath();
            ctx.strokeStyle = '#00ffcc'; 
            ctx.lineWidth = 2.5;
            ctx.shadowBlur = 15;
            ctx.shadowColor = '#00ffcc';

            for (let i = 0; i < points.length; i++) {
                let ptX = points[i].x - x + canvas.width - 100;
                if (i === 0) {
                    ctx.moveTo(ptX, points[i].y);
                } else {
                    ctx.lineTo(ptX, points[i].y);
                }
            }
            ctx.stroke();
            x += speed;
        }
        drawHeartbeat();
    }

    // =============================================================
    // 2. تأثير وميض وتفاعل مربع الرفع عند سحب وإفلات الصورة (Drag & Drop)
    // =============================================================
    const dropZone = document.querySelector('.card') || document.getElementById('originalPreview');
    const fileInput = document.getElementById('fileInput');

    if (dropZone && fileInput) {
        window.addEventListener('dragover', (e) => {
            e.preventDefault();
            dropZone.style.borderColor = "var(--accent-blue)";
            dropZone.style.boxShadow = "0 0 25px rgba(0, 136, 255, 0.4), inset 0 0 15px rgba(0, 136, 255, 0.2)";
            dropZone.style.transform = "scale(1.02)";
        });

        window.addEventListener('dragleave', (e) => {
            e.preventDefault();
            dropZone.style.borderColor = "rgba(255, 255, 255, 0.08)";
            dropZone.style.boxShadow = "none";
            dropZone.style.transform = "scale(1)";
        });

        window.addEventListener('drop', (e) => {
            e.preventDefault();
            dropZone.style.borderColor = "rgba(255, 255, 255, 0.08)";
            dropZone.style.boxShadow = "none";
            dropZone.style.transform = "scale(1)";
            
            if (e.dataTransfer.files.length > 0) {
                fileInput.files = e.dataTransfer.files;
                fileInput.dispatchEvent(new Event('change'));
            }
        });
    }

    // =============================================================
    // 3. تأثير وميض الأزرار النيون عند الضغط
    // =============================================================
    const uploadBtn = document.getElementById('uploadBtn');
    if (uploadBtn) {
        uploadBtn.addEventListener('mousedown', () => {
            uploadBtn.style.transform = "scale(0.98)";
            uploadBtn.style.boxShadow = "0 2px 10px rgba(0, 136, 255, 0.5)";
        });

        uploadBtn.addEventListener('mouseup', () => {
            uploadBtn.style.transform = "scale(1)";
            uploadBtn.style.boxShadow = "0 6px 20px rgba(0, 136, 255, 0.3)";
        });
    }
});
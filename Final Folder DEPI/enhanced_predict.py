import torch
import torchvision.models as models
from PIL import Image, ImageTk, ImageDraw, ImageFont, ImageEnhance, ImageFilter
from torchvision import transforms
import tkinter as tk
from tkinter import filedialog, ttk
import numpy as np
import tifffile
import cv2
import csv

# ================= تحميل الموديل =================
num_classes = 10

model = models.efficientnet_b0(weights=None)
model.classifier = torch.nn.Sequential(
    torch.nn.Dropout(p=0.2, inplace=True),
    torch.nn.Linear(model.classifier[1].in_features, 512),
    torch.nn.ReLU(),
    torch.nn.Dropout(p=0.3),
    torch.nn.Linear(512, num_classes)
)

model.load_state_dict(torch.load("best_efficientnet_b0_model.pth", map_location=torch.device('cpu')))
model.eval()
print("✅ Model loaded successfully and ready for predictions")

CLASS_NAMES = [
    'AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway',
    'Industrial', 'Pasture', 'PermanentCrop', 'Residential',
    'River', 'SeaLake'
]

# أيقونات إيموجي لكل فئة
CLASS_ICONS = {
    'AnnualCrop': '🌾',
    'Forest': '🌲',
    'HerbaceousVegetation': '🌿',
    'Highway': '🛣️',
    'Industrial': '🏭',
    'Pasture': '🐄',
    'PermanentCrop': '🌳',
    'Residential': '🏘️',
    'River': '🌊',
    'SeaLake': '🏞️'
}

# ألوان لكل فئة
CLASS_COLORS = {
    'AnnualCrop': '#FFA726',
    'Forest': '#66BB6A',
    'HerbaceousVegetation': '#9CCC65',
    'Highway': '#78909C',
    'Industrial': '#EF5350',
    'Pasture': '#FFEB3B',
    'PermanentCrop': '#8D6E63',
    'Residential': '#42A5F5',
    'River': '#29B6F6',
    'SeaLake': '#26C6DA'
}

# ================= دالة تحسين الصورة =================
def enhance_image(img_array):
    """تحسين جودة الصورة والبكسل"""
    # تحويل إلى PIL
    img = Image.fromarray(img_array)
    
    # 1. تحسين الحدة (Sharpness)
    enhancer = ImageEnhance.Sharpness(img)
    img = enhancer.enhance(1.5)
    
    # 2. تحسين الت대조(Contrast)
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.2)
    
    # 3. تحسين السطوع (Brightness)
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(1.1)
    
    # 4. تحسين الألوان (Color)
    enhancer = ImageEnhance.Color(img)
    img = enhancer.enhance(1.15)
    
    # 5. تطبيق فلتر UnsharpMask لمزيد من الوضوح
    img = img.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
    
    return np.array(img)

# ================= دالة تكبير الصورة بجودة عالية =================
def upscale_image(img_array, scale_factor=2):
    """تكبير الصورة باستخدام LANCZOS interpolation"""
    img = Image.fromarray(img_array)
    new_size = (img.width * scale_factor, img.height * scale_factor)
    img_upscaled = img.resize(new_size, Image.Resampling.LANCZOS)
    return np.array(img_upscaled)
def read_image_auto(path):
    try:
        img = Image.open(path)
        if img.mode != "RGB":
            img = img.convert("RGB")
        arr = np.array(img)
    except Exception:
        arr = tifffile.imread(path)
        if arr.ndim == 3 and arr.shape[0] == 13:
            arr = np.moveaxis(arr, 0, -1)
        if arr.shape[2] >= 4:
            arr = np.stack([arr[:, :, 3], arr[:, :, 2], arr[:, :, 1]], axis=-1)
        else:
            arr = np.repeat(arr[:, :, 0:1], 3, axis=2)
        arr = arr.astype(np.float32)
        arr /= (arr.max() + 1e-6)
        arr = (arr * 255).astype(np.uint8)
    return arr

# ================= دالة التنبؤ مع الاحتمالات =================
def predict_image(image_path):
    arr = read_image_auto(image_path)
    img = Image.fromarray(arr)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    img_t = transform(img).unsqueeze(0)
    with torch.no_grad():
        outputs = model(img_t)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        predicted = outputs.argmax(dim=1).item()
        confidence = probabilities[0][predicted].item()
    
    pred_label = CLASS_NAMES[predicted]
    
    # احصل على أعلى 3 تنبؤات
    top3_prob, top3_idx = torch.topk(probabilities, 3)
    top3_results = [(CLASS_NAMES[idx.item()], prob.item()) for idx, prob in zip(top3_idx[0], top3_prob[0])]
    
    return arr, pred_label, confidence, top3_results

# ================= عرض الصورة بحجم كامل =================
def show_fullscreen_image(img_array):
    """عرض الصورة بحجم كامل في نافذة منفصلة"""
    fullscreen_window = tk.Toplevel(root)
    fullscreen_window.title("🔍 عرض الصورة بالحجم الكامل")
    fullscreen_window.configure(bg='#2c3e50')
    
    # الحصول على حجم الشاشة
    screen_width = fullscreen_window.winfo_screenwidth()
    screen_height = fullscreen_window.winfo_screenheight()
    
    # تكبير الصورة لتناسب الشاشة
    img = Image.fromarray(img_array)
    img_ratio = img.width / img.height
    
    # حساب الحجم المناسب (مع ترك هوامش)
    max_width = int(screen_width * 0.9)
    max_height = int(screen_height * 0.9)
    
    if img_ratio > max_width / max_height:
        new_width = max_width
        new_height = int(max_width / img_ratio)
    else:
        new_height = max_height
        new_width = int(max_height * img_ratio)
    
    img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    # إضافة إطار ملون
    bordered_width = new_width + 20
    bordered_height = new_height + 20
    bordered_img = Image.new('RGB', (bordered_width, bordered_height), '#3498db')
    bordered_img.paste(img_resized, (10, 10))
    
    tk_img = ImageTk.PhotoImage(bordered_img)
    
    # إنشاء إطار قابل للتمرير
    canvas = tk.Canvas(fullscreen_window, bg='#2c3e50', highlightthickness=0)
    scrollbar_y = tk.Scrollbar(fullscreen_window, orient="vertical", command=canvas.yview)
    scrollbar_x = tk.Scrollbar(fullscreen_window, orient="horizontal", command=canvas.xview)
    
    canvas.configure(yscrollcommand=scrollbar_y.set, xscrollcommand=scrollbar_x.set)
    
    scrollbar_y.pack(side="right", fill="y")
    scrollbar_x.pack(side="bottom", fill="x")
    canvas.pack(side="left", fill="both", expand=True)
    
    img_label = tk.Label(canvas, image=tk_img, bg='#2c3e50')
    img_label.image = tk_img
    canvas.create_window(0, 0, anchor="nw", window=img_label)
    canvas.config(scrollregion=canvas.bbox("all"))
    
    # زر الإغلاق
    close_btn = tk.Button(
        fullscreen_window,
        text="✖ إغلاق",
        command=fullscreen_window.destroy,
        font=("Arial", 12, "bold"),
        bg='#e74c3c',
        fg='white',
        relief='flat',
        cursor='hand2'
    )
    close_btn.place(x=20, y=20)
    
    # مركز النافذة
    window_width = min(bordered_width + 40, screen_width)
    window_height = min(bordered_height + 80, screen_height)
    x = (screen_width - window_width) // 2
    y = (screen_height - window_height) // 2
    fullscreen_window.geometry(f"{window_width}x{window_height}+{x}+{y}")
    
    # اختصار لوحة المفاتيح للإغلاق
    fullscreen_window.bind('<Escape>', lambda e: fullscreen_window.destroy())

# متغير عام للصورة الحالية
current_image = None
last_file_path = None
batch_paths = []
batch_index = -1
batch_results = []  # تخزين نتائج الدُفعة
def predict_and_render(file_path):
    global current_image, last_file_path
    progress_label.config(text="🔄 جاري التحليل...")
    root.update()
    arr, pred_label, confidence, top3_results = predict_image(file_path)
    last_file_path = file_path
    current_image = arr
    img = Image.fromarray(arr).resize((400, 400), Image.Resampling.LANCZOS)
    bordered_img = Image.new('RGB', (420, 420), CLASS_COLORS[pred_label])
    bordered_img.paste(img, (10, 10))
    tk_img = ImageTk.PhotoImage(bordered_img)
    image_label.config(image=tk_img)
    image_label.image = tk_img
    icon = CLASS_ICONS.get(pred_label, '📍')
    result_label.config(
        text=f"{icon} {pred_label}",
        fg=CLASS_COLORS[pred_label]
    )
    confidence_label.config(text="")
    top3_text = "Top 3 Predictions:\n\n"
    for i, (label, prob) in enumerate(top3_results, 1):
        icon = CLASS_ICONS.get(label, '📍')
        bar_length = int(prob * 30)
        bar = "█" * bar_length + "░" * (30 - bar_length)
        top3_text += f"{i}. {icon} {label}\n   {bar} {prob*100:.1f}%\n"
    top3_label.config(text=top3_text)
    # حفظ/تحديث نتيجة هذه الصورة
    found = False
    for r in batch_results:
        if r['file'] == file_path:
            r.update({
                'prediction': pred_label,
                'confidence': confidence,
                'top3': top3_results,
            })
            found = True
            break
    if not found:
        batch_results.append({
            'file': file_path,
            'prediction': pred_label,
            'confidence': confidence,
            'top3': top3_results,
        })
    if batch_paths:
        progress_label.config(text=f"✅ {batch_index+1}/{len(batch_paths)}")
    else:
        progress_label.config(text="✅ تم التحليل بنجاح!")
def choose_and_predict():
    file_path = filedialog.askopenfilename(
        title="اختر صورة للتحليل",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff")]
    )
    if file_path:
        predict_and_render(file_path)
        
    else:
        progress_label.config(text="⚠️ لم يتم اختيار أي صورة.")

def choose_and_predict_multiple():
    global batch_paths, batch_index
    files = filedialog.askopenfilenames(
        title="اختر صور متعددة للتحليل",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff")]
    )
    if files:
        batch_paths = list(files)
        batch_index = 0
        # إعادة تعيين النتائج السابقة
        global batch_results
        batch_results = []
        prev_button.config(state='normal' if len(batch_paths) > 1 else 'disabled')
        next_button.config(state='normal' if len(batch_paths) > 1 else 'disabled')
        predict_and_render(batch_paths[batch_index])
    else:
        progress_label.config(text="⚠️ لم يتم اختيار أي صور.")

def show_prev():
    global batch_index
    if batch_paths and batch_index > 0:
        batch_index -= 1
        predict_and_render(batch_paths[batch_index])

def show_next():
    global batch_index
    if batch_paths and batch_index < len(batch_paths) - 1:
        batch_index += 1
        predict_and_render(batch_paths[batch_index])

def clear_ui():
    global current_image, last_file_path, batch_paths, batch_index, batch_results
    current_image = None
    last_file_path = None
    batch_paths = []
    batch_index = -1
    batch_results = []
    image_label.config(image='')
    image_label.image = None
    result_label.config(text="", fg='#34495e')
    confidence_label.config(text="")
    top3_label.config(text="")
    progress_label.config(text="🧹 تم التنظيف. اختر صورة للبدء")
    prev_button.config(state='disabled')
    next_button.config(state='disabled')

def save_results_csv():
    if not batch_results:
        progress_label.config(text="⚠️ لا توجد نتائج لحفظها.")
        return
    save_path = filedialog.asksaveasfilename(
        title="حفظ نتائج الدُفعة",
        defaultextension=".csv",
        filetypes=[("CSV", "*.csv")],
        initialfile="results.csv"
    )
    if not save_path:
        progress_label.config(text="⚠️ لم يتم اختيار مسار للحفظ.")
        return
    headers = [
        'file', 'prediction', 'confidence',
        'top1_label', 'top1_prob',
        'top2_label', 'top2_prob',
        'top3_label', 'top3_prob'
    ]
    with open(save_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for r in batch_results:
            t3 = r.get('top3', [])
            row = [
                r.get('file', ''),
                r.get('prediction', ''),
                f"{r.get('confidence', 0.0):.6f}",
            ]
            # تعبئة أعلى 3
            for i in range(3):
                if i < len(t3):
                    row.append(t3[i][0])
                    row.append(f"{t3[i][1]:.6f}")
                else:
                    row.extend(['', ''])
            writer.writerow(row)
    progress_label.config(text=f"✅ تم حفظ النتائج: {save_path}")

# ================= واجهة المستخدم المحسّنة =================
root = tk.Tk()
root.title("🌍 Land Type Classification System")
root.geometry("700x1000")
root.configure(bg='#f0f4f8')
root.resizable(True, True)  # السماح بتكبير النافذة


# إطار العنوان
title_frame = tk.Frame(root, bg='#2c3e50')
title_frame.pack(fill='x', pady=(0, 15))

title_label = tk.Label(
    title_frame,
    text="🌍 Land Type Classification",
    font=("Arial", 22, "bold"),
    bg='#2c3e50',
    fg='white'
)
title_label.pack(pady=(15, 5))

subtitle_label = tk.Label(
    title_frame,
    text="Powered by EfficientNet-B0 | AI-Driven Analysis | Enhanced Quality",
    font=("Arial", 9),
    bg='#2c3e50',
    fg='#95a5a6'
)
subtitle_label.pack(pady=(0, 15))

# إطار الأزرار
button_frame = tk.Frame(root, bg='#f0f4f8')
button_frame.pack(pady=15, fill='x')

choose_button = tk.Button(
    button_frame,
    text="📁 اختر صورة جديدة",
    command=choose_and_predict,
    width=22,
    height=2,
    font=("Arial", 13, "bold"),
    bg='#3498db',
    fg='white',
    relief='flat',
    cursor='hand2',
    activebackground='#2980b9',
    bd=0
)
choose_button.pack(side='left', padx=5)

multi_button = tk.Button(
    button_frame,
    text="📂 اختر صور متعددة",
    command=choose_and_predict_multiple,
    width=22,
    height=2,
    font=("Arial", 13, "bold"),
    bg='#8e44ad',
    fg='white',
    relief='flat',
    cursor='hand2',
    activebackground='#7d3c98',
    bd=0
)
multi_button.pack(side='left', padx=5)

prev_button = tk.Button(
    button_frame,
    text="◀ السابق",
    command=show_prev,
    width=12,
    height=2,
    font=("Arial", 12, "bold"),
    bg='#7f8c8d',
    fg='white',
    relief='flat',
    cursor='hand2',
    activebackground='#707b7c',
    bd=0
)
prev_button.pack(side='left', padx=5)
prev_button.config(state='disabled')

next_button = tk.Button(
    button_frame,
    text="التالي ▶",
    command=show_next,
    width=12,
    height=2,
    font=("Arial", 12, "bold"),
    bg='#7f8c8d',
    fg='white',
    relief='flat',
    cursor='hand2',
    activebackground='#707b7c',
    bd=0
)
next_button.pack(side='left', padx=5)
next_button.config(state='disabled')

# زر حفظ النتائج CSV
save_button = tk.Button(
    button_frame,
    text="💾 حفظ النتائج CSV",
    command=save_results_csv,
    width=18,
    height=2,
    font=("Arial", 12, "bold"),
    bg='#27ae60',
    fg='white',
    relief='flat',
    cursor='hand2',
    activebackground='#1e8449',
    bd=0
)
save_button.pack(side='left', padx=5)

# زر تنظيف الواجهة
clear_button = tk.Button(
    button_frame,
    text="🧹 تنظيف",
    command=clear_ui,
    width=10,
    height=2,
    font=("Arial", 12, "bold"),
    bg='#e74c3c',
    fg='white',
    relief='flat',
    cursor='hand2',
    activebackground='#c0392b',
    bd=0
)
clear_button.pack(side='left', padx=5)
 



# شريط التقدم
progress_label = tk.Label(
    root,
    text="👆 اضغط لاختيار صورة",
    font=("Arial", 10),
    bg='#f0f4f8',
    fg='#7f8c8d'
)
progress_label.pack(pady=(8, 15))

# إطار الصورة
image_frame = tk.Frame(root, bg='#ecf0f1', relief='solid', borderwidth=0)
image_frame.pack(pady=5, fill='both', expand=True)

image_label = tk.Label(image_frame, bg='#ecf0f1')
image_label.pack(padx=10, pady=10)

# إطار النتائج
results_frame = tk.Frame(root, bg='#f0f4f8')
results_frame.pack(pady=15, fill='both', expand=True, padx=20)

# النتيجة الرئيسية
result_label = tk.Label(
    results_frame,
    text="",
    font=("Arial", 26, "bold"),
    bg='#f0f4f8'
)
result_label.pack(pady=(5, 8))

confidence_label = tk.Label(
    results_frame,
    text="",
    font=("Arial", 13),
    bg='#f0f4f8',
    fg='#27ae60'
)
confidence_label.pack(pady=(0, 10))

# خط فاصل
separator = tk.Frame(results_frame, height=2, bg='#bdc3c7')
separator.pack(fill='x', padx=40, pady=12)

# أعلى 3 تنبؤات
top3_label = tk.Label(
    results_frame,
    text="",
    font=("Courier", 10),
    bg='#f0f4f8',
    fg='#34495e',
    justify='left'
)
top3_label.pack(pady=5)


# تذييل
footer_label = tk.Label(
    root,
    text="© 2024 Land Classification System | Developed with ❤️",
    font=("Arial", 8),
    bg='#f0f4f8',
    fg='#95a5a6'
)
footer_label.pack(side='bottom', pady=12)

root.mainloop()

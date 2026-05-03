"""
=============================================================================
  WHEAT DISEASE DETECTION SYSTEM — AI-Powered PPT Generator
=============================================================================
  USAGE:
    1. Install requirements:
         pip install python-pptx pillow openai

    2. Set your OpenAI API key:
         Option A: Set environment variable:
                   set OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxx       (Windows CMD)
                   $env:OPENAI_API_KEY="sk-xxxxxxxxxxxxxxxx"   (PowerShell)

         Option B: Paste your key directly in the OPENAI_API_KEY variable below

    3. Run:
         python generate_ppt_chatgpt.py

    4. Output: "Wheat_Disease_Detection_Presentation.pptx" in the same folder
=============================================================================
"""

import os
import json
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt
import textwrap

# ─────────────────────────────────────────────────────
#  CONFIGURATION — SET YOUR OPENAI API KEY HERE
# ─────────────────────────────────────────────────────
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")  # or paste key here: "sk-..."
USE_CHATGPT = bool(OPENAI_API_KEY)  # Automatically uses ChatGPT if key is set

# ─────────────────────────────────────────────────────
#  SLIDE THEME COLORS  (Professional Indigo + Green)
# ─────────────────────────────────────────────────────
COLOR_PRIMARY    = RGBColor(0x43, 0x38, 0xCA)  # Indigo #4338CA
COLOR_SECONDARY  = RGBColor(0x10, 0xB9, 0x81)  # Emerald #10B981
COLOR_DARK       = RGBColor(0x11, 0x18, 0x27)  # Near-black
COLOR_WHITE      = RGBColor(0xFF, 0xFF, 0xFF)
COLOR_LIGHT_GRAY = RGBColor(0xF3, 0xF4, 0xF6)
COLOR_YELLOW     = RGBColor(0xF5, 0x9E, 0x0B)


# ─────────────────────────────────────────────────────
#  SLIDE CONTENT DATA
#  (Used directly OR enriched by ChatGPT if key provided)
# ─────────────────────────────────────────────────────
SLIDE_DATA_DEFAULT = [
    {
        "slide_num": 1,
        "layout": "title",
        "title": "Wheat Vision AI",
        "subtitle": "Wheat Disease Detection System using Deep Learning (ResNet50)",
        "footer": "A Final Year Project | 2024",
        "points": []
    },
    {
        "slide_num": 2,
        "layout": "content"
        "title": "Problem Statement",
        "icon": "🌾",
        "points": [
            "Wheat diseases like Leaf Rust, Loose Smut & Crown Rot destroy millions of tonnes annually",
            "Manual inspection by farm experts is slow, expensive & subjective",
            "Visual similarity between diseases causes frequent misdiagnosis",
            "Rural farmers lack access to trained pathologists",
            "No digital record-keeping of field disease history"
        ]
    },
    {
        "slide_num": 3,
        "layout": "content",
        "title": "Project Objectives",
        "icon": "🎯",
        "points": [
            "Build a ResNet50-based model for wheat disease classification (5 classes)",
            "Achieve > 90% prediction accuracy using Transfer Learning",
            "Deploy as a Flask web application with an intuitive UI",
            "Implement disease severity estimation (Spread % + Severity Level)",
            "Provide expert treatment recommendations for each disease",
            "Build a secure login system with role-based access (Admin/User)",
            "Store disease scan history per user in SQLite database"
        ]
    },
    {
        "slide_num": 4,
        "layout": "two_column",
        "title": "Technology Stack",
        "icon": "⚙️",
        "col1_title": "AI & Backend",
        "col1_points": [
            "ResNet50 — Deep Learning Model",
            "TensorFlow / Keras — ML Framework",
            "Python Flask — Web Server",
            "NumPy — Array Processing",
            "SQLite — Database",
            "Werkzeug — Security & Hashing"
        ],
        "col2_title": "Frontend & Tools",
        "col2_points": [
            "HTML5 + CSS3 + JavaScript",
            "Bootstrap Icons Library",
            "Google Fonts (Inter)",
            "Google Translate API",
            "Python-PPTX (this script!)",
            "Git — Version Control"
        ]
    },
    {
        "slide_num": 5,
        "layout": "content",
        "title": "System Architecture",
        "icon": "🏗️",
        "points": [
            "User uploads leaf image via Browser (HTML5 drag-and-drop UI)",
            "Flask Backend receives image at /predict REST API endpoint",
            "Image preprocessed: Resize 224×224 → Normalize (ResNet50 standard)",
            "ResNet50 model returns 5-class probability vector via model.predict()",
            "np.argmax() identifies the highest-probability disease class",
            "Disease info (symptoms, treatment) retrieved from expert knowledge base",
            "Result returned as JSON → Frontend renders diagnosis cards"
        ]
    },
    {
        "slide_num": 6,
        "layout": "content",
        "title": "ResNet50 — The AI Model",
        "icon": "🧠",
        "points": [
            "ResNet50 = Residual Network with 50 layers (He et al., CVPR 2016)",
            "Skip Connections solve the Vanishing Gradient Problem in deep networks",
            "Pre-trained on ImageNet (14 million images) — Transfer Learning applied",
            "Phase 1: Base layers frozen → Custom head trained (10 epochs, LR=0.001)",
            "Phase 2: Top 40 layers unfrozen → Fine-tuned (50 epochs, LR=0.00001)",
            "Custom Head: GlobalAvgPool → Dense(1024) → Dropout(0.5) → Dense(5 softmax)",
            "Focal Loss used to handle class imbalance in training data"
        ]
    },
    {
        "slide_num": 7,
        "layout": "two_column",
        "title": "Detected Disease Classes",
        "icon": "🔬",
        "col1_title": "Disease Classes (5 Total)",
        "col1_points": [
            "🌱 Healthy Wheat",
            "🟠 Leaf Rust (Puccinia triticina)",
            "⚫ Wheat Loose Smut",
            "🟤 Crown and Root Rot",
            "❓ Unknown (low confidence)"
        ],
        "col2_title": "Model Performance",
        "col2_points": [
            "Input: 224 × 224 × 3 pixel image",
            "Output: 5-class probability vector",
            "Accuracy: > 90% on validation set",
            "Healthy Wheat confidence: > 95%",
            "Inference Time: 200–500 ms (CPU)",
            "Augmentation: rotation, zoom, flip, brightness"
        ]
    },
    {
        "slide_num": 8,
        "layout": "content",
        "title": "Key Application Features",
        "icon": "✨",
        "points": [
            "🔐 Secure Login/Register with Bcrypt password hashing",
            "📸 Upload wheat leaf image (JPG/PNG) for instant diagnosis",
            "📊 Disease probability bar chart for all 5 classes",
            "⚠️ Severity Estimation: Low → Medium → High → Critical",
            "💊 Expert Treatment Card: Symptoms, Medicine, Prevention",
            "📜 Scan History: Track past diagnoses per user",
            "👑 Admin Panel: View all users and all scan histories",
            "🌐 Google Translate integration for multilingual support"
        ]
    },
    {
        "slide_num": 9,
        "layout": "content",
        "title": "Disease Severity Estimation",
        "icon": "📊",
        "points": [
            "Novel heuristic logic built on top of model confidence scores",
            "Formula: Spread % = Base_Factor[Disease] × Confidence_Score",
            "Aggressive diseases (e.g., Loose Smut) get higher Base_Factor",
            "Spread 0%   → 🟢 Healthy (No action needed)",
            "Spread < 15% → 🟡 Low Severity (Monitor)",
            "Spread < 40% → 🟠 Medium Severity (Treat Soon)",
            "Spread < 70% → 🔴 High Severity (Urgent Treatment)",
            "Spread > 70% → ⛔ Critical (Immediate Action Required)"
        ]
    },
    {
        "slide_num": 10,
        "layout": "content",
        "title": "Key Challenges & Solutions",
        "icon": "🔧",
        "points": [
            "Challenge: Keras version mismatch (batch_shape error) → Sol: Custom build_model() rebuilds architecture, weights loaded separately",
            "Challenge: Class imbalance (few Healthy images) → Sol: Focal Loss + Balanced weights + 1.5× boost for minority class",
            "Challenge: Overfitting on small dataset → Sol: Heavy augmentation (±40° rotation, zoom±30%, brightness 70-130%)",
            "Challenge: Ambiguous images (poor lighting) → Sol: Return top-5 probabilities; correct class in top-2",
            "Challenge: Admin vs User security → Sol: New users forcibly assigned 'user' role server-side"
        ]
    },
    {
        "slide_num": 11,
        "layout": "content",
        "title": "Testing Results",
        "icon": "✅",
        "points": [
            "TC-001: User Registration → ✅ PASS",
            "TC-002: Duplicate Registration Check → ✅ PASS",
            "TC-003: Valid Login → ✅ PASS",
            "TC-004: Invalid Password Rejection → ✅ PASS",
            "TC-007: Healthy Wheat Detection → ✅ PASS (>95% confidence)",
            "TC-008: Leaf Rust Detection → ✅ PASS",
            "TC-009: Wheat Loose Smut Detection → ✅ PASS",
            "TC-011: End-to-End Workflow (Register→Login→Upload→Result) → ✅ PASS"
        ]
    },
    {
        "slide_num": 12,
        "layout": "content",
        "title": "Future Scope",
        "icon": "🔮",
        "points": [
            "📱 Mobile App: TensorFlow Lite + React Native for offline field use",
            "🚁 Drone Integration: Connect /predict API to agricultural UAVs for aerial scanning",
            "🌐 Multilingual: Hindi, Punjabi, Telugu translations via Flask-Babel",
            "🔄 Active Learning: User-corrected predictions used to retrain the model",
            "🗺️ Disease Heatmap: GPS-tagged disease outbreak mapping",
            "☁️ Cloud Deployment: Host on AWS/GCP with auto-scaling for large farms",
            "🏥 Integration: Connect with government agricultural disease tracking portals"
        ]
    },
    {
        "slide_num": 13,
        "layout": "title",
        "title": "Thank You!",
        "subtitle": "Wheat Vision AI — Wheat Disease Detection System\nBuilt with ResNet50 + Flask + SQLite",
        "footer": "Questions? We are happy to explain any part of the system!",
        "points": []
    }
]


# ─────────────────────────────────────────────────────
#  OPTIONAL: ENRICH SLIDES WITH CHATGPT
# ─────────────────────────────────────────────────────
def enrich_slides_with_chatgpt(slide_data: list) -> list:
    """Use ChatGPT to improve and enrich slide bullet points."""
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)

        print("🤖 Connecting to ChatGPT to enrich slide content...")

        project_summary = """
        Project: Wheat Disease Detection System (Wheat Vision AI)
        Tech: ResNet50 (Transfer Learning), Flask Python backend, SQLite DB, HTML/CSS/JS frontend
        5 Disease Classes: Healthy Wheat, Leaf Rust, Wheat Loose Smut, Crown & Root Rot, Unknown
        Key Features: Image upload, AI diagnosis, severity estimation, treatment advice, scan history, 
                      secure login, admin panel, Google Translate support
        Target Users: Farmers, agricultural workers, researchers
        """

        enriched_slides = []
        for slide in slide_data:
            if slide.get("points") and slide["layout"] == "content":
                prompt = f"""
You are an expert in creating professional PowerPoint presentations for final year engineering projects.

Project context:
{project_summary}

For the slide titled: "{slide['title']}"
The current bullet points are:
{chr(10).join(f'- {p}' for p in slide['points'])}

Please improve these bullet points for a clear, professional external/viva presentation:
1. Keep each point concise (maximum 15 words per bullet)
2. Use simple language that non-technical evaluators can understand
3. Add a relevant emoji at the start of each point
4. Return exactly {len(slide['points'])} improved bullet points
5. Return ONLY a JSON array of strings, nothing else

Example format: ["Point 1 here", "Point 2 here"]
"""
                try:
                    response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {"role": "system", "content": "You are a professional presentation designer. Always return valid JSON arrays of strings."},
                            {"role": "user", "content": prompt}
                        ],
                        max_tokens=500,
                        temperature=0.6
                    )

                    content = response.choices[0].message.content.strip()
                    # Clean JSON if wrapped in code blocks
                    if "```" in content:
                        content = content.split("```")[1]
                        if content.startswith("json"):
                            content = content[4:]
                    
                    improved_points = json.loads(content)
                    if isinstance(improved_points, list) and len(improved_points) > 0:
                        slide["points"] = improved_points
                        print(f"  ✅ Enriched: {slide['title']}")
                    else:
                        print(f"  ⚠️  Used default for: {slide['title']}")

                except Exception as e:
                    print(f"  ⚠️  ChatGPT error for '{slide['title']}': {e}")

            enriched_slides.append(slide)

        return enriched_slides

    except ImportError:
        print("⚠️  openai package not installed. Run: pip install openai")
        return slide_data


# ─────────────────────────────────────────────────────
#  PPT HELPER FUNCTIONS
# ─────────────────────────────────────────────────────
def set_slide_background(slide, color: RGBColor):
    """Set solid background color for a slide."""
    background = slide.background
    fill = background.fill
    fill.solid()
    fill.fore_color.rgb = color


def add_text_box(slide, text, left, top, width, height,
                 font_size=14, bold=False, color=COLOR_DARK,
                 align=PP_ALIGN.LEFT, italic=False):
    """Add a styled text box to a slide."""
    txBox = slide.shapes.add_textbox(left, top, width, height)
    tf = txBox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.alignment = align

    run = p.add_run()
    run.text = text
    run.font.size = Pt(font_size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.color.rgb = color
    return txBox


def add_rectangle(slide, left, top, width, height, fill_color: RGBColor, line_color=None):
    """Add a filled rectangle shape."""
    shape = slide.shapes.add_shape(
        1,  # MSO_SHAPE_TYPE.RECTANGLE
        left, top, width, height
    )
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill_color
    if line_color:
        shape.line.color.rgb = line_color
    else:
        shape.line.fill.background()
    return shape


def build_title_slide(prs, data):
    """Slide 1 or Last: Full gradient title slide."""
    slide_layout = prs.slide_layouts[6]  # Blank
    slide = prs.slides.add_slide(slide_layout)

    # Background
    set_slide_background(slide, COLOR_PRIMARY)

    W = prs.slide_width
    H = prs.slide_height

    # Decorative top strip (emerald)
    add_rectangle(slide, 0, 0, W, Inches(0.2), COLOR_SECONDARY)

    # Wheat icon placeholder
    add_text_box(slide, "🌾", Inches(4), Inches(1.0), Inches(2.2), Inches(1.2),
                 font_size=54, align=PP_ALIGN.CENTER, color=COLOR_WHITE)

    # Main Title
    add_text_box(slide, data["title"],
                 Inches(1), Inches(2.1), Inches(8.2), Inches(1.2),
                 font_size=38, bold=True, color=COLOR_WHITE, align=PP_ALIGN.CENTER)

    # Subtitle
    add_text_box(slide, data["subtitle"],
                 Inches(1), Inches(3.3), Inches(8.2), Inches(1.1),
                 font_size=17, color=RGBColor(0xC7, 0xD2, 0xFE), align=PP_ALIGN.CENTER)

    # Divider line
    add_rectangle(slide, Inches(3), Inches(4.5), Inches(4), Inches(0.04), COLOR_SECONDARY)

    # Footer
    add_text_box(slide, data.get("footer", ""),
                 Inches(1), Inches(4.7), Inches(8.2), Inches(0.7),
                 font_size=13, color=RGBColor(0xA5, 0xB4, 0xFC), align=PP_ALIGN.CENTER)

    # Bottom strip
    add_rectangle(slide, 0, H - Inches(0.2), W, Inches(0.2), COLOR_SECONDARY)

    print(f"  ✅ Built title slide: {data['title']}")
    return slide


def build_content_slide(prs, data):
    """Standard bullet-point content slide."""
    slide_layout = prs.slide_layouts[6]  # Blank
    slide = prs.slides.add_slide(slide_layout)

    W = prs.slide_width
    H = prs.slide_height

    # White background
    set_slide_background(slide, COLOR_WHITE)

    # Header bar
    add_rectangle(slide, 0, 0, W, Inches(1.2), COLOR_PRIMARY)

    # Slide number pill
    slide_num = data.get("slide_num", "")
    add_text_box(slide, str(slide_num),
                 Inches(0.25), Inches(0.3), Inches(0.5), Inches(0.55),
                 font_size=11, bold=True, color=COLOR_WHITE, align=PP_ALIGN.CENTER)

    # Icon
    icon = data.get("icon", "")
    add_text_box(slide, icon,
                 Inches(0.7), Inches(0.1), Inches(0.8), Inches(0.9),
                 font_size=26, align=PP_ALIGN.CENTER, color=COLOR_WHITE)

    # Title
    add_text_box(slide, data["title"],
                 Inches(1.4), Inches(0.18), Inches(7.8), Inches(0.85),
                 font_size=24, bold=True, color=COLOR_WHITE, align=PP_ALIGN.LEFT)

    # Emerald accent bar
    add_rectangle(slide, Inches(0.4), Inches(1.35), Inches(0.07), Inches(3.5), COLOR_SECONDARY)

    # Bullet points
    points = data.get("points", [])
    y_start = Inches(1.3)
    y_step = Inches(0.55)

    for i, point in enumerate(points[:8]):  # Max 8 bullets
        y = y_start + (i * y_step)
        # Bullet dot
        add_rectangle(slide, Inches(0.55), y + Inches(0.17), Inches(0.09), Inches(0.09), COLOR_PRIMARY)
        # Bullet text
        add_text_box(slide, point,
                     Inches(0.75), y, Inches(8.8), y_step,
                     font_size=14.5, color=COLOR_DARK, align=PP_ALIGN.LEFT)

    # Bottom strip
    add_rectangle(slide, 0, H - Inches(0.4), W, Inches(0.4), COLOR_LIGHT_GRAY)
    add_text_box(slide, "Wheat Vision AI  |  Wheat Disease Detection System",
                 Inches(0.5), H - Inches(0.38), Inches(7), Inches(0.35),
                 font_size=9, color=RGBColor(0x9C, 0xA3, 0xAF), align=PP_ALIGN.LEFT)
    add_text_box(slide, f"Slide {slide_num}",
                 Inches(8.5), H - Inches(0.38), Inches(1), Inches(0.35),
                 font_size=9, color=RGBColor(0x9C, 0xA3, 0xAF), align=PP_ALIGN.RIGHT)

    print(f"  ✅ Built content slide: {data['title']}")
    return slide


def build_two_column_slide(prs, data):
    """Two-column comparison slide."""
    slide_layout = prs.slide_layouts[6]  # Blank
    slide = prs.slides.add_slide(slide_layout)

    W = prs.slide_width
    H = prs.slide_height

    set_slide_background(slide, COLOR_WHITE)

    # Header bar
    add_rectangle(slide, 0, 0, W, Inches(1.2), COLOR_PRIMARY)
    slide_num = data.get("slide_num", "")
    icon = data.get("icon", "")
    add_text_box(slide, str(slide_num),
                 Inches(0.25), Inches(0.3), Inches(0.5), Inches(0.55),
                 font_size=11, bold=True, color=COLOR_WHITE, align=PP_ALIGN.CENTER)
    add_text_box(slide, icon,
                 Inches(0.7), Inches(0.1), Inches(0.8), Inches(0.9),
                 font_size=26, align=PP_ALIGN.CENTER, color=COLOR_WHITE)
    add_text_box(slide, data["title"],
                 Inches(1.4), Inches(0.18), Inches(7.8), Inches(0.85),
                 font_size=24, bold=True, color=COLOR_WHITE, align=PP_ALIGN.LEFT)

    # Column 1 card
    add_rectangle(slide, Inches(0.3), Inches(1.25), Inches(4.4), Inches(4.7),
                  COLOR_LIGHT_GRAY, COLOR_PRIMARY)
    add_text_box(slide, data.get("col1_title", "Column 1"),
                 Inches(0.4), Inches(1.35), Inches(4.2), Inches(0.55),
                 font_size=15, bold=True, color=COLOR_PRIMARY, align=PP_ALIGN.LEFT)

    for i, point in enumerate(data.get("col1_points", [])[:6]):
        y = Inches(1.95) + i * Inches(0.6)
        add_rectangle(slide, Inches(0.45), y + Inches(0.18), Inches(0.09), Inches(0.09), COLOR_SECONDARY)
        add_text_box(slide, point, Inches(0.65), y, Inches(3.9), Inches(0.55),
                     font_size=13, color=COLOR_DARK)

    # Column 2 card
    add_rectangle(slide, Inches(5.0), Inches(1.25), Inches(4.4), Inches(4.7),
                  COLOR_LIGHT_GRAY, COLOR_SECONDARY)
    add_text_box(slide, data.get("col2_title", "Column 2"),
                 Inches(5.1), Inches(1.35), Inches(4.2), Inches(0.55),
                 font_size=15, bold=True, color=COLOR_SECONDARY, align=PP_ALIGN.LEFT)

    for i, point in enumerate(data.get("col2_points", [])[:6]):
        y = Inches(1.95) + i * Inches(0.6)
        add_rectangle(slide, Inches(5.15), y + Inches(0.18), Inches(0.09), Inches(0.09), COLOR_PRIMARY)
        add_text_box(slide, point, Inches(5.35), y, Inches(3.9), Inches(0.55),
                     font_size=13, color=COLOR_DARK)

    # Bottom strip
    add_rectangle(slide, 0, H - Inches(0.4), W, Inches(0.4), COLOR_LIGHT_GRAY)
    add_text_box(slide, "Wheat Vision AI  |  Wheat Disease Detection System",
                 Inches(0.5), H - Inches(0.38), Inches(7), Inches(0.35),
                 font_size=9, color=RGBColor(0x9C, 0xA3, 0xAF), align=PP_ALIGN.LEFT)
    add_text_box(slide, f"Slide {slide_num}",
                 Inches(8.5), H - Inches(0.38), Inches(1), Inches(0.35),
                 font_size=9, color=RGBColor(0x9C, 0xA3, 0xAF), align=PP_ALIGN.RIGHT)

    print(f"  ✅ Built two-column slide: {data['title']}")
    return slide


# ─────────────────────────────────────────────────────
#  MAIN PPT GENERATOR
# ─────────────────────────────────────────────────────
def generate_ppt(output_filename="Wheat_Disease_Detection_Presentation.pptx"):
    """
    Main function to generate the complete PowerPoint presentation.
    """
    print("=" * 60)
    print("  🌾 Wheat Vision AI — PPT Generator")
    print("=" * 60)

    # Step 1: Get slide data (optionally enrich with ChatGPT)
    slide_data = SLIDE_DATA_DEFAULT.copy()

    if USE_CHATGPT:
        print(f"\n🔑 OpenAI API Key detected. Enhancing slides with ChatGPT...")
        slide_data = enrich_slides_with_chatgpt(slide_data)
    else:
        print("\n⚠️  No OpenAI API key found.")
        print("    Using default slide content (still creates a great PPT!)")
        print("    To use ChatGPT: set OPENAI_API_KEY environment variable\n")

    # Step 2: Create Presentation object (widescreen 16:9)
    prs = Presentation()
    prs.slide_width  = Inches(10)
    prs.slide_height = Inches(7.5)

    # Step 3: Build each slide
    print(f"\n📊 Building {len(slide_data)} slides...\n")

    for data in slide_data:
        layout = data.get("layout", "content")

        if layout == "title":
            build_title_slide(prs, data)
        elif layout == "two_column":
            build_two_column_slide(prs, data)
        else:
            build_content_slide(prs, data)

    # Step 4: Save
    prs.save(output_filename)
    print(f"\n{'=' * 60}")
    print(f"  ✅ SUCCESS! PPT saved as: {output_filename}")
    print(f"  📂 Location: {os.path.abspath(output_filename)}")
    print(f"  📊 Total Slides: {len(slide_data)}")
    print(f"  🤖 ChatGPT Enhanced: {'Yes' if USE_CHATGPT else 'No'}")
    print(f"{'=' * 60}")

    # Auto-open on Windows
    try:
        os.startfile(output_filename)
        print("  🚀 Opening presentation in PowerPoint...")
    except Exception:
        print("  💡 Please open the file manually in PowerPoint.")

    return output_filename


# ─────────────────────────────────────────────────────
#  ADVANCED: Generate ChatGPT prompt for manual use
# ─────────────────────────────────────────────────────
def generate_chatgpt_prompt_for_manual_use():
    """
    If you don't want to use the API, this prints a ready-to-use
    prompt you can paste directly into ChatGPT.com
    """
    prompt = """
I am creating a PowerPoint presentation for my final year project:
"Wheat Disease Detection System using Deep Learning (ResNet50)" — also called "Wheat Vision AI".

Project Summary:
- AI web app that diagnoses wheat leaf diseases from uploaded photos
- Uses ResNet50 (50-layer CNN) with Transfer Learning from ImageNet
- 5 Disease Classes: Healthy Wheat, Leaf Rust, Wheat Loose Smut, Crown & Root Rot, Unknown
- Backend: Python Flask  |  Frontend: HTML5/CSS3/JS  |  Database: SQLite
- Features: Upload image → AI diagnosis → Severity estimation → Treatment advice → Scan History
- Security: Bcrypt hashing, role-based access (Admin/User)
- Training: Focal Loss + Class Weights + Data Augmentation (rotation, zoom, brightness, flips)
- Testing: All test cases passed (registration, login, predictions, end-to-end flow)

Please generate professional, concise slide content for a 13-slide PowerPoint presentation
with these slide titles:
1. Title Slide
2. Problem Statement
3. Project Objectives
4. Technology Stack
5. System Architecture
6. ResNet50 AI Model
7. Disease Classes & Performance
8. Key Application Features
9. Severity Estimation Logic
10. Challenges & Solutions
11. Testing Results
12. Future Scope
13. Thank You

For each slide, provide 6-8 bullet points. Keep each point under 15 words.
Format the output as structured text I can copy into slides.
"""
    print("\n" + "=" * 60)
    print("  📋 CHATGPT PROMPT — Paste this on ChatGPT.com")
    print("=" * 60)
    print(prompt)
    print("=" * 60)
    return prompt


# ─────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--prompt":
        # Just print the ChatGPT prompt for manual use
        generate_chatgpt_prompt_for_manual_use()
    else:
        # Default: Generate the full PPT
        generate_ppt("Wheat_Disease_Detection_Presentation.pptx")

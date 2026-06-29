from django.shortcuts import render
import pandas as pd
import os
from PIL import Image
import pytesseract
import ollama
import tempfile
import os
import ollama

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def read_prescription_with_huggingface(image_file):
    from transformers import DonutProcessor, VisionEncoderDecoderModel
    from PIL import Image
    import torch

    processor = DonutProcessor.from_pretrained("chinmays18/medical-prescription-ocr")
    model = VisionEncoderDecoderModel.from_pretrained("chinmays18/medical-prescription-ocr")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    image = Image.open(image_file).convert("RGB")

    pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)

    task_prompt = "<s_ocr>"
    decoder_input_ids = processor.tokenizer(
        task_prompt,
        return_tensors="pt"
    ).input_ids.to(device)

    generated_ids = model.generate(
        pixel_values,
        decoder_input_ids=decoder_input_ids,
        max_length=512,
        num_beams=1,
        early_stopping=True
    )

    generated_text = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True
    )[0]

    return generated_text



def load_medicines():
    csv_path = os.path.join(os.getcwd(), "medicines.csv")
    return pd.read_csv(csv_path)


def search_medicines(query):
    medicines = load_medicines()
    query = query.lower()

    results = []

    for _, row in medicines.iterrows():
        score = 0
        text = f"{row['name']} {row['usage']} {row['color']} {row['form']}".lower()

        for word in query.split():
            if word in text:
                score += 1

        if score > 0:
            results.append({
                "name": row["name"],
                "usage": row["usage"],
                "color": row["color"],
                "form": row["form"],
                "image": row["image"],
                "score": score
            })

    results = sorted(results, key=lambda x: x["score"], reverse=True)
    return results[:6]


def ask_local_ai(question):
    try:
        medicines = load_medicines()

        medicine_context = ""
        for _, row in medicines.iterrows():
            medicine_context += (
                f"Medicine name: {row['name']}\n"
                f"Used for: {row['usage']}\n"
                f"Color: {row['color']}\n"
                f"Form: {row['form']}\n\n"
            )

        response = ollama.chat(
            model="llama3.2",
            messages=[
                {
                    "role": "system",
                    "content": f"""
You are a Pharmacist AI Assistant for a student pharmacy web app.

You can answer about ANY medicine, not only the local database.

Use these rules:
- If the medicine exists in the local database, use that information.
- If the medicine is not in the local database, answer using your general medical knowledge.
- Give general educational information only.
- Do not diagnose disease.
- Do not prescribe treatment.
- Do not give exact dosage instructions.
- Keep the answer short and clear.
- Explain:
  1. What the medicine is used for
  2. Active ingredient if known
  3. Common warnings or side effects
  4. Say final decision belongs to pharmacist or doctor

Local medicine database:
{medicine_context}
"""
                },
                {
                    "role": "user",
                    "content": question
                }
            ]
        )

        return response["message"]["content"]

    except Exception as e:
        return f"Ollama AI error: {str(e)}"


def read_prescription_with_qwen(image_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp:
        for chunk in image_file.chunks():
            temp.write(chunk)
        temp_path = temp.name

    try:
        response = ollama.chat(
            model="qwen2.5vl:3b",
            messages=[
                {
                    "role": "user",
                    "content": """
You are an experienced pharmacist AI assistant.

Analyze this handwritten prescription image.

Return the answer in clean plain text.

Use this exact format:

DETECTED PRESCRIPTION

1. Medicine:
   Strength:
   Dosage:
   Duration:

2. Medicine:
   Strength:
   Dosage:
   Duration:

PATIENT EXPLANATION

1. Medicine:
   Purpose:
   How to take:
   Duration:

2. Medicine:
   Purpose:
   How to take:
   Duration:

SAFETY NOTICE

This interpretation is AI-generated and may contain mistakes.
Always verify the prescription with your doctor or pharmacist before taking any medicine.

Rules:
- Do not use markdown symbols like ** or ###.
- Do not use long separator lines.
- Do not invent medicine names.
- If unclear, write [unclear].
- Explain dosage meanings:
  1-0-1 = morning and night
  1-0-0 = morning only
  0-1-0 = afternoon only
  0-0-1 = night only
""",
                    "images": [temp_path],
                }
            ],
        )

        return response["message"]["content"]

    finally:
        os.remove(temp_path)



def home(request):

    drug_results = []
    prescription_results = []
    extracted_text = ""
    ai_answer = ""

    # Which page should stay open after refresh
    active_tab = "dashboard"

    if request.method == "POST":

        # --------------------------
        # Drug Finder
        # --------------------------
        if "find_drug" in request.POST:

            active_tab = "drug"

            description = request.POST.get("description", "")
            drug_results = search_medicines(description)

        # --------------------------
        # Prescription Reader
        # --------------------------
        elif "read_prescription" in request.POST:

            active_tab = "prescription"

            image_file = request.FILES.get("prescription")

            if image_file:
                try:
                    extracted_text = read_prescription_with_qwen(image_file)

                    if not extracted_text.strip():
                        extracted_text = "No readable text detected. Please upload a clearer image."

                except Exception as e:
                    extracted_text = f"Qwen vision error: {str(e)}"
        # --------------------------
        # Pharmacist AI
        # --------------------------
        elif "ask_ai" in request.POST:

            active_tab = "chat"

            question = request.POST.get("ai_question", "")

            ai_answer = ask_local_ai(question)

    return render(request, "finder/home.html", {
        "drug_results": drug_results,
        "prescription_results": prescription_results,
        "extracted_text": extracted_text,
        "ai_answer": ai_answer,
        "active_tab": active_tab,
    })

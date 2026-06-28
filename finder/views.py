from django.shortcuts import render
import pandas as pd
import os
from PIL import Image
import pytesseract
from dotenv import load_dotenv
import google.generativeai as genai


load_dotenv()


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


def ask_gemini(question):
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        return "Gemini API key is missing. Add GEMINI_API_KEY to your .env file."

    try:
        genai.configure(api_key=api_key)

        model = genai.GenerativeModel("gemini-2.5-flash")

        prompt = f"""
You are a Pharmacist AI Assistant for a student project.

Rules:
- Give general educational medicine information.
- Do not diagnose disease.
- Do not prescribe treatment.
- Do not give exact dosage instructions.
- Always remind that final decision belongs to pharmacist or doctor.
- Keep the answer simple and short.

User question:
{question}
"""

        response = model.generate_content(prompt)

        return response.text

    except Exception as e:
        return f"AI error: {str(e)}"


def home(request):
    drug_results = []
    prescription_results = []
    extracted_text = ""
    ai_answer = ""

    if request.method == "POST":

        if "find_drug" in request.POST:
            description = request.POST.get("description", "")
            drug_results = search_medicines(description)

        if "read_prescription" in request.POST:
            image_file = request.FILES.get("prescription")

            if image_file:
                try:
                    image = Image.open(image_file)
                    extracted_text = pytesseract.image_to_string(image)
                    prescription_results = search_medicines(extracted_text)
                except Exception:
                    extracted_text = "OCR is not ready yet. We will add Tesseract OCR later."

        if "ask_ai" in request.POST:
            question = request.POST.get("ai_question", "")
            ai_answer = ask_gemini(question)

    return render(request, "finder/home.html", {
        "drug_results": drug_results,
        "prescription_results": prescription_results,
        "extracted_text": extracted_text,
        "ai_answer": ai_answer
    })
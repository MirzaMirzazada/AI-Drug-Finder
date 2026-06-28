from django.shortcuts import render
import pandas as pd
import os
from PIL import Image
import pytesseract
import ollama


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
You are a Pharmacist AI Assistant for a student project.

IMPORTANT:
- The user asks about medicines and pharmacy.
- If the user says Parol, Panadol, Majezik, Nurofen, Dolorex, Augmentin, Vitamin D, Aspirin, Calpol, or Rennie, treat it as a medicine.
- Do not answer with law, contracts, politics, or unrelated meanings.
- Use the medicine database below when possible.
- Give short and simple answers.
- Do not diagnose disease.
- Do not prescribe treatment.
- Do not give exact dosage.
- Always say the pharmacist or doctor must make the final decision.

Medicine database:
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
                    image = Image.open(image_file)
                    extracted_text = pytesseract.image_to_string(image)

                    prescription_results = search_medicines(extracted_text)

                except Exception:
                    extracted_text = "OCR is not ready yet."

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

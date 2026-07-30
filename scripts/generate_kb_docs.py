import os
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

# Base Directory of the Project (NEONEXUS)
BASE_DIR = Path(__file__).resolve().parent.parent
KB_DIR = BASE_DIR / "data" / "knowledge_base"

# Ensure output directory exists
KB_DIR.mkdir(parents=True, exist_ok=True)

# Centralized structured content for compiling RAG knowledge manuals
DOCUMENTS = {
    "food_safety_manual.pdf": {
        "title": "Jivahar Food Safety & Hygiene Manual",
        "sections": [
            ("1. Introduction to Food Redistribution Safety",
             "This manual outlines the core safety guidelines for donors, volunteers, and NGOs on the Jivahar platform. "
             "Ensuring food safety is the highest priority during donation, transportation, and redistribution to prevent foodborne illness."),
            ("2. Temperature Danger Zone Constraints",
             "Bacteria grow rapidly in the temperature danger zone between 4°C (40°F) and 60°C (140°F). "
             "Hot cooked food must be kept above 60°C. Cold perishable food must be kept below 4°C. Perishable food left in the danger zone "
             "for more than 2 hours must be discarded. If ambient temperatures exceed 32°C, the safety window is reduced to 1 hour."),
            ("3. Perishable Food Storage and Shelf Life Guidelines",
             "Cooked meals (rice, pasta, meat, vegetables) have a maximum shelf life of 4 hours at room temperature, "
             "or 3 to 4 days if stored in airtight containers under refrigeration (< 4°C). "
             "Raw meat must always be stored in the freezer below -18°C. Dairy items must be kept refrigerated at all times."),
            ("4. Visual and Sensory Inspection Protocols",
             "Volunteers must execute a sensory check at pickup. Reject items if they exhibit: "
             "1. Foul, sour, or rancid smell. "
             "2. Visual indicators of decay (mold growth, discoloration, slime). "
             "3. Damage to packaging (dents, leaks, broken seals, bulging cans)."),
        ]
    },
    "government_regulations.pdf": {
        "title": "National Food Redistribution Regulatory Policy",
        "sections": [
            ("1. Regulatory Framework Overview",
             "This document details local health codes and national legislation regulating surplus food donation and distribution. "
             "Surplus food redistribution acts as a vital tool to reduce waste while strictly conforming to food security standards."),
            ("2. Good Samaritan Protection Act",
             "Under the National Food Donor Protection Act (Good Samaritan Law), donors (hotels, restaurants, individuals) "
             "and non-profit distributors are protected from civil and criminal liability regarding the condition of donated food, "
             "provided the food was donated in good faith, without gross negligence or intentional misconduct."),
            ("3. Health and Hygiene Licensing Requirements",
             "Any facility preparing or distributing cooked food must hold a valid Food Safety License. "
             "Volunteers handling unpackaged food must wear clean gloves, hairnets, and face masks, "
             "and must have undergone basic hygiene training to ensure zero cross-contamination."),
            ("4. Food Waste Disposal Policies",
             "Surplus food that fails visual inspection must be sent to composting or waste-to-energy facilities. "
             "It is illegal to discard fit-for-compost food into general garbage landfill sites under municipal waste guidelines."),
        ]
    },
    "ngo_policies.pdf": {
        "title": "NGO Food Distribution Guidelines & Proximity Scope",
        "sections": [
            ("1. Scope and Objective",
             "This policy outlines operating rules for NGOs participating in the Jivahar food distribution system. "
             "NGOs must align their storage capability and matching thresholds to donation categories."),
            ("2. Matching and Distribution Priority",
             "Hot cooked meals must be matched to NGOs within a 5 km radius of the donor to ensure transport time "
             "does not exceed 45 minutes, minimizing the duration in the Temperature Danger Zone. "
             "Dry rations (grains, lentils, canned goods) can be matched to NGOs within a 20 km radius."),
            ("3. Capacity and Storage Matching",
             "Perishable donations require the receiving NGO to possess verified refrigeration units. "
             "Donations exceeding 50 kg can only be routed to tier-1 NGOs with active cold storage chambers and "
             "at least 5 volunteers available for immediate sorting and distribution."),
            ("4. Rating and Track Record Compliance",
             "The platform assigns priority matching to NGOs with high ratings (above 4.0). "
             "Ratings are calculated based on response speed, successful delivery logs, and food safety reviews "
             "submitted by volunteers and donors."),
        ]
    },
    "platform_faqs.pdf": {
        "title": "Jivahar Redistribution Portal: FAQs",
        "sections": [
            ("Q1: How do donors list a food donation?",
             "Donors log into the portal, click 'Create Donation', and input the food name, quantity, preparation timestamp, "
             "and storage conditions. They can upload an image for CNN-based food category classification."),
            ("Q2: How are volunteers notified of a pickup?",
             "Nearby volunteers receive an automated push notification detailing the food category, priority, and pickup location. "
             "Volunteers can click 'Accept Task' to navigate to the donor."),
            ("Q3: What should a volunteer do if the food fails inspection?",
             "If the food smells sour, is moldy, or has a broken seal, the volunteer must tap 'Reject Pickup' in the app, "
             "select the rejection reason, upload a photo, and inform the donor. The task is then canceled."),
            ("Q4: How do NGOs claim food?",
             "NGOs set their preferences (e.g., veg/non-veg, capacity limits). When a donation matches their profile, "
             "the system suggests the match, and the NGO has 15 minutes to accept the recommendation before it is routed to another NGO."),
        ]
    }
}

def generate_pdfs():
    # Setup document styles
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontSize=18,
        leading=22,
        spaceAfter=15
    )
    section_title_style = ParagraphStyle(
        'SectionTitle',
        parent=styles['Heading2'],
        fontSize=12,
        leading=16,
        spaceBefore=12,
        spaceAfter=6
    )
    body_style = ParagraphStyle(
        'DocBody',
        parent=styles['Normal'],
        fontSize=10,
        leading=14,
        spaceAfter=10
    )

    for filename, info in DOCUMENTS.items():
        filepath = KB_DIR / filename
        doc = SimpleDocTemplate(str(filepath), pagesize=letter)
        story = []
        
        # Add Title
        story.append(Paragraph(info["title"], title_style))
        story.append(Spacer(1, 10))
        
        # Add Sections
        for heading, body in info["sections"]:
            story.append(Paragraph(heading, section_title_style))
            story.append(Paragraph(body, body_style))
            story.append(Spacer(1, 5))
            
        doc.build(story)
        print(f"Generated PDF: {filepath.name}")

if __name__ == "__main__":
    generate_pdfs()

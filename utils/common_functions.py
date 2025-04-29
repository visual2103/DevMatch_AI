import spacy
import glob
import os
from docx import Document
import warnings
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore', category=UserWarning)

def extract_text_from_docx(filepath, is_cv):
    try:
        doc = Document(filepath)
        paragraphs = doc.paragraphs[1 if is_cv else 0:]  # Skip first paragraph for CVs
        text = ' '.join(para.text for para in paragraphs if para.text.strip())
        
        if not text.strip():
            logger.warning(f"Empty text extracted from {filepath}")
            return None
            
        return text
    except Exception as e:
        logger.error(f"Error extracting text from {filepath}: {str(e)}")
        return None

def load_docx_from_folder(folder_path, is_cv=True):
    if not os.path.exists(folder_path):
        logger.error(f"Folder path does not exist: {folder_path}")
        return [], [], []
        
    filepaths = glob.glob(os.path.join(folder_path, '*.docx'))
    if not filepaths:
        logger.error(f"No DOCX files found in {folder_path}")
        return [], [], []
        
    logger.info(f"Found {len(filepaths)} DOCX files in {folder_path}")
    
    documents = []  
    filenames = []  
    descriptions = []  
    
    for filepath in filepaths:
        try:
            if os.path.basename(filepath).startswith('~$'):
                continue
                
            text = extract_text_from_docx(filepath=filepath, is_cv=is_cv)
            if text is None:
                continue
                
            text = text.replace('\n', ' ').replace('  ', ' ')
            
            if not is_cv:
                text = text.split("Benefits:")[0]
                description = text.split("Key Responsibilities:")[0]
            else:
                if "Project Experience" in text:
                    description = text.split("Project Experience")[1]
                else:
                    description = ""  # fallback if section is missing
                    logger.warning(f"No Project Experience section found in {filepath}")

            if text.strip() and description.strip():  # Only add if we have valid text
                documents.append(text)
                filenames.append(os.path.basename(filepath))
                descriptions.append(description)
            
        except Exception as e:
            logger.error(f"Error processing file {filepath}: {str(e)}")
            continue
    
    if not documents:
        logger.error("No valid documents were processed")
        return [], [], []
        
    logger.info(f"Successfully processed {len(documents)} documents")
    return documents, filenames, descriptions


nlp = spacy.load("en_core_web_lg")
def spacy_tokenizer(text):
    doc = nlp(text)  #
    return [token.lemma_.lower() for token in doc if
            not token.is_stop and not token.is_punct and not token.text.isspace() and not token.text.isnumeric()]


def progress_bar_update(percent_complete,
                        progress_bar,
                        status_text):
    progress_bar.progress(percent_complete)
    progress_text = "Operation in progress. Please wait. ⏳" if percent_complete < 100 else "Operation Completed."
    status_text.text(f"{progress_text} - {percent_complete}% - ")

#---------------------------------------------------
domain_data = {
    "Banking": {
        "desc": "Developed software for banking platforms, digital wallets, loan processing, or credit systems.",
        "keywords": ["bank", "loan", "credit", "atm", "fintech", "interest", "account", "ledger"]},
    "Healthcare": {
        "desc": "Built medical systems such as EHR, hospital platforms, clinical apps, or telemedicine services.",
        "keywords": ["healthcare", "ehr", "patient", "hospital", "clinic", "medical", "doctor", "nurse"]},
    "E-commerce": {
        "desc": "Created platforms or tools for online stores, shopping carts, payments, or product discovery.",
        "keywords": ["ecommerce", "checkout", "cart", "payment", "shopify", "woocommerce", "product", "sku"]},
    "Telecommunications": {
        "desc": "Engineered tools for telecom networks, call management, VoIP, or network monitoring.",
        "keywords": ["telecom", "sms", "voip", "5g", "network", "bandwidth", "subscriber", "lte"]},
    "Education": {"desc": "Built education platforms such as learning portals, student dashboards, or LMS systems.",
                  "keywords": ["education", "student", "teacher", "learning", "course", "classroom", "lms", "school"]},
    "Retail": {"desc": "Developed software for retail businesses such as POS systems, inventory, or loyalty programs.",
               "keywords": ["retail", "store", "inventory", "pos", "stock", "sku", "receipt", "shopping"]},
    "Insurance": {"desc": "Created applications for policy management, claims processing, or underwriting systems.",
                  "keywords": ["insurance", "claim", "policy", "underwriting", "premium", "broker", "risk"]},
    "Legal": {"desc": "Built tools for legal case management, document processing, or e-discovery platforms.",
              "keywords": ["legal", "law", "contract", "case", "compliance", "jurisdiction", "litigation"]},
    "Manufacturing": {
        "desc": "Developed automation systems, supply chain tools, or MES solutions for production plants.",
        "keywords": ["manufacturing", "plant", "automation", "mes", "machine", "factory", "assembly"]},
    "Transportation & Logistics": {
        "desc": "Built platforms for delivery tracking, fleet management, logistics optimization, or routing.",
        "keywords": ["logistics", "delivery", "routing", "fleet", "dispatch", "transport", "warehouse"]},
    "Energy & Utilities": {
        "desc": "Developed monitoring systems, SCADA platforms, or analytics for power and water utilities.",
        "keywords": ["energy", "power", "electricity", "gas", "grid", "meter", "solar", "utility", "scada"]},
    "Real Estate": {
        "desc": "Engineered platforms for property listings, CRM tools for agents, or real estate analytics.",
        "keywords": ["real estate", "property", "mortgage", "agent", "listing", "tenant", "lease"]},
    "Government": {"desc": "Built public service portals, civic data dashboards, or digital identity platforms.",
                   "keywords": ["government", "municipal", "civic", "permit", "id", "citizen", "registry"]},
    "Marketing": {
        "desc": "Built marketing analytics tools, email campaign platforms, or digital ad performance systems.",
        "keywords": ["marketing", "campaign", "seo", "email", "ads", "promotion", "branding", "targeting"]},
    "Media & Entertainment": {
        "desc": "Created digital content platforms, streaming services, or entertainment production tools.",
        "keywords": ["media", "streaming", "video", "music", "entertainment", "broadcast", "subscriber"]},
    "Construction": {
        "desc": "Engineered project management tools, BIM integrations, or field apps for construction teams.",
        "keywords": ["construction", "site", "blueprint", "project", "bim", "architect", "contractor"]},
    "Finance (non-banking)": {
        "desc": "Worked on accounting systems, budgeting tools, payroll, or financial planning platforms.",
        "keywords": ["finance", "accounting", "budget", "payroll", "invoice", "expense", "audit", "report"]}
}
import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate

# --- 1. CONFIGURATION ---
DB_PATH = "./chroma_db"
OLLAMA_MODEL = "llama3.2" 

# --- 2. SETUP DATABASE ---
embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

if os.path.exists(DB_PATH):
    vector_db = Chroma(persist_directory=DB_PATH, embedding_function=embedding_function)
else:
    vector_db = None

# --- 3. SETUP MODEL ---
llm = Ollama(model=OLLAMA_MODEL, temperature=0)

def get_agent_response(category_key, user_story):
    """
    Main function to process visa applications.
    """
    if not vector_db:
        return "⚠️ Error: Database not found. Please run create_db.py first."

    print(f"\n🔎 Processing: {user_story.split('.')[0]}...")
    
    # RETRIEVAL
    retriever = vector_db.as_retriever(search_kwargs={"k": 5})
    docs = retriever.invoke(user_story)
    context_text = "\n\n".join([doc.page_content for doc in docs])

    if not context_text:
        return "⚠️ Error: No relevant rules found in the database."

    # PROMPT - "SILENT PROFESSIONAL"
    prompt_template = """
    You are the Official AI Visa Assessor for the UK Home Office.
    Your task is to evaluate the applicant and output a formal decision letter.

    --- OFFICIAL RULES ---
    {context}

    --- APPLICANT DATA ---
    {question}

    --- INTERNAL AUDIT RULES (DO NOT PRINT THESE STEPS) ---
    1. **Identity:** If CAS/CoS is Missing/"No" -> Fail. (Note: "TEST-123" is Valid).
    2. **English:** If "No" -> Fail.
    3. **Funds:** If Funds 28 Days is "No" -> Fail.
    4. **Declarations:** If Criminal Record or Refusals is "Yes" -> Fail.
    5. **Pass Condition:** Applicant ONLY passes if ALL checks are OK.

    --- RESPONSE INSTRUCTIONS (STRICT) ---
    1. **NO THINKING:** Do NOT print your checklist, steps, or internal logic.
    2. **NO PREAMBLE:** Start immediately with "**Verdict**".
    3. **TONE:** Professional, neutral, and direct.

    --- OUTPUT FORMAT ---
    **Verdict**: [ELIGIBLE / NOT ELIGIBLE]
    
    **Summary**:
    [One single, professional sentence summarizing the decision.]
    
    **Key Reasons**:
    * [Reason 1 (e.g., "The mandatory financial requirement was not met.")]
    * [Reason 2 (e.g., "Adverse immigration history declared.")]
    * [Reason 3 (e.g., "English language proficiency not evidenced.")]
    """
    
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = prompt | llm
    
    try:
        response = chain.invoke({"context": context_text, "question": user_story})
        return response
    except Exception as e:
        return f"AI Generation Error: {str(e)}"

if __name__ == "__main__":
    print("✅ Visa Agent Loaded Successfully.")
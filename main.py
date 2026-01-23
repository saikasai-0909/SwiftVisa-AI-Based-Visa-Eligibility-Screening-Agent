import sys
from langchain_chroma import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- CONFIGURATION ---
DB_PATH = "./visa_db"
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3"

# Map user choices to the exact category tags we created in Step 2
VISA_OPTIONS = {
    "1": ("student_visa", "Student Visa"),
    "2": ("skilled_worker_visa", "Skilled Worker Visa"),
    "3": ("visitor_visa", "General Visitor Visa"),
    "4": ("healthcare_visa", "Health and Care Visa"),
    "5": ("short_term_student_visa", "Short-term Student (English)")
}

def get_user_category():
    print("\n--- SELECT VISA CATEGORY ---")
    for key, (tag, name) in VISA_OPTIONS.items():
        print(f"{key}. {name}")
    
    while True:
        choice = input("\nEnter number (1-5): ").strip()
        if choice in VISA_OPTIONS:
            return VISA_OPTIONS[choice][0]  # Return the internal tag (e.g., 'student_visa')
        print("Invalid selection. Try again.")

def run_screening_agent():
    # 1. Load the Database
    print("\nLoading Visa Rules Database...")
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    vector_db = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)

    # 2. Get User's Target Visa
    selected_category = get_user_category()
    print(f"\n[LOCKED IN] Screening against **{selected_category.replace('_', ' ').upper()}** rules only.")

    # 3. Create the Retriever (The "Search Engine")
    # This filter is the MAGIC. It prevents the AI from reading the wrong files.
    retriever = vector_db.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": 6,  # Read top 6 most relevant rules
            "filter": {"category": selected_category} 
        }
    )

    # 4. Define the AI Brain
    llm = ChatOllama(model=LLM_MODEL)

    # 5. The Prompt (Strict instructions for the AI)
    template = """
    You are a strict UK Visa Eligibility Officer.
    You must determine if the candidate is eligible based ONLY on the provided context rules.
    
    CONTEXT (Official Rules):
    {context}
    
    CANDIDATE SITUATION:
    {question}
    
    INSTRUCTIONS:
    1. Start with "ELIGIBLE", "NOT ELIGIBLE", or "UNCERTAIN".
    2. List the specific criteria they met or failed.
    3. Cite the exact rule or paragraph from the context if possible.
    4. If the context does not contain the answer, say "The provided rules do not cover this specific situation."
    
    OFFICER'S DECISION:
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    
    # 6. Build the Chain
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # 7. Start Chatting
    print("\n" + "="*50)
    print(f"VISA SCREENING AGENT READY ({LLM_MODEL})")
    print("Describe your situation (e.g., 'I am 25, earn 30k, want to bring my wife').")
    print("Type 'exit' to quit or 'switch' to change visa type.")
    print("="*50)

    while True:
        user_input = input("\nCandidate Profile: ")
        
        if user_input.lower() in ["exit", "quit"]:
            print("Closing session.")
            break
        
        if user_input.lower() == "switch":
            # Restart the function to pick a new category
            run_screening_agent()
            break

        print("\nAnalying rules... (This depends on your PC speed)\n")
        try:
            response = chain.invoke(user_input)
            print("-" * 20)
            print(response)
            print("-" * 20)
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    run_screening_agent()
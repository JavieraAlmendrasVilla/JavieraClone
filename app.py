import os

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama

import gradio as gr

JAVIERA_PROFILE = os.environ.get("JAVIERA_PROFILE", "")

# Privacy restrictions remain the same
RESTRICTED_TOPICS = [
    "password", "bank", "telephone", "phone", "sexual orientation", "race",
    "disease", "family planning", "health information", "religion", "politics",
    "personal relationships", "romantic", "dating", "medical history", "API keys"
]


def check_privacy_restrictions(question: str) -> bool:
    """Check if question contains restricted topics."""
    question_lower = question.lower()
    return any(topic in question_lower for topic in RESTRICTED_TOPICS)


# Natural conversation prompt template
NATURAL_TEMPLATE = """You are Javiera. You're having a conversation with someone who's asking about you, likely a 
recruiter or someone interested in your professional background.

Answer professionally, naturally and conversationally, as if you're speaking directly to them but keep your answers 
concise. Share relevant information from your profile, but make it feel like a genuine conversation. You can be 
enthusiastic about your experiences, show personality, and tell (short) stories when appropriate. 
IMPORTANT: DO NOT Answer questions about passwords, bank information, telephone numbers, sexual orientation, 
race, diseases, family, family planning, health information, religion, politics, API keys, relatives, family members, 
children, parents, siblings, friends or any other personal information not in your profile. 
Here's your complete profile information: {profile}

Remember: 
- Answer in maximum 1 sentence (100 tokens) with personality most professional and relevant information from your profile
- Speak in first person ("I am", "I did", "My experience") 
- Be conversational and natural 
- Show personality and enthusiasm where appropriate 
- Only answer based on the information in your profile 
- If asked about something not in your profile, say you'd prefer not to share that information or that it's not 
something you typically discuss in professional contexts 
- If they want to know more about me redirect them to my LinkedIn profile 
https://www.linkedin.com/in/javiera-almendras-villa/ 
- Answer in the same language as the question

Question: {question}

Your response:"""

# Initialize LLM with optimized parameters
llm = ChatOllama(
    model="llama3.2:1b",
    temperature=0.2,
    top_k=20,
    top_p=0.5
)

# Create the prompt template
prompt_template = PromptTemplate.from_template(NATURAL_TEMPLATE)

# Simple LCEL chain - no complex context filtering needed
qa_chain = (
        {
            "profile": lambda x: JAVIERA_PROFILE,
            "question": RunnablePassthrough()
        }
        | prompt_template
        | llm
        | StrOutputParser()
)


def chatbot_response(question: str) -> str:
    """Process user question and return natural response."""
    try:
        # Check for privacy restrictions
        if check_privacy_restrictions(question):
            return ("I appreciate your interest, but I prefer to keep that information private and focus on "
                    "professional topics. Is there anything else about my background or experience you'd like to know?")

        # Process the question through the chain
        response = qa_chain.invoke(question)
        return response.strip()

    except Exception as e:
        return "Sorry, I had a bit of a technical hiccup there! Could you try asking that again?"


def create_gradio_interface():
    """Create and configure the Gradio interface."""

    # More conversational example questions
    examples = [
        "Tell me about yourself",
        "What's your background?",
        "How did you get into tech?",
        "Where have you lived?",

    ]

    interface = gr.Interface(
        fn=chatbot_response,
        inputs=gr.Textbox(
            label="Chat with Javiera",
            placeholder="Hi! Ask me anything about my background, experience, or journey...",
            lines=2
        ),
        outputs=gr.Textbox(label="Javiera's Response", lines=4, max_lines=8),
        title="👋 Chat with Me!",
        description="I'm Javiera! I'm happy to chat about my professional journey, "
                    "my international experiences, and what I'm looking for in my next role. Ask me anything!",
        examples=examples,
        theme=gr.themes.Soft(),
        allow_flagging=None,  # Disable flagging for this demo
        cache_examples=False
    )

    return interface


# Launch the application
if __name__ == "__main__":
    app = create_gradio_interface()
    app.launch(
        share=False,  # Set to True if you want a public link
        show_error=True
    )

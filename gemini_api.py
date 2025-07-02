import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class GeminiAPI:
    def __init__(self):
        """
        Initialize the Gemini API client
        
        Requires GOOGLE_API_KEY environment variable to be set in .env file
        """
        # Load environment variables again to ensure it's loaded
        load_dotenv()
        
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set in .env file")
        
        genai.configure(api_key=api_key)
        
        # Choose a model (using Gemini-1.5-pro model for better accuracy)
        self.model = genai.GenerativeModel('gemini-1.5-pro')
    
    def generate_response(self, question, context, query_history=None):
        """
        Generate a response using Gemini model with given context
        
        Args:
            question: User's question
            context: Context from relevant PDF chunks
            query_history: Previous user queries for context
            
        Returns:
            Gemini's response
        """
        # Special handling for questions about previous queries
        if query_history and len(query_history) > 0:
            lower_question = question.lower()
            meta_query_phrases = [
                "what did i ask before", 
                "what was my previous question", 
                "what were my previous questions", 
                "what did i ask previously", 
                "what have i asked", 
                "what questions did i ask",
                "what was my last question",
                "what did i just ask",
                "previous query",
                "previous questions"
            ]
            
            # Check if this is a meta-query about conversation history
            is_meta_query = any(phrase in lower_question for phrase in meta_query_phrases)
            
            if is_meta_query:
                if len(query_history) == 1:
                    return f"Your previous question was: \"{query_history[0]}\""
                else:
                    response = "Here are your previous questions:\n\n"
                    # Show last 5 questions in reverse order (most recent first)
                    for i, q in enumerate(reversed(query_history[-5:])):
                        response += f"{i+1}. \"{q}\"\n"
                    return response
        
        # Check if we have enough relevant context
        if not context or len(context) == 0:
            return "I don't have enough information to answer that question accurately. Please try rephrasing or asking a different question about university topics."
        
        # Check if this is a casual greeting or small talk
        greeting_phrases = ["hi", "hello", "hey", "greetings", "good morning", "good afternoon", "good evening", 
                           "how are you", "what's up", "nice to meet you", "how's it going", "howdy"]
                   
        small_talk_patterns = {
            "thanks": ["thank", "thanks", "thank you", "appreciate"],
            "goodbye": ["bye", "goodbye", "see you", "farewell", "good night"],
            "help": ["help", "assist", "support", "guidance"],
            "capabilities": ["what can you do", "how can you help", "your capabilities", "what are you able to"],
            "identity": ["who are you", "what are you", "your name", "chatbot"]
        }
        
        # Handle greetings
        if any(question.lower().strip() == phrase or question.lower().strip().startswith(phrase + " ") 
               for phrase in greeting_phrases):
            return f"Hey! How's it going?"
        
        # Handle other small talk patterns
        for category, phrases in small_talk_patterns.items():
            if any(phrase in question.lower() for phrase in phrases):
                if category == "thanks":
                    return "You're welcome! If you have any more questions about university matters, feel free to ask."
                elif category == "goodbye":
                    return "Goodbye! Feel free to return anytime you have questions about university matters."
                elif category == "help":
                    return "I can help you with information about university procedures, admissions, fee structures, courses, campus facilities, and more. What specifically would you like to know about?"
                elif category == "capabilities":
                    return "I can provide information about university admissions, fee structures, course details, campus facilities, student services, and other university-related questions. How can I assist you today?"
                elif category == "identity":
                    return "I'm a university assistant chatbot designed to provide accurate information about university procedures, courses, fees, and other university-related topics. How can I help you today?"
        
        # Check for sufficient context relevance
        has_high_relevance = any(chunk.get("metadata", {}).get("relevance_score", 0) > 0.75 for chunk in context)
        if not has_high_relevance and len(context) < 3:
            return "I don't have enough relevant information to answer that question accurately. Please ask about topics related to university procedures, fees, courses, or other university-specific information."
        
        prompt = self._create_prompt(question, context, query_history)
        
        try:
            # Set specific safety settings to prevent hallucinations
            safety_settings = [
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                }
            ]
            
            # Generation config to improve factuality and reduce hallucinations
            generation_config = {
                "temperature": 0.2,  # Lower temperature for more focused responses
                "top_p": 0.85,
                "top_k": 40,
                "max_output_tokens": 2048,
            }
            
            response = self.model.generate_content(
                prompt,
                safety_settings=safety_settings,
                generation_config=generation_config
            )
            
            # Add a post-processing step to verify answer integrity
            return self._verify_and_refine_response(response.text, question, context)
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def _verify_and_refine_response(self, response_text, question, context):
        """
        Verify that the response contains information that is supported by the context
        
        Args:
            response_text: The raw response from the model
            question: The original question
            context: The context chunks
        
        Returns:
            Verified and potentially refined response
        """
        # If response indicates no information, double-check with the context
        if "don't have enough information" in response_text.lower() or "can't answer" in response_text.lower():
            # Check if we actually have some reasonably relevant chunks
            if any(chunk.get("metadata", {}).get("relevance_score", 0) > 0.7 for chunk in context):
                verification_prompt = f"""
I received a response stating that there isn't enough information to answer the user's question,
but there may actually be relevant information in the context. 

The user asked: {question}

Please recheck the following context carefully and determine if it contains information
to answer the user's question. If it does, provide that answer. If it doesn't, say "I don't have enough information."

CONTEXT:
{chr(10).join([chunk["text"] for chunk in context[:3]])}

VERIFICATION DECISION (only respond with "CONTAINS_ANSWER" or "NO_ANSWER"):
"""
                try:
                    verification_result = self.model.generate_content(verification_prompt)
                    if "CONTAINS_ANSWER" in verification_result.text:
                        # Try generating a better response
                        retry_prompt = self._create_prompt(question, context, None)
                        retry_response = self.model.generate_content(retry_prompt)
                        return retry_response.text
                except:
                    pass
                
        return response_text
    
    def generate_conversation_summary(self, query_history):
        """
        Generate a summary of the conversation history
        
        Args:
            query_history: List of previous user queries
            
        Returns:
            A summary of the conversation context
        """
        if not query_history or len(query_history) <= 2:
            return None
            
        # Only generate summary for longer conversations
        if len(query_history) > 5:
            summary_prompt = f"""
Generate a very brief, concise summary (max 3 sentences) of the main topics and context from these user queries.
Focus only on the key information needs and context that would be helpful for answering future questions.

USER QUERIES:
{chr(10).join([f"- {q}" for q in query_history])}

SUMMARY:
"""
            try:
                response = self.model.generate_content(summary_prompt)
                return response.text
            except Exception:
                return None
        
        return None
    
    def _create_prompt(self, question, context, query_history=None):
        """
        Create a prompt for the Gemini model
        
        Args:
            question: User's question
            context: Context from relevant PDF chunks
            query_history: Previous user queries for context
            
        Returns:
            Formatted prompt string
        """
        # Instruction to format the response nicely
        system_prompt = """
You are a smart, helpful university assistant chatbot that talks like a real person—not like a robot or generic AI. You're here to help students understand university procedures, fee structures, admissions, and other university-related topics. Your tone should be friendly, clear, and beginner-friendly — like you're a supportive student guide.

Your answers must always follow these rules:

1. Be extremely accurate and comprehensive — provide all relevant information.
2. Stay completely within the provided context. Do NOT guess or invent anything.
3. Format your answers well — use bullet points, numbering, and spacing for easy reading.
4. Avoid technical jargon unless necessary. Be clear and simple, especially for new students.
5. Never cite document names or sources in your answer.
6. If something isn’t found in the context, just say: “I don’t have enough information to answer that question accurately.”
7. If only part of a question is answerable, explain which part you can answer and which part isn’t clear or present.
8. If the user refers to an earlier message in the chat, use that history if available to answer.
9. Never sound like a machine or use phrases like “As an AI” or “According to the document”. Just answer directly and naturally.
10. Stick only to university-related topics.

Your knowledge comes from a combination of official university PDFs and website content, giving you accurate insight into programs, fees, policies, and procedures.

Always think carefully, double-check the context, and answer naturally — like a helpful university senior guiding a confused new student.
"""

        # Organize context by relevance and type for better analysis
        sorted_context = sorted(context, key=lambda x: x.get("metadata", {}).get("relevance_score", 0), reverse=True)
        pdf_chunks = [chunk for chunk in sorted_context if chunk["metadata"].get("type") == "pdf"]
        web_chunks = [chunk for chunk in sorted_context if chunk["metadata"].get("type") == "web"]
        
        # Format the context information with relevance scores
        context_sections = []
        
        if pdf_chunks:
            pdf_context = "\n\n".join([f"[Relevance: {chunk['metadata'].get('relevance_score', 0):.2f}] {chunk['text']}" 
                                      for chunk in pdf_chunks])
            context_sections.append("OFFICIAL DOCUMENT INFORMATION:\n" + pdf_context)
            
        if web_chunks:
            web_context = "\n\n".join([f"[Relevance: {chunk['metadata'].get('relevance_score', 0):.2f}] {chunk['text']}" 
                                      for chunk in web_chunks])
            context_sections.append("WEBSITE INFORMATION:\n" + web_context)
            
        combined_context = "\n\n" + "\n\n".join(context_sections)
        
        # Add conversation history if available
        conversation_history = ""
        conversation_summary = ""
        if query_history and len(query_history) > 0:
            # Generate conversation summary for longer conversations
            if len(query_history) > 5:
                summary = self.generate_conversation_summary(query_history)
                if summary:
                    conversation_summary = f"\n\nCONVERSATION SUMMARY:\n{summary}"
            
            # Limit history to last 10 queries to avoid token limits
            limited_history = query_history[-10:] if len(query_history) > 10 else query_history
            conversation_history = "\n\nRECENT QUERIES:\n"
            conversation_history += "\n".join([f"- {q}" for q in limited_history])
        
        # Put it all together
        prompt = f"{system_prompt}\n\nCONTEXT:{combined_context}{conversation_summary}{conversation_history}\n\nCURRENT QUESTION: {question}\n\nANSWER:"
        
        return prompt 
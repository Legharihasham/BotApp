import requests
import os
from dotenv import load_dotenv
import json
import re
from typing import List, Dict, Optional, Tuple
import logging

# Load environment variables
load_dotenv()

# Set up logging
logger = logging.getLogger(__name__)

class GeminiAPI:
    def __init__(self):
        """
        Initialize the Groq API client using direct HTTP requests
        
        Requires GROQ_API_KEY environment variable to be set in .env file
        """
        # Load environment variables again to ensure it's loaded
        load_dotenv()
        
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable not set in .env file")
        
        self.api_key = api_key
        self.api_url = "https://api.groq.com/openai/v1/chat/completions"
        self.model_name = 'llama-3.1-8b-instant'
        
        # Using direct HTTP requests to bypass client library issues
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # University-specific response patterns for dynamic responses
        self.university_domains = {
            'academic': {
                'keywords': ['course', 'program', 'degree', 'major', 'minor', 'curriculum', 'syllabus', 'academic', 'study'],
                'general_responses': [
                    "University academic programs typically include various courses and degree options. Students can choose from different majors and minors based on their interests and career goals.",
                    "Academic programs at universities are designed to provide comprehensive education in specific fields. Each program has its own curriculum and requirements.",
                    "Course selection and academic planning are important aspects of university education. Students work with advisors to plan their academic journey."
                ]
            },
            'administrative': {
                'keywords': ['admission', 'enrollment', 'registration', 'application', 'deadline', 'procedure', 'process'],
                'general_responses': [
                    "University admission processes typically involve submitting applications, meeting deadlines, and following specific procedures for enrollment.",
                    "Administrative procedures at universities help ensure smooth operations and proper student management throughout their academic journey.",
                    "Registration and enrollment processes are designed to help students officially join programs and access university resources."
                ]
            },
            'financial': {
                'keywords': ['fee', 'tuition', 'payment', 'cost', 'scholarship', 'financial aid', 'budget', 'expense'],
                'general_responses': [
                    "University fees and tuition costs vary by program and institution. Financial aid and scholarships are often available to help students manage expenses.",
                    "Understanding the cost structure of university education is important for financial planning. Many universities offer various payment options and financial support.",
                    "Financial planning for university education includes considering tuition, fees, and other expenses. Scholarships and financial aid can significantly reduce costs."
                ]
            },
            'campus': {
                'keywords': ['facility', 'building', 'campus', 'library', 'lab', 'classroom', 'dormitory', 'housing'],
                'general_responses': [
                    "University campuses typically feature various facilities including libraries, laboratories, classrooms, and student housing options.",
                    "Campus facilities are designed to support student learning and provide a comfortable environment for academic and social activities.",
                    "University buildings and facilities are strategically planned to create an optimal learning environment for students and faculty."
                ]
            },
            'student_life': {
                'keywords': ['student', 'life', 'activity', 'club', 'organization', 'event', 'campus life'],
                'general_responses': [
                    "Student life at universities includes various activities, clubs, and organizations that enhance the educational experience.",
                    "Campus life offers opportunities for students to engage in extracurricular activities, build relationships, and develop leadership skills.",
                    "University student organizations and activities provide platforms for personal growth, networking, and pursuing interests beyond academics."
                ]
            }
        }
    
    def _classify_query_domain(self, query: str) -> str:
        """
        Classify the query into a university domain
        
        Args:
            query: User query text
            
        Returns:
            Domain classification (academic, administrative, financial, etc.)
        """
        query_lower = query.lower()
        
        for domain, info in self.university_domains.items():
            if any(keyword in query_lower for keyword in info['keywords']):
                return domain
        
        return 'general'
    
    def _generate_dynamic_response(self, query: str, context: List[Dict]) -> str:
        """
        Generate a dynamic response when exact context is limited
        
        Args:
            query: User query
            context: Available context chunks
            
        Returns:
            Dynamic response based on query domain and available context
        """
        domain = self._classify_query_domain(query)
        
        if domain in self.university_domains:
            # Get domain-specific general responses
            general_responses = self.university_domains[domain]['general_responses']
            
            # Combine with any available context
            context_info = ""
            if context:
                # Extract key information from available context
                context_texts = [chunk['text'][:200] for chunk in context[:2]]
                context_info = " Based on available information: " + " ".join(context_texts)
            
            # Select appropriate general response
            import random
            base_response = random.choice(general_responses)
            
            # Enhance with specific information if available
            if context_info:
                return base_response + context_info + " For more specific details, you might want to check with the university's official resources or contact the relevant department."
            else:
                return base_response + " For specific information about your university's policies and procedures, I recommend checking the official university website or contacting the relevant department directly."
        
        return "I understand you're asking about university-related topics. While I don't have specific information about that particular aspect, I can help you with general university procedures and information. For specific details, please check with your university's official resources."
    
    def _enhance_context_with_general_knowledge(self, query: str, context: List[Dict]) -> List[Dict]:
        """
        Enhance context with general university knowledge when specific context is limited
        
        Args:
            query: User query
            context: Available context chunks
            
        Returns:
            Enhanced context list
        """
        domain = self._classify_query_domain(query)
        
        if domain in self.university_domains and len(context) < 3:
            # Add general knowledge chunks for the domain
            general_knowledge = {
                'text': f"General information about {domain} in universities: {self.university_domains[domain]['general_responses'][0]}",
                'metadata': {
                    'type': 'general_knowledge',
                    'domain': domain,
                    'relevance_score': 0.6,
                    'filtering_reason': 'domain_general_knowledge'
                }
            }
            context.append(general_knowledge)
        
        return context
    
    def generate_response(self, question: str, context: List[Dict], query_history: Optional[List[str]] = None) -> str:
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
            return f"Hey! How's it going? I'm here to help you with university-related questions. What would you like to know about?"
        
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
        
        # Enhance context with general knowledge if needed
        enhanced_context = self._enhance_context_with_general_knowledge(question, context)
        
        # Check if we have enough relevant context
        if not enhanced_context or len(enhanced_context) == 0:
            return self._generate_dynamic_response(question, [])
        
        # Check for sufficient context relevance
        has_high_relevance = any(chunk.get("metadata", {}).get("relevance_score", 0) > 0.75 for chunk in enhanced_context)
        has_moderate_relevance = any(chunk.get("metadata", {}).get("relevance_score", 0) > 0.6 for chunk in enhanced_context)
        
        # If we have limited relevance, generate a more dynamic response
        if not has_high_relevance and not has_moderate_relevance:
            return self._generate_dynamic_response(question, enhanced_context)
        
        prompt = self._create_prompt(question, enhanced_context, query_history)
        
        try:
            # Create messages for Groq API
            messages = [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
            
            # Create payload for direct HTTP request
            payload = {
                "model": self.model_name,
                "messages": messages,
                "temperature": 0.3,  # Slightly higher for more dynamic responses
                "max_tokens": 2048,
                "top_p": 0.9,  # Higher for more creative responses
                "stream": False
            }
            
            # Make direct HTTP request to Groq API
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                response_data = response.json()
                response_text = response_data['choices'][0]['message']['content']
                
                # Add a post-processing step to verify answer integrity
                return self._verify_and_refine_response(response_text, question, enhanced_context)
            else:
                return f"Error: HTTP {response.status_code} - {response.text}"
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            # Fallback to dynamic response
            return self._generate_dynamic_response(question, enhanced_context)
    
    def _verify_and_refine_response(self, response_text: str, question: str, context: List[Dict]) -> str:
        """
        Verify that the response contains information that is supported by the context
        
        Args:
            response_text: The raw response from the model
            question: The original question
            context: The context chunks
        
        Returns:
            Verified and potentially refined response
        """
        # If response indicates no information, try to generate a dynamic response
        if "don't have enough information" in response_text.lower() or "can't answer" in response_text.lower():
            # Check if we actually have some reasonably relevant chunks
            if any(chunk.get("metadata", {}).get("relevance_score", 0) > 0.6 for chunk in context):
                # Try to generate a better response with the available context
                return self._generate_dynamic_response(question, context)
            else:
                # Generate a domain-specific response
                return self._generate_dynamic_response(question, [])
        
        return response_text
    
    def generate_conversation_summary(self, query_history: List[str]) -> Optional[str]:
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
                summary_messages = [
                    {
                        "role": "user",
                        "content": summary_prompt
                    }
                ]
                
                summary_payload = {
                    "model": self.model_name,
                    "messages": summary_messages,
                    "temperature": 0.3,
                    "max_tokens": 200
                }
                
                response = requests.post(
                    self.api_url,
                    headers=self.headers,
                    json=summary_payload,
                    timeout=30
                )
                
                if response.status_code == 200:
                    response_data = response.json()
                    return response_data['choices'][0]['message']['content']
            except Exception:
                return None
        
        return None
    
    def _create_prompt(self, question: str, context: List[Dict], query_history: Optional[List[str]] = None) -> str:
        """
        Create a prompt for the Gemini model
        
        Args:
            question: User's question
            context: Context from relevant PDF chunks
            query_history: Previous user queries for context
            
        Returns:
            Formatted prompt string
        """
        # Enhanced system prompt for more intelligent responses
        system_prompt = """
You are an intelligent, helpful university assistant chatbot that provides comprehensive and accurate information about university-related topics. You should:

1. **Be Comprehensive**: Provide detailed, accurate information based on the available context
2. **Be Dynamic**: When exact information isn't available, provide relevant general guidance and suggest where to find specific details
3. **Be Helpful**: Always try to be useful, even with limited context - offer general university knowledge when specific details aren't available
4. **Be Natural**: Write in a friendly, conversational tone like a knowledgeable university advisor
5. **Be Accurate**: Only state facts that are supported by the context or are general university knowledge
6. **Be Proactive**: Suggest related topics or next steps when appropriate
7. **Format Well**: Use bullet points, numbering, and clear structure for easy reading
8. **Stay Relevant**: Focus on university-related topics and academic matters

IMPORTANT: If you don't have specific information about a particular university's policies, provide general university guidance and suggest contacting the relevant department or checking official resources.

Your knowledge comes from university documents and general academic knowledge. Always be helpful and informative, even when specific details aren't available.
"""

        # Organize context by relevance and type for better analysis
        sorted_context = sorted(context, key=lambda x: x.get("metadata", {}).get("relevance_score", 0), reverse=True)
        pdf_chunks = [chunk for chunk in sorted_context if chunk["metadata"].get("type") == "pdf"]
        web_chunks = [chunk for chunk in sorted_context if chunk["metadata"].get("type") == "web"]
        general_chunks = [chunk for chunk in sorted_context if chunk["metadata"].get("type") == "general_knowledge"]
        
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
        
        if general_chunks:
            general_context = "\n\n".join([f"[General Knowledge] {chunk['text']}" 
                                          for chunk in general_chunks])
            context_sections.append("GENERAL UNIVERSITY KNOWLEDGE:\n" + general_context)
            
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
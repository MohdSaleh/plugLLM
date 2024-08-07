from src.base import PlugBase
from src.chromadb import ChromaDB_VectorStore
from src.openai.openai_chat import OpenAI, OpenAI_Chat
# import PyPDF2
import re
import json

class PlugLLM_Base(ChromaDB_VectorStore, OpenAI_Chat):
    def __init__(self, config=None, client=None, allow_llm_to_see_data=True):
        ChromaDB_VectorStore.__init__(self, config=config)
        OpenAI_Chat.__init__(self, config=config, client=client)
        self.allow_llm_to_see_data = allow_llm_to_see_data

    def add_item(self, collection_key: str, content: str, content_type: str, **kwargs):
        return super().add_item(collection_key, content, content_type, **kwargs)

    def get_related_items(self, collection_key: str, question: str, **kwargs):
        return super().get_related_items(collection_key, question, **kwargs)

    def add_collection(self, key: str, name: str, n_results: int = 10):
        return super().add_collection(key, name, n_results)

    def remove_collection(self, key: str):
        return super().remove_collection(key)

    def list_collections(self):
        return super().list_collections()

    def get_training_data(self, **kwargs):
        return super().get_training_data(**kwargs)

    def remove_training_data(self, id: str, **kwargs):
        return super().remove_training_data(id, **kwargs)
    
    # Methods for managing ideas
    def get_related_ideas(self, question: str, **kwargs) -> list:
        raise NotImplementedError("This method is not implemented in this version of PlugRAG")

    def add_idea(self, idea: str, **kwargs) -> str:
        raise NotImplementedError("This method is not implemented in this version of PlugRAG")

    # Methods for managing insights
    def get_related_insights(self, question: str, **kwargs) -> list:
        raise NotImplementedError("This method is not implemented in this version of PlugRAG")

    def add_insight(self, insight: str, **kwargs) -> str:
        raise NotImplementedError("This method is not implemented in this version of PlugRAG")

    # Methods for managing quotes
    def get_related_quotes(self, question: str, **kwargs) -> list:
        raise NotImplementedError("This method is not implemented in this version of PlugRAG")

    def add_quote(self, quote: str, **kwargs) -> str:
        raise NotImplementedError("This method is not implemented in this version of PlugRAG")

    # Methods for managing habits
    def get_related_habits(self, question: str, **kwargs) -> list:
        raise NotImplementedError("This method is not implemented in this version of PlugRAG")

    def add_habit(self, habit: str, **kwargs) -> str:
        raise NotImplementedError("This method is not implemented in this version of PlugRAG")

    # Methods for managing facts
    def get_related_facts(self, question: str, **kwargs) -> list:
        raise NotImplementedError("This method is not implemented in this version of PlugRAG")

    def add_fact(self, fact: str, **kwargs) -> str:
        raise NotImplementedError("This method is not implemented in this version of PlugRAG")

    # Methods for managing references
    def get_related_references(self, question: str, **kwargs) -> list:
        raise NotImplementedError("This method is not implemented in this version of PlugRAG")

    def add_reference(self, reference: str, **kwargs) -> str:
        raise NotImplementedError("This method is not implemented in this version of PlugRAG")

    # Methods for managing recommendations
    def get_related_recommendations(self, question: str, **kwargs) -> list:
        raise NotImplementedError("This method is not implemented in this version of PlugRAG")

    def add_recommendation(self, recommendation: str, **kwargs) -> str:
        raise NotImplementedError("This method is not implemented in this version of PlugRAG")


    def log(self, message: str):
        print(message)

    # def extract_text_from_pdf(self, pdf_path):
    #     text = ""
    #     try:
    #         with open(pdf_path, 'rb') as file:
    #             pdf_reader = PyPDF2.PdfReader(file)
    #             for page in pdf_reader.pages:
    #                 text += page.extract_text()
    #     except Exception as e:
    #         print(f"An error occurred: {str(e)}")
    #         return None
    #     print("EXTRACTED TEXT FROM PDF", text[:100])
    #     return text

    # def extract_wisdom_info(self, pdf_context):
    #     # self.max_tokens = 32000
    #     wisdom_pre = """# IDENTITY and PURPOSE\n\nYou extract surprising, insightful, and interesting information from text content. You are interested in insights related to the purpose and meaning of life, human flourishing, the role of technology in the future of humanity, artificial intelligence and its affect on humans, memes, learning, reading, books, continuous improvement, and similar topics.\n\nTake a step back and think step-by-step about how to achieve the best possible results by following the steps below.\n\n# STEPS\n\n- Extract a summary of the content in 25 words, including who is presenting and the content being discussed into a section called SUMMARY.\n\n- Extract 20 to 50 of the most surprising, insightful, and/or interesting ideas from the input in a section called IDEAS:. If there are less than 50 then collect all of them. Make sure you extract at least 20.\n\n- Extract 10 to 20 of the best insights from the input and from a combination of the raw input and the IDEAS above into a section called INSIGHTS. These INSIGHTS should be fewer, more refined, more insightful, and more abstracted versions of the best ideas in the content. \n\n- Extract 15 to 30 of the most surprising, insightful, and/or interesting quotes from the input into a section called QUOTES:. Use the exact quote text from the input.\n\n- Extract 15 to 30 of the most practical and useful personal habits of the speakers, or mentioned by the speakers, in the content into a section called HABITS. Examples include but aren't limited to: sleep schedule, reading habits, things they always do, things they always avoid, productivity tips, diet, exercise, etc.\n\n- Extract 15 to 30 of the most surprising, insightful, and/or interesting valid facts about the greater world that were mentioned in the content into a section called FACTS:.\n\n- Extract all mentions of writing, art, tools, projects and other sources of inspiration mentioned by the speakers into a section called REFERENCES. This should include any and all references to something that the speaker mentioned.\n\n- Extract the most potent takeaway and recommendation into a section called ONE-SENTENCE TAKEAWAY. This should be a 15-word sentence that captures the most important essence of the content.\n\n- Extract the 15 to 30 of the most surprising, insightful, and/or interesting recommendations that can be collected from the content into a section called RECOMMENDATIONS.\n\n# OUTPUT INSTRUCTIONS\n\n- Only output Markdown.\n\n- Write the IDEAS bullets as exactly 15 words.\n\n- Write the RECOMMENDATIONS bullets as exactly 15 words.\n\n- Write the HABITS bullets as exactly 15 words.\n\n- Write the FACTS bullets as exactly 15 words.\n\n- Write the INSIGHTS bullets as exactly 15 words.\n\n- Extract at least 25 IDEAS from the content.\n\n- Extract at least 10 INSIGHTS from the content.\n\n- Extract at least 20 items for the other output sections.\n\n- Do not give warnings or notes; only output the requested sections.\n\n- You use bulleted lists for output, not numbered lists.\n\n- Do not repeat ideas, quotes, facts, or resources.\n\n- Do not start items with the same opening words.\n\n- Ensure you follow ALL these instructions when creating your output.\n\n# INPUT\n\nINPUT:\n"""
    #     llm_response = self.submit_prompt([self.system_message(wisdom_pre), self.user_message(pdf_context)])
    #     print("LLM_RES", llm_response)
    #     knowledge_json = self.extract_knowledge_as_json(llm_response)
    #     print(knowledge_json)
    #     return knowledge_json

    # def extract_knowledge_as_json(self, llm_response: str) -> str:
    #     # If the llm_response contains a markdown code block, with or without the json tag, extract the json from it
    #     json_match = re.search(r"```json\n(.*)```", llm_response, re.DOTALL)
    #     if json_match:
    #         self.log(f"Output from LLM: {llm_response} \nExtracted JSON: {json_match.group(1)}")
    #         return json_match.group(1)

    #     json_match = re.search(r"```(.*)```", llm_response, re.DOTALL)
    #     if json_match:
    #         self.log(f"Output from LLM: {llm_response} \nExtracted JSON: {json_match.group(1)}")
    #         return json_match.group(1)

    #     return llm_response

    def attach_knowledge(self, pdf_path):
        pass
        # pdf_context = self.extract_text_from_pdf(pdf_path)
        # wisdom_json = self.extract_wisdom_info(pdf_context)
        # TO DO
    
    def easy_prompt(self, prompt):
        # self.max_tokens = 32000
        query=prompt
        temp_list = self.get_focused_related_items(collection_key="prompt_templates", focus="use_case", question=query, n_results=1)
        pre = temp_list[0]
        system_prompt_refine = pre['pre-prompt']
        print("PRE: ", pre['use_case'])
        llm_response = self.submit_prompt([self.system_message(system_prompt_refine), self.user_message(query)])
        return llm_response

    @classmethod
    def create_plugLLM(cls, llm_model, client):
        config = {
            "path": "src/PlugLLM_Knowledge",
            "client": "persistent",
            "model": llm_model,
            "collections": {
                'prompt_templates': {'name': 'Prompt_Templates', 'n_results': 1, 'content_heads': ['template_id', 'template_text', 'use_case'], 'main_content_type': 'Text'},
                'knowledge_base': {'name': 'The_ONE_Knowledge', 'n_results': 5, 'content_heads': ['knowledge_name', 'INFO', 'IDEAS', 'INSIGHTS', 'QUOTES', 'HABITS', 'FACTS', 'REFERENCES', 'RECOMMENDATIONS'], 'main_content_type': 'Text'},
            }
        }
        return cls(config=config, client=client)
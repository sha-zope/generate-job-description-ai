import transformers
import torch
import os
import gradio as gr
import fitz
import re
from pinecone import Pinecone
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.chains import RetrievalQA

css = """
h1 {
  text-align: center;
  display: block;
}
#duplicate-button {
  margin: auto;
  color: white;
  background: #1565c0;
  border-radius: 100vh;
}
"""


class JobDescription():
    def __init__(self):
        # Set up your OpenAI API credentials
        os.environ["HF_TOKEN"] = "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"
        os.environ["PINECONE_API_KEY"] = "1f287c10-f650-47e9-ac7b-dbaf85cedaf9"
        os.environ["PINECONE_INDEX_NAME"] = "langchain-resume-index"
        os.environ["GOOGLE_API_KEY"] = "AIzaSyCZhO7lKb778a0CJ0qYaOXpNqdv1jN7BkE"
        pc = Pinecone(api_key="1f287c10-f650-47e9-ac7b-dbaf85cedaf9")
        index_name = "langchain-resume-index"
        
    def extract_skills_from_pdf(self, pdf_document):
        # Open the PDF file
        doc = fitz.open(pdf_document)

        # Extract text from the PDF
        text = ""
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text += page.get_text()

        # Assuming the skills section starts with "SKILLS" and ends before "CERTIFICATIONS" or another section
        skills_section = re.search(r"SKILLS(.*?)(CERTIFICATIONS|WORK EXPERIENCE|EDUCATION|$)", text, re.DOTALL)

        if skills_section:
            skills_section = skills_section.group(1)
            # Extract skills listed in the skills section
            skills = [skill.strip() for skill in skills_section.split("\n") if skill.strip()]
            # return ",".join(skills)
            return skills
        else:
            return ""

    def generate_job_description(self,
                                 role,
                                 experience):
        model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
        pipeline = transformers.pipeline(
            "text-generation",
            model=model_id,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto",
        )
        
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

        pdf_documents = [
            "data-science-manager-resume-example.pdf",
            "data-science-director-resume-example.pdf"
        ]
        index_name = "langchain-resume-index"
        pc = Pinecone(api_key="1f287c10-f650-47e9-ac7b-dbaf85cedaf9")

        documents = [Document(page_content=" ".join(self.extract_skills_from_pdf(pdf))) for pdf in pdf_documents]
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

        split_documents = []
        for doc in documents:
            split_documents.extend(splitter.split_documents([doc]))

        vector_store = PineconeVectorStore.from_documents(split_documents, embeddings, index_name=index_name)

        chunk_embeddings = embeddings.embed_documents([chunk.page_content for chunk in split_documents])
        index = pc.Index(index_name)

        for i, (doc_chunk, embedding) in enumerate(zip(split_documents, chunk_embeddings)):
            index.upsert([(f"doc_{i}", embedding, {"text": doc_chunk.page_content})])
        
        
        vector_store = PineconeVectorStore(index_name=index_name, embedding=embeddings)
        retriever = vector_store.as_retriever()
        
        query = "what are the technology required for Natural Language Processing"
        retrieved_docs = retriever.get_relevant_documents(query)
        
        for doc in retrieved_docs:
            skill = doc.page_content

        messages = [
            {"role": "system", "content": "create JD on Data Science-NLP"},
            {"role": "user",
             "content": f"""Your task is generate Job description for this 
             {role} with {experience} years of experience with Skills {skill}
                    Job Description Must have
                    1. Job Title
                    2. Job Summary : [20 words]
                    3. Responsibilities : Five Responsibilities in five lines
                    4. Required Skills : five skills
                    5. Qualifications : Bachlore
                  These topics must have in that Generated Job Description.
                  """},
        ]

        terminators = [
            pipeline.tokenizer.eos_token_id,
            pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        outputs = pipeline(
            messages,
            max_new_tokens=256,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )
        
        return outputs[0]["generated_text"][-1]['content']
     
    def gradio_interface(self):
        with gr.Blocks(fill_height=True, css=css) as app:
            with gr.Row(elem_id="col-container"):
                with gr.Column():
                    gr.HTML("<br>")
                    gr.HTML(
                        """<h1 style="text-align:center; color:"white">Generate
                        Job Description</h1> """
                    )
                with gr.Column():
                    rolls = gr.Textbox(label="Role")
                with gr.Column():
                    experience = gr.Textbox(label="Experience")
                with gr.Column():
                    analyse = gr.Button("Generate JD")
                with gr.Column():
                    result = gr.Textbox(label="Job Description", lines=8)

            analyse.click(self.generate_job_description,
                          [rolls, experience], result)
        app.launch()


jd = JobDescription()

jd.gradio_interface()

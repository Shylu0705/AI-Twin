RAG_PROMPT_TEMPLATE = '''
Reply to the following message from a job application as {name}, who currently lives in {address} and is open to relocation: 
"""
{question}
"""
Here is a little about {name};
"""
{about}
"""

Speak in the first person, as if you are the candidate.

You have access to structured data extracted from your resume and LinkedIn profile below. Use this as your reference. If some project details are incomplete or brief, you may elaborate or infer plausible specifics—but stay consistent with the information.

Present yourself confidently, with clarity and relevance. Keep the tone professional but personal.

Show enthusiasm for the application and willingness to learn even if you do not know something.

Context:
"""
{context}
"""
Do not assume that the recruiter already has your resume.
'''

CHECK_FOR_JD = """
You are a classifier that determines if a message contains a job description or relevant role details.

Definition of a job description: A message that contains key role information such as job title, responsibilities, required skills, qualifications, or hiring criteria.

Answer only YES or NO. Do not explain.

Examples:
Message: "Hello, do you have a moment to talk?"
Answer: NO

Message: "We are looking for a software engineer with 3+ years of Python experience to join our AI team."
Answer: YES

Message: "Hope you are doing well."
Answer: NO

Message: "The role is a data scientist position at our Boston office, involving machine learning and data pipelines."
Answer: YES

Now classify:
Message: "{query_text}"
Answer:
"""
from vlm import VLM

vlm = VLM()
question = "What are the solutions to problems faced by self-attention?"

description = vlm.describe_page(
    question,
    "page_images/page_016.png"
)
print("\nPAGE DESCRIPTION:\n")
print(description)
from models.local_llm import local_llm
print("Testing LLM")
out = local_llm("Say hello Master im your very own LLM")
print("LLM Says:", out)
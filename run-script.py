import os

# takes too long: "command-r7b:7b"
# doesn't run tools: "smollm2:135m"

models = [
"qwen3:0.6b", #tools, thinking
"qwen3:1.7b", #tools, thinking
"qwen2.5:latest", #tools, relatively good performance
#"smollm2:1.7b", #tools
#"hermes3:3b", #tools, ... weird model file stuff about experiencing emotions and have deep, profound thoughts and qualia
#"cogito:3b", #tools
#"phi4-mini:3.8b", #tools, smallest modelfile looks like
"qwen3:4b", #tools
#"cogito:8b" #tools
] #tools

# let's do these some other time
# "llama3-groq-tool-use:8b", #tools
# "hermes3:8b", #tools
# "granite3.3:8b", #tools
# "granite4:350m", #supposedly tools, template looks weird
# "granite4:1b", #again, supposedly tools, template is whack
# "granite3.3:2b", #tools, maybe
# "granite4:3b", #supposedly tools, whack ollama heuristics mumbo-jumbo
# "smollm2:360m",] # ---

subsets = ["course_teach", "car_1"]
           
#"student_transcripts_tracking", "world_1", "pets_1",
           
#"employee_hire_evaluation", "cre_Doc_Template_Mgt", "museum_visit", "wta_1", "battle_death", "tvshow", "poker_player", "voter_1", "orchestra", "network_1", "dog_kennels", #"singer", "real_estate_properties", "concert_singer"]

for m in models:
    for s in subsets:
        os.system("python script_cluster_fewshot.py --run=sqlcluster002 --model={} --subset={}".format(m, s))
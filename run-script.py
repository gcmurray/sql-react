import os
models = ["deepseek-r1:8b",
"deepseek-r1:7b",
"deepseek-r1:1.5b",
"command-r7b:7b",
"llama3-groq-tool-use:8b",
"granite4:3b",
"granite4:1b",
"granite4:350m",
"hermes3:8b",
"hermes3:3b",
"phi4-mini:3.8b",
"granite3.3:8b",
"granite3.3:2b",
"cogito:8b",
"cogito:3b",
"smollm2:1.7b",
"smollm2:360m",
"smollm2:135m",
"qwen3:4b",
"qwen3:1.7b",
"qwen3:0.6b"]

subsets = ["concert_singer", "pets_1", "car_1", "flight_2", "employee_hire_evaluation", "cre_Doc_Template_Mgt", "course_teach", "museum_visit", "wta_1", "battle_death", "student_transcripts_tracking", "tvshow", "poker_player", "voter_1", "world_1", "orchestra", "network_1", "dog_kennels", "singer", "real_estate_properties"]

for m in models:
    for s in subsets:
        os.system("python script_zeroshot.py --run=run001 --model={} --subset={}".format(m, s))
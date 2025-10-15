import streamlit as st
import pandas as pd
import model_run.counterparty_analysis as cp_analysis_agent 
import os
import pandas as pd
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import json
from PIL import Image

# Sidebar navigation
st.sidebar.title("Skill Matching Portal")
page = st.sidebar.radio("Go to", ["Welcome", "Skill Matching", "Data Ind KvK"])

# Welcome Page
if page == "Welcome":
	st.title("Skill Matching Portal")
	st.markdown("""
				Welcome to the Skill Matching Portal. Use the sidebar to navigate.

				**We label kvk with skills with custom labels:**

			 Finally, we provide insights into potential Companies.
				""")
	st.markdown("Trained on small dataset, large training in-progress :)")
	ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
	# # # st.write(ROOT_DIR)
	# # bs = (ROOT_DIR +"/webresource/images/TrainingGraph.png").replace("/", "\\")
	# # # st.write(bs)
	# st.image("./TrainingGraph.png", caption="", use_container_width =True)


# Counter Party Analysis Page
elif page == "Skill Matching":
	st.title("Skill Matching Portal")

	skills = st.text_area("Enter your skills (comma separated)", placeholder="python, pandas, sql")
	skills_input = str((skills or "").strip())

	batch_size = st.number_input("Enter Batch Size Max 1000", min_value=0, max_value=1000, value=10, step=1)
	st.write("You entered (type):", batch_size)
	if skills:
		skills_list = [s.strip() for s in skills.split(",") if s.strip()]
		st.write("Parsed skills:", skills_list)

	# Step3a: Upload CSV
	uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
	llm_agent = st.selectbox("Choose LLM Agent", ["gpt-3.5-turbo", "gpt-4", "ft:gpt-5-FineTuned-Rabopersonal::CLygEFH0"])

	if uploaded_file:
		# Step3e: Validate CSV
		try:
			df = pd.read_csv(uploaded_file)
			st.success("CSV file uploaded successfully.")
		except Exception as e:
			st.error(f"Error reading CSV: {e}")
			st.stop()

		# Step3b: Choose LLM agent
		

		# Step3c: Edit CSV
		edited_df = st.data_editor(df, num_rows="dynamic")

		# Step3d: Submit CSV
	if st.button("Submit for Skill Matching"):
		# Step3f: Send to OpenAI model
		try:
			for start in range(0, len(df), batch_size):
				st.write(f"Processing rows", start, "to", min(start + batch_size, len(df)) - 1)
				batch = edited_df.iloc[start:start + batch_size]
				prompt = f'''From the list of companies and skills below, which companies could this person work in? List only the matching companies. {skills_input} Also explain about company and why the skills match.
				{batch.to_csv(index=False)}
				'''
				if llm_agent.startswith("ft:"):
					st.subheader("Fine-tuned Model Output")
					result = cp_analysis_agent.run_counterparty_analysis(prompt, skills_input, "gpt-5")
					st.write(result)
				else:
					result = cp_analysis_agent.run_counterparty_analysis(prompt, skills_input, llm_agent)
					st.subheader("Model Output")
					# Step3g: Display result
					st.write(result)
				break
		except Exception as e:
			st.error(f"No file submitted")
			st.markdown("Error from Agent Run:")
			st.error(f"Error from OpenAI: {e}")

# Data Counter Party Analysis
elif page == "Data Ind KvK":
	st.title("Data Ind KvK")

	st.markdown("Data Ind KvK")
	ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

	df = pd.read_csv(os.path.join(ROOT_DIR, "test/data/kvk.csv"))
	edited_df = st.data_editor(df, num_rows="dynamic")


# 	st.markdown("Training Data Generated for Counter Party Analysis (by GPT-5):")
# 	df = pd.read_csv(os.path.join(ROOT_DIR, "test/data/transactions_generated_scenarios_gpt5.csv"))
# 	edited_df = st.data_editor(df, num_rows="dynamic")

# # Data Counter Party Analysis
# elif page == "Training Data Generation":
# 	st.title("Training Data For Counter Party Analysis")

# 	st.markdown("Training JSONL Data for Counter Party Analysis:")
# 	ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# 	with open(os.path.join(ROOT_DIR, "test/data/transactions_finetune.jsonl"), "r", encoding="utf-8") as f:
# 		for line in f:
# 			record = json.loads(line)
# 			print(record)
# 	st.write(record)

# 	st.markdown("Validation JSONL Data for Counter Party Analysis:")
# 	with open(os.path.join(ROOT_DIR, "test/data/transactions_training.jsonl"), "r", encoding="utf-8") as f:
# 		for line in f:
# 			record = json.loads(line)
# 			print(record)
# 	st.write(record)

# 	st.markdown("Validation JSONL Data for Counter Party Analysis:")
# 	with open(os.path.join(ROOT_DIR, "test/data/transactions_validation.jsonl"), "r", encoding="utf-8") as f:
# 		for line in f:
# 			record = json.loads(line)
# 			print(record)
# 	st.write(record)

st.markdown(
    """
    <style>
    .watermark {
        position: fixed;
        bottom: 40px;
        right: 30px;
        opacity: 1;
        font-size: 20px;
        color: #888;
        z-index: 9999;
        pointer-events: none;
        user-select: none;
    }
    </style>
    <div class="watermark">apz v.1.0.4</div>
    """,
    unsafe_allow_html=True

)



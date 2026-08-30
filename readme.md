# 1. Clone the repository

git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git

cd mplad

# 2. Create and activate a virtual environment

python -m venv venv

venv\Scripts\activate

source venv/bin/activate

# 3. Install dependencies

pip install -r requirements.txt

# 4. Set up environment key

cp .env.example .env

# Open .env and insert your GROQ_API_KEY (or enter it in the Streamlit sidebar)

# 5. Run the app

streamlit run app.py
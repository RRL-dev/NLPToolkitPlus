"""The script demonstrates the use of the LightChain library to interact with a SQL database.

Setup:
1. Install the required packages using `pip install lightchain`.
2. Set the OPENAI_API_KEY in a .env file.
3. Ensure your dataset configuration is available in `DATASET_CFG`.

Steps:
1. Cache data to SQL using FilesToSQL.
2. Connect to the SQLite database.
3. Create an instance of SqlAgent.
4. Interpret and explain natural language queries and log the explanations.
"""

from lightchain.connector import SQLDatabase
from lightchain.connector.sql import FilesToSQL
from lightchain.modules import SqlAgent
from lightchain.utils import DATASET_CFG, LOGGER

# Step 1: Cache data to SQL
FilesToSQL(config=DATASET_CFG).cache_to_sql()

# Step 2: Connect to the SQL database
db: SQLDatabase = SQLDatabase.from_uri(database_uri="sqlite:///resources/database/data.db")

# Step 3: Create an instance of the SQL agent
agent = SqlAgent(db=db)

# Step 4: Define natural language queries and get explanations
NL_QUERY = "which type of data we have at our data?"
EXPLANATION: str = agent.interpret_and_explain(nl_query=NL_QUERY)
LOGGER.info(msg=EXPLANATION)

NL_QUERY = "what is the average age?"
EXPLANATION = agent.interpret_and_explain(nl_query=NL_QUERY)
LOGGER.info(msg=EXPLANATION)

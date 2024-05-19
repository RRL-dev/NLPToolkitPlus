<div align="center">
  <img src="lightchain/assets/logo.png" alt="uim" width="400" height="auto">
</div>


<div align="center">

# LightChain

**Welcome to LightChain**, a lightweight version of LangChain designed to provide efficient and straightforward capabilities for natural language processing. LightChain is tailored for use with machine learning agents and integrates seamlessly with various data sources to enable powerful querying and database management.

</div>

## Features

- **Text-to-SQL Conversion**: Effortlessly convert natural language queries to SQL using the `SqlAgent` class.
- **Text-to-Elasticsearch**: Integrate natural language queries with Elasticsearch (To-do).
- **Text-to-MongoDB**: Integrate natural language queries with MongoDB (To-do).
- **Database Management**: Easily create and manage databases using the `FilesToSQL` utility.
- **Support for Multiple Data Sources**: Integrate with different datasets including Chinook, cancer, and diabetes data.

## Getting Started

### Prerequisites

Ensure you have the following installed:

- Python 3.10 or higher
- Required Python packages (install via `pyproject.toml`)

### Installation

Clone the repository and navigate to the project directory:

```bash
git clone https://github.com/RRL-dev/lightchain.git
cd lightchain
```

Install the required packages:

```bash
pip install -e .
```

### Directory Structure

- `lightchain/modules/agent/sql/base.py`: Contains the `SqlAgent` class for text-to-SQL conversion.
- `resources/database/Chinook.db`: Sample database used for SQL queries.
- `resources/data`: Directory containing cancer and diabetes data files.
- `superchain/connector/sql/create.py`: Contains the `FilesToSQL` class for creating databases from data files.
- `superchain/cfg/dataset/base.yaml`: YAML configuration file for setting up the `FilesToSQL` utility.

## Usage

### Text-to-SQL Conversion

The `SqlAgent` class in `lightchain/modules/agent/sql/base.py` allows you to convert natural language queries into SQL. Here’s how to use it:

```python
from lightchain.modules.agent.sql.base import SqlAgent
from superchain.connector import SQLDatabase

# Initialize the database
db = SQLDatabase.from_uri("sqlite:///resources/database/Chinook.db")

# Create an instance of SqlAgent
agent = SqlAgent(db=db)

# Perform a natural language query
nl_query = "Show me all tracks by composer 'Johann Sebastian Bach'"
sql_result = agent.interpret_and_explain(nl_query)
print(sql_result)
```

### Creating Databases from Data Files

Use the `FilesToSQL` class from `superchain/connector/sql/create.py` to create databases from your cancer and diabetes data:

1. Ensure your data files are placed in the `resources/data` directory.
2. Use the provided YAML configuration file `superchain/cfg/dataset/base.yaml` for setting up the database.

```python
from superchain.connector.sql.create import FilesToSQL
from types import SimpleNamespace
import yaml

# Load the YAML configuration
with open('superchain/cfg/dataset/base.yaml', 'r') as file:
    config = yaml.safe_load(file)
config = SimpleNamespace(**config)

# Initialize FilesToSQL with the configuration
files_to_sql = FilesToSQL(config=config)

# Cache all data files and create the database
files_to_sql.cache_to_sql()
```

### Example Usage with Cancer and Diabetes Data

The cancer and diabetes data are located in `resources/data`. To integrate these into a database (`data.db`), follow the instructions above to use `FilesToSQL`.

## To-do List

- [ ] Implement Text-to-Elasticsearch functionality.
- [ ] Implement Text-to-MongoDB functionality.
- [ ] Develop tools-lightchain for additional utility functions.

## Contributions

Contributions to LightChain are welcome! Please fork the repository and submit pull requests for any enhancements or bug fixes.

### License

This project is licensed under the Apache License 2.0.

## Contact

For any questions or issues, please open an issue on GitHub or contact the repository maintainer at reznik.roni@gmail.com.

---

Thank you for using LightChain! We hope it simplifies your natural language processing tasks and enhances your data management capabilities.
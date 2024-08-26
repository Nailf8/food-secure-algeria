import pytest
from langchain.callbacks.base import BaseCallbackHandler
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_openai import ChatOpenAI
from langchain_community.agent_toolkits import create_sql_agent

from scripts.sql_handler import BaseCallbackHandler

from dotenv import load_dotenv

load_dotenv()

# Get SQLite database from URI
db = SQLDatabase.from_uri("sqlite:///demo.db")

# Here we use GPT 3.5 
gpt35 = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# Agent creation
sql_agent = create_sql_agent(
    gpt35, db = db, agent_type = "openai-tools", verbose = True
)


class SQLHandler(BaseCallbackHandler):
    def __init__(self):
        self.sql_result = None

    def on_agent_action(self, action, **kwargs):
        """Run on agent action. if the tool being used is sql_db_query,
         it means we're submitting the sql and we can 
         record it as the final sql"""

        if action.tool == "sql_db_query":
            self.sql_result = action.tool_input


# Test cases
@pytest.mark.parametrize("user_query, expected_query", [
    (
        "Quel est le pourcentage de terres arables aménagées pour l’irrigation sur la periode 2017 2019?",
        "SELECT periode_2017_2019 FROM evolution_stabilite_sa WHERE sous_indicateur = 'Pourcentage des terres arables aménagées pour l’irrigation (%) (moyenne sur 3 ans)'"
    ),
    (
        "Comment la production totale du secteur halieutique a-t-elle évolué entre 1990 et 2006 ?",
        "SELECT Year, Production_totale FROM production_data WHERE Year BETWEEN 1990 AND 2006 ORDER BY Year"
    ),
    (
        "Quelle est la valeur la plus élevée et la plus basse de la production totale enregistrée entre 1990 et 2006, et en quelle année ?", 
        "SELECT MAX(Production_totale) AS Max_Production, Year FROM production_data WHERE Year BETWEEN 1990 AND 2006 UNION SELECT MIN(Production_totale) AS Min_Production, Year FROM production_data WHERE Year BETWEEN 1990 AND 2006"
    ),
    (
        "Quelle est la tendance de la consommation apparente des ressources halieutiques au fil des ans ? Y a-t-il des années où la consommation a fortement augmenté ou diminué ?", 
        "SELECT Year, Consommation_apparente FROM Production_and_Consumption_Statistics ORDER BY Year"
    ),
    (
        "Existe-t-il une corrélation entre l'augmentation de la population et la consommation apparente en produit halieutique ?", 
        "SELECT Year, Population, Consommation_apparente FROM Production_and_Consumption_Statistics ORDER BY Year LIMIT 10"
    )
])
def test_generate_query(user_query, expected_query):
    agent = create_sql_agent(gpt35, db = db, agent_type = "openai-tools", verbose = True)
    handler = SQLHandler()
    result = agent.invoke({"input": user_query}, {"callbacks": [handler]})
    generated_query = handler.sql_result['query']
    assert generated_query == expected_query
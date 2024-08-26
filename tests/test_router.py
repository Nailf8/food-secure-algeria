import pytest
import os
from dotenv import load_dotenv

from llama_index.core import PromptTemplate
from llama_index.llms.databricks import Databricks
from scripts.router import RouterOutputParser


from scripts.router import RouterOutputParser


# Load environment variables
load_dotenv()

# API key setup
DATABRICKS_TOKEN = os.environ.get('DATABRICKS_TOKEN')

llm = Databricks(
    model="databricks-meta-llama-3-1-70b-instruct",
    api_key=DATABRICKS_TOKEN,
    api_base="https://adb-7215147325717155.15.azuredatabricks.net/serving-endpoints",
)

choices = [
    "Utile pour répondre aux questions littéraires sur la sécurité alimentaire, l'agriculture, la pêche, politiques de soutien, l'innovation et les scénarios prospectifs.",
    "Utile pour répondre aux questions sur des données statistiques précises relatives à la sécurité alimentaire, l'agriculture, la pêche, les politiques de soutien, l'innovation et les scénarios prospectifs."
]

choices_str = "\n\n".join([f"{idx+1}. {choice}" for idx, choice in enumerate(choices)])

router_prompt_template = PromptTemplate(
    "Quelques choix sont donnés ci-dessous. Il est fourni dans une liste numérotée (de 1 à"
    " {num_choices}), où chaque élément de la liste correspond à un résumé.\n"
    "---------------------\n{context_list}\n---------------------\nEn utilisant uniquement les choix"
    " ci-dessus et sans connaissances préalables, retournez le choix le plus pertinent"
    " pour la question : '{query_str}'\n"

)

output_parser = RouterOutputParser()

# Test cases
@pytest.mark.parametrize("query_str, expected_choice", [
    (
        "Quelles sont les dernières innovations en matière de politique agricole?",
        1  # Supposons que le premier choix soit le plus pertinent
    ),
    (
        "Pouvez-vous fournir des données statistiques détaillées sur l'irrigation?",
        2  # Supposons que le deuxième choix soit le plus pertinent
    ),
    (
        "Quels sont les scénarios prospectifs pour l'avenir de l'agriculture?",
        1  # Supposons que le premier choix soit le plus pertinent
    ),
    (
        "Quelle est la part des terres arables aménagées pour l'irrigation?",
        2  # Supposons que le deuxième choix soit le plus pertinent
    ),
    (
        "Comment les politiques de soutien influencent-elles l'innovation agricole?",
        1  # Supposons que le premier choix soit le plus pertinent
    ),
    (
        "Avez-vous des statistiques sur l'utilisation des terres pour l'irrigation?",
        2  # Supposons que le deuxième choix soit le plus pertinent
    ),
    (
        "Quelles sont les nouvelles stratégies pour soutenir l'innovation agricole?",
        1  # Supposons que le premier choix soit le plus pertinent
    ),
    (
        "Pouvez-vous détailler le scénario idéal de la sécurité alimentaire en Algérie?",
        1  # Supposons que le deuxième choix soit le plus pertinent
    ),
    (
        "Quels sont les impacts des politiques sur les scénarios futurs de l'agriculture?",
        1  # Supposons que le premier choix soit le plus pertinent
    ),
    (
        "Quels sont les chiffres récents concernant l'irrigation des terres agricoles?",
        2  # Supposons que le deuxième choix soit le plus pertinent
    ),
])

def test_route_query(query_str, expected_choice):
    fmt_prompt = router_prompt_template.format(
        num_choices=len(choices),
        context_list=choices_str,
        query_str=query_str,
        max_outputs=1
    )
    fmt_json_prompt = output_parser.format(fmt_prompt)
    raw_output = llm.complete(fmt_json_prompt)
    generated_parsed = output_parser.parse(str(raw_output))
    
    # Assert the expected choice matches the actual choice
    assert generated_parsed.choice == expected_choice